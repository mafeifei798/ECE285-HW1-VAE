"""FID, Inception Score and reconstruction metrics for the trained VAE."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for ops MPS lacks
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image
from ignite.metrics import FID, InceptionScore

from data import ChestXrayDataset
from latent_density import fit_gaussian_mixture, sample_gaussian_mixture
from model import load_vae_checkpoint


def configure_runtime(threads=8):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def fid_from_features(real, fake):
    """FID with sample covariance (n-1); the SVD is computed in sample space."""
    real = torch.as_tensor(real).double().cpu()
    fake = torch.as_tensor(fake).double().cpu()
    mean_real, mean_fake = real.mean(0), fake.mean(0)
    a = (real - mean_real) / math.sqrt(len(real) - 1)
    b = (fake - mean_fake) / math.sqrt(len(fake) - 1)
    cross_trace = torch.linalg.svdvals(a @ b.T).sum()
    return float((mean_real - mean_fake).square().sum()
                 + a.square().sum() + b.square().sum() - 2 * cross_trace)


def inception_score(probabilities):
    probabilities = torch.as_tensor(probabilities).double().cpu()
    marginal = probabilities.mean(0)
    entropy = (marginal * torch.log(marginal + 1e-16)).sum()
    conditional = (probabilities * torch.log(probabilities + 1e-16)).sum(1).mean()
    return float(torch.exp(conditional - entropy))


def compute_metrics_from_features(features, splits=10):
    real_probs = torch.as_tensor(features["real_probabilities"])
    fake_probs = torch.as_tensor(features["fake_probabilities"])
    if not 2 <= splits <= len(fake_probs):
        raise ValueError("is-splits must be between 2 and the generated sample count")
    count = len(real_probs)
    scores = [inception_score(group) for group in torch.tensor_split(fake_probs, splits)]
    fid_metric = FID(num_features=real_probs.shape[1], feature_extractor=nn.Identity(), device="cpu")
    fid_metric.update((fake_probs[:count].double(), real_probs.double()))
    is_metric = InceptionScore(num_features=fake_probs.shape[1], feature_extractor=nn.Identity(), device="cpu")
    is_metric.update(fake_probs.double())
    return {
        "real_count": count, "fake_count_fid": count, "fake_count_is": len(fake_probs),
        "fid_ignite_probability_1000": float(fid_metric.compute()),
        "fid_torchvision_pool3_2048": fid_from_features(
            features["real_pool3"], features["fake_pool3"][:count]),
        "is_ignite_full": float(is_metric.compute()),
        "is_splits": {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores},
    }


def reference_metrics(features, split="test"):
    """Context for the scores: what real images and reconstructions achieve under the same
    protocol. FID uses the same sample count as the main FID."""
    real_probs, real_pool = features["real_probabilities"], features["real_pool3"]
    count = len(real_probs)
    metrics = {f"is_real_{split}": inception_score(real_probs),
               "is_generated_same_count": inception_score(features["fake_probabilities"][:count])}
    for name in ("train_real", "reconstruction"):
        if f"{name}_pool3" in features:
            metrics[f"fid_{name}_vs_{split}_pool3_2048"] = fid_from_features(
                real_pool, features[f"{name}_pool3"])
            metrics[f"fid_{name}_vs_{split}_probability_1000"] = fid_from_features(
                real_probs, features[f"{name}_probabilities"])
    return metrics


def tutorial_preprocess(pixels_u8):
    images = []
    for pixels in pixels_u8:
        image = Image.fromarray(pixels[0].numpy()).convert("RGB")
        image = image.resize((299, 299), Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255))
    return torch.stack(images)


class SharedInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True).eval()
        self.backbone.avgpool.register_forward_hook(self.capture_pool)
        self.requires_grad_(False)

    def capture_pool(self, module, inputs, output):
        self.pool = output.flatten(1)

    @torch.inference_mode()
    def forward(self, images):
        logits = self.backbone(images)
        return logits.softmax(1), self.pool


@torch.inference_mode()
def extract_features(extractor, pixels, batch_size, device, verbose=True):
    probabilities, pooled = [], []
    for offset in range(0, len(pixels), batch_size):
        images = tutorial_preprocess(pixels[offset:offset + batch_size]).to(device)
        probs, pool = extractor(images)
        probabilities.append(probs.cpu())
        pooled.append(pool.cpu())
        if verbose:
            print(f"Inception features: {min(offset + batch_size, len(pixels))}/{len(pixels)}", flush=True)
    return torch.cat(probabilities).numpy(), torch.cat(pooled).numpy()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_uint8(images):
    """Use the same floor quantization as the original evaluation."""
    return images.mul(255).clamp_(0, 255).to(torch.uint8)


def dataset_pixels(dataset):
    if isinstance(dataset, Subset):
        return dataset_pixels(dataset.dataset)[dataset.indices]
    return torch.from_numpy(np.array(dataset.images, copy=True)).unsqueeze(1)


def training_split(data_root, cache_dir, image_size, checkpoint):
    """The training images the VAE learned from, and its held-out ones (None if it has none)."""
    images = ChestXrayDataset(data_root, "train", image_size, cache_dir)
    holdout = list(checkpoint.get("holdout_indices") or [])
    held = set(holdout)
    seen = Subset(images, [i for i in range(len(images)) if i not in held])
    return seen, (Subset(images, holdout) if holdout else None)


@torch.inference_mode()
def encode_means(model, dataset, batch_size, device):
    model.eval()
    return torch.cat([model.encode(batch.to(device))[0].cpu().double()
                      for batch in DataLoader(dataset, batch_size=batch_size)]).numpy()


@torch.inference_mode()
def sample_latents(model, count, temperature, seed, device, mixture=None):
    """Latents from the model's prior, or from a fitted Gaussian mixture when given one."""
    if mixture is not None:
        return torch.from_numpy(sample_gaussian_mixture(mixture, count, temperature, seed)).float()
    epsilon = torch.randn(count, model.latent_dim, generator=torch.Generator().manual_seed(seed))
    return model.latent_from_base(temperature * epsilon.to(device)).cpu()


@torch.inference_mode()
def decode_images(model, z, batch_size, device):
    model.eval()
    return torch.cat([to_uint8(model.pixel_mean(model.decode(latent.to(device))).cpu())
                      for latent in z.split(batch_size)])


@torch.inference_mode()
def reconstruct_images(model, pixels_u8, batch_size, device):
    model.eval()
    reconstructed = []
    for batch in pixels_u8.split(batch_size):
        mu, _ = model.encode(batch.float().div(255).to(device))
        reconstructed.append(to_uint8(model.pixel_mean(model.decode(mu)).cpu()))
    return torch.cat(reconstructed)


@torch.inference_mode()
def reconstruction_metrics(model, dataset, device, batch_size):
    errors, similarities = [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        mu, _ = model.encode(batch.to(device))
        reconstructed = model.pixel_mean(model.decode(mu)).cpu()
        errors.extend((batch - reconstructed).square().mean((1, 2, 3)).tolist())
        similarities.extend(structural_similarity(
            x[0].numpy(), y[0].numpy(), data_range=1.0, gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False) for x, y in zip(batch, reconstructed))
    return {"mse": float(np.mean(errors)), "ssim": float(np.mean(similarities))}


@torch.inference_mode()
def evaluate(args):
    device = resolve_device(args.device)
    model, checkpoint = load_vae_checkpoint(args.checkpoint)
    model.to(device)
    seen = None
    if args.split == "holdout":
        seen, evaluation_data = training_split(args.data_root, args.cache_dir, model.image_size, checkpoint)
        if evaluation_data is None:
            raise ValueError("This checkpoint has no held-out training images")
    else:
        evaluation_data = ChestXrayDataset(args.data_root, "test", model.image_size, args.cache_dir)
    real = dataset_pixels(evaluation_data)
    if args.num_samples < len(real):
        raise ValueError("num-samples must be at least the evaluation-set size")
    if seen is None and (args.sampler == "gmm" or args.reference_baselines):
        seen, _ = training_split(args.data_root, args.cache_dir, model.image_size, checkpoint)
    mixture = None
    if args.sampler == "gmm":
        if args.mixture_file:
            with np.load(args.mixture_file, allow_pickle=False) as saved:
                mixture = SimpleNamespace(weights_=saved["weights"], means_=saved["means"],
                                          covariances_=saved["covariances"])
            if mixture.means_.shape[1] != model.latent_dim:
                raise ValueError("Saved mixture latent dimension differs from the model")
        else:
            latents = encode_means(model, seen, args.batch_size, device)
            mixture = fit_gaussian_mixture(latents, args.gmm_components, args.seed)
    if mixture is not None:
        np.savez_compressed(args.output_dir / "sampler.npz", weights=mixture.weights_,
                            means=mixture.means_, covariances=mixture.covariances_)
    z = sample_latents(model, args.num_samples, args.temperature, args.seed, device, mixture)
    generated = decode_images(model, z, args.batch_size, device)
    save_image(generated[:64].float().div(255), args.output_dir / "generated_samples.png", nrow=8, pad_value=1)
    reconstructed = reconstruct_images(model, real, args.batch_size, device)
    np.savez_compressed(args.output_dir / "evaluation_samples.npz",
                        real=real.numpy(), generated=generated.numpy(),
                        reconstructed=reconstructed.numpy())
    pairs = torch.stack((real[:16], reconstructed[:16]), 1).flatten(0, 1)
    save_image(pairs.float().div(255), args.output_dir / "reconstructions.png", nrow=8, pad_value=1)
    validation_data = ChestXrayDataset(args.data_root, "val", model.image_size, args.cache_dir)
    reconstruction = {f"{split}_{key}": value
                      for split, dataset in (("validation", validation_data), (args.split, evaluation_data))
                      for key, value in reconstruction_metrics(model, dataset, device, args.batch_size).items()}
    model.to("cpu")

    extractor = SharedInception().to(device).eval()
    real_probs, real_pool = extract_features(extractor, real, args.batch_size, device)
    fake_probs, fake_pool = extract_features(extractor, generated, args.batch_size, device)
    features = {"real_probabilities": real_probs, "fake_probabilities": fake_probs,
                "real_pool3": real_pool, "fake_pool3": fake_pool}
    if args.reference_baselines:
        # Reference distributions under the same feature and image preprocessing.
        chosen = np.sort(np.random.default_rng(args.seed).choice(
            len(seen), len(real), replace=False))
        train_real = dataset_pixels(Subset(seen, chosen.tolist()))
        for name, pixels in (("train_real", train_real), ("reconstruction", reconstructed)):
            probs, pool = extract_features(extractor, pixels, args.batch_size, device)
            features[f"{name}_probabilities"], features[f"{name}_pool3"] = probs, pool
    np.savez_compressed(args.output_dir / "evaluation_features.npz", **features)
    metrics = compute_metrics_from_features(features, args.is_splits)
    if args.reference_baselines:
        metrics["reference"] = reference_metrics(features, args.split)
    metrics.update(reconstruction)
    metrics["sampler"] = {"name": args.sampler, "temperature": args.temperature,
                          "gmm_components": len(mixture.weights_) if mixture is not None else None,
                          "fit_images": len(seen) if args.sampler == "gmm" else None}
    metrics["checkpoint_epoch"] = checkpoint["epoch"]
    metrics["model_config"] = model.model_config
    metrics["seed"] = args.seed
    metrics["evaluation_split"] = args.split
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--cache-dir")
    parser.add_argument("--split", default="test", choices=("test", "holdout"),
                        help="Use training holdout for selection, and test only after selection")
    parser.add_argument("--features", help="Compute scores from a saved evaluation_features.npz")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--sampler", default="prior", choices=("prior", "gmm"),
                        help="prior: the model's prior; gmm: a Gaussian mixture fitted to the "
                             "training posterior means")
    parser.add_argument("--gmm-components", type=int, default=10)
    parser.add_argument("--mixture-file", help="Reuse a mixture saved during holdout selection")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Scale of the sampling noise around the prior or mixture means")
    parser.add_argument("--reference-baselines", action="store_true",
                        help="Also score real training images and test reconstructions")
    args = parser.parse_args()
    if not args.features and not args.checkpoint:
        parser.error("Provide --checkpoint or --features")
    if args.batch_size < 1 or args.cpu_threads < 1 or args.gmm_components < 1:
        parser.error("batch-size, cpu-threads and gmm-components must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    configure_runtime(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            features = dict(saved)
        metrics = compute_metrics_from_features(features, args.is_splits)
        if "train_real_pool3" in features or "reconstruction_pool3" in features:
            metrics["reference"] = reference_metrics(features, args.split)
    else:
        metrics = evaluate(args)
    metrics["evaluation_split"] = args.split
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
