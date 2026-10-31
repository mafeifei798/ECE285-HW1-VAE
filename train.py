"""Train a VAE and select the checkpoint with the lowest validation ELBO loss."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for ops MPS lacks

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from data import ChestXrayDataset
from model import ConvVAE, elbo_loss
from validation_metrics import paired_reconstruction_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    # warn_only: nearest upsampling has no deterministic CUDA backward pass, and MPS
    # lacks some deterministic kernels. CPU results are unaffected.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_holdout(count, size, seed):
    """Indices of a fixed random subset of the de-duplicated training images."""
    if not 0 < size < count:
        raise ValueError(f"holdout must be between 1 and {count - 1}")
    return sorted(np.random.default_rng(seed).permutation(count)[:size].tolist())


def split_training_images(images, holdout):
    held = set(holdout)
    return Subset(images, [i for i in range(len(images)) if i not in held]), Subset(images, holdout)


@torch.inference_mode()
def fit_prior(model, dataset, batch_size, device):
    """Fit m=E[mu], C=Var(mu)+E[diag(exp(logvar))] using training images."""
    model.eval()
    means, variances = [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        mu, logvar = model.encode(batch.to(device))
        # Move to the CPU before casting: MPS has no float64.
        means.append(mu.cpu().double())
        variances.append(logvar.cpu().double().exp())
    mu, variance = torch.cat(means), torch.cat(variances)
    center = mu.mean(0)
    centered = mu - center
    covariance = centered.T @ centered / len(mu) + torch.diag(variance.mean(0))
    scale = torch.linalg.cholesky(covariance)
    model.prior_mean.copy_(center.to(model.prior_mean))
    model.prior_scale_tril.copy_(scale.to(model.prior_scale_tril))


@torch.no_grad()
def shift_images(images):
    """Random whole-pixel shifts of up to 5% of the width. Without interpolation the
    training targets keep their full sharpness; no flips."""
    size = images.shape[-1]
    limit = max(1, round(0.05 * size))
    padded = F.pad(images, (limit,) * 4, mode="replicate")
    offsets = torch.randint(0, 2 * limit + 1, (len(images), 2)).tolist()
    return torch.stack([padded[i, :, y:y + size, x:x + size] for i, (y, x) in enumerate(offsets)])


@torch.no_grad()
def augment_images(images, mode="affine"):
    """Small rotations, translations and scale changes (bilinear); no horizontal flips."""
    if mode == "shift":
        return shift_images(images)
    count = len(images)
    angle = (torch.rand(count, device=images.device) * 2 - 1) * math.radians(5)
    scale = 0.95 + 0.1 * torch.rand(count, device=images.device)
    theta = images.new_zeros(count, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = angle.cos() / scale
    theta[:, 0, 1] = -angle.sin() / scale
    theta[:, 1, 0] = angle.sin() / scale
    theta[:, :, 2] = (torch.rand(count, 2, device=images.device) * 2 - 1) * 0.1
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def run_epoch(model, loader, device, validation_seed, optimizer=None, grad_clip=0.0, augment=None):
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator().manual_seed(validation_seed)
    totals = {key: 0.0 for key in ("loss", "reconstruction", "kl")}
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training and augment:
                batch = augment_images(batch, augment)
            epsilon = None if training else torch.randn(
                len(batch), model.latent_dim, generator=generator).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(batch, epsilon)
            losses = elbo_loss(logits, batch, mu, logvar, model.likelihood,
                               model.observation_std, model.prior_mean, model.prior_scale_tril)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite loss")
            if training:
                losses["loss"].backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in losses.items():
                totals[key] += value.item() * len(batch)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


@torch.inference_mode()
def fitted_gaussian_latents(model, dataset, batch_size, device, epsilon):
    """Preview a means-only Gaussian sampler, separate from the training prior."""
    means = torch.cat([model.encode(batch.to(device))[0].cpu().double()
                       for batch in DataLoader(dataset, batch_size=batch_size)])
    centered = means - means.mean(0)
    covariance = centered.T @ centered / len(means)
    covariance += 1e-5 * torch.eye(means.shape[1], dtype=means.dtype)
    scale = torch.linalg.cholesky(covariance)
    return (epsilon.cpu().double() @ scale.T + means.mean(0)).float().to(device)


@torch.inference_mode()
def save_examples(model, validation_data, output, device, seed, training_data=None, batch_size=128):
    model.eval()
    generator = torch.Generator().manual_seed(seed + 200000)
    epsilon = torch.randn(64, model.latent_dim, generator=generator).to(device)
    samples = model.pixel_mean(model.decode(model.latent_from_base(epsilon)))
    save_image(samples, output / "prior_samples.png", nrow=8, pad_value=1)
    if training_data is not None:
        # A K=1, T=1 density-estimation preview; the final sweep can choose differently.
        latents = fitted_gaussian_latents(model, training_data, batch_size, device, epsilon)
        save_image(model.pixel_mean(model.decode(latents)), output / "fitted_samples.png",
                   nrow=8, pad_value=1)
    indices = torch.linspace(0, len(validation_data) - 1, min(16, len(validation_data))).long()
    originals = torch.stack([validation_data[int(i)] for i in indices]).to(device)
    mu, _ = model.encode(originals)
    reconstructions = model.pixel_mean(model.decode(mu))
    pairs = torch.stack((originals, reconstructions), dim=1).flatten(0, 1)
    save_image(pairs, output / "validation_reconstructions.png", nrow=8, pad_value=1)
    first_four = torch.cat((originals[:4], reconstructions[:4]))
    save_image(first_four, output / "validation_first4.png", nrow=4, pad_value=1)


def train(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() and not args.resume:
        raise FileExistsError(f"Choose a new output directory: {output}")
    if args.prior == "fitted_gaussian" and not (args.initialize_from or args.resume):
        raise ValueError("A fitted prior needs a trained parent checkpoint")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = ConvVAE(args.image_size, args.latent_dim, args.base_channels,
                    args.likelihood, args.observation_std, args.prior,
                    arch=args.arch, res_blocks=args.res_blocks).to(device)
    resumed = (torch.load(args.resume, map_location=device, weights_only=False)
               if args.resume else None)
    parent = (torch.load(args.initialize_from, map_location=device, weights_only=False)
              if args.initialize_from and not args.resume else None)
    if parent is not None:
        missing = model.load_state_dict(parent["model_state_dict"], strict=False)
        changing_prior = (args.prior == "fitted_gaussian"
                          and parent["model_config"].get("prior", "standard_normal") != "fitted_gaussian")
        allowed = {"prior_mean", "prior_scale_tril"} if changing_prior else set()
        if set(missing.missing_keys) != allowed or missing.unexpected_keys:
            raise ValueError("Parent checkpoint does not match this model; pass the parent's "
                             "--arch, --res-blocks, --base-channels and --latent-dim")

    training_images = ChestXrayDataset(args.data_root, "train", args.image_size, args.cache_dir)
    if resumed is not None:
        holdout = resumed.get("holdout_indices")
    elif parent is not None:
        # A fine-tuned child keeps its parent's held-out images out of training.
        holdout = parent.get("holdout_indices")
        if holdout is None and args.holdout:
            raise ValueError("The parent trained on every training image, so nothing is held out")
        if holdout is not None and args.holdout not in (0, len(holdout)):
            raise ValueError(f"The parent held out {len(holdout)} images; omit --holdout or match it")
    else:
        holdout = draw_holdout(len(training_images), args.holdout, args.seed) if args.holdout else None
    if holdout:
        training_data, validation_data = split_training_images(training_images, holdout)
    else:
        training_data = training_images
        validation_data = ChestXrayDataset(args.data_root, "val", args.image_size, args.cache_dir)
    reference_images64 = ChestXrayDataset(args.data_root, "train", 64, args.cache_dir)
    if len(reference_images64) != len(training_images):
        raise ValueError("64/128 caches must use the same training-image order")
    validation_reference64 = (Subset(reference_images64, holdout) if holdout else
                              ChestXrayDataset(args.data_root, "val", 64, args.cache_dir))
    print(f"{len(training_data)} training and {len(validation_data)} validation images "
          f"on {device}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
        if args.lr_schedule == "cosine" else None)
    if args.prior == "fitted_gaussian" and resumed is None:
        fit_prior(model, training_data, args.batch_size, device)
    shuffle_generator = torch.Generator()
    training_loader = DataLoader(training_data, batch_size=args.batch_size,
                                 shuffle=True, generator=shuffle_generator)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    start_epoch, best_loss, best_epoch, rows = 1, float("inf"), None, []
    if resumed is not None:
        saved = resumed["config"]
        for key in ("seed", "batch_size", "learning_rate"):
            if saved[key] != getattr(args, key):
                raise ValueError(f"Keep {key} unchanged when resuming")
        if saved.get("lr_schedule", "constant") != args.lr_schedule:
            raise ValueError("Keep lr-schedule unchanged when resuming")
        if saved.get("augment", False) != args.augment:
            raise ValueError("Keep augment unchanged when resuming")
        if args.augment and saved.get("augment_mode", "affine") != args.augment_mode:
            raise ValueError("Keep augment-mode unchanged when resuming")
        if args.lr_schedule == "cosine" and saved["epochs"] != args.epochs:
            raise ValueError("Keep epochs unchanged when resuming a cosine schedule")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_epoch = resumed["epoch"] + 1
        best_loss, best_epoch = resumed["best_val_loss"], resumed["best_epoch"]
        with (output / "epoch_metrics.csv").open() as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    end_epoch = min(args.epochs, args.stop_after) if args.stop_after else args.epochs
    for epoch in range(start_epoch, end_epoch + 1):
        started = time.perf_counter()
        seed_everything(args.seed + 100000 + epoch)
        shuffle_generator.manual_seed(args.seed + epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, training_loader, device, args.seed + 300000,
                             optimizer, args.grad_clip, args.augment_mode if args.augment else None)
        validation = run_epoch(model, validation_loader, device, args.seed + 300000)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch,
               **{f"train_{key}": value for key, value in training.items()},
               **{f"val_{key}": value for key, value in validation.items()},
               "seconds": time.perf_counter() - started, "learning_rate": learning_rate,
               "peak_cuda_gb": (torch.cuda.max_memory_reserved() / 2 ** 30
                                if device.type == "cuda" else 0.0)}
        rows.append(row)
        improved = validation["loss"] < best_loss
        if improved:
            best_loss, best_epoch = validation["loss"], epoch
        checkpoint = {"epoch": epoch, "model_config": model.model_config,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                      "holdout_indices": holdout,
                      "config": vars(args), "val_loss": validation["loss"],
                      "best_val_loss": best_loss, "best_epoch": best_epoch}
        with (output / "epoch_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if improved:
            torch.save(checkpoint, output / "best.pt.tmp")
            (output / "best.pt.tmp").replace(output / "best.pt")
        torch.save(checkpoint, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        print(f"Epoch {epoch:3d}: train {training['loss']:.3f}, val {validation['loss']:.3f}, "
              f"lr {learning_rate:.2e}, {row['seconds']:.1f}s", flush=True)
        if args.save_every and (epoch % args.save_every == 0 or epoch == end_epoch):
            examples = output / f"epoch_{epoch:04d}"
            examples.mkdir(exist_ok=True)
            save_examples(model, validation_data, examples, device, args.seed)
            quality = paired_reconstruction_metrics(
                model, validation_data, validation_reference64, device, args.batch_size)
            quality_file = output / "validation_quality.csv"
            fresh = not quality_file.exists()
            with quality_file.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "mse", "ssim", "mse64", "ssim64"])
                if fresh:
                    writer.writeheader()
                writer.writerow({"epoch": epoch, **quality})
            compact = {key: checkpoint[key] for key in
                       ["epoch", "model_config", "model_state_dict", "holdout_indices", "val_loss", "config"]}
            compact["validation_quality"] = quality
            torch.save(compact, examples / "model.pt")
            print(f"Holdout epoch {epoch}: native MSE {quality['mse']:.6f}, SSIM {quality['ssim']:.4f}; "
                  f"common64 MSE {quality['mse64']:.6f}, SSIM {quality['ssim64']:.4f}", flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_examples(model, validation_data, output, device, args.seed)
    print(f"Stopped at epoch {end_epoch} of {args.epochs} planned cosine epochs", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=64, choices=(32, 64, 128, 256))
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--arch", default="plain", choices=("plain", "resnet"))
    parser.add_argument("--res-blocks", type=int, default=1,
                        help="Residual blocks per resolution (resnet only)")
    parser.add_argument("--likelihood", default="gaussian",
                        choices=("bernoulli", "continuous_bernoulli", "gaussian"))
    parser.add_argument("--observation-std", type=float, default=0.05)
    parser.add_argument("--prior", default="standard_normal", choices=("standard_normal", "fitted_gaussian"))
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="Optional checkpoint boundary without changing the planned cosine schedule")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-schedule", default="constant", choices=("constant", "cosine"))
    parser.add_argument("--min-learning-rate", type=float, default=1e-5,
                        help="Final learning rate of the cosine schedule")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true", help="Augment training images")
    parser.add_argument("--augment-mode", default="affine", choices=("affine", "shift"),
                        help="affine: bilinear rotation/scale/translation (blurs targets); "
                             "shift: whole-pixel translation up to 5%% of the width")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save fixed samples and holdout reconstruction metrics at this epoch interval")
    parser.add_argument("--holdout", type=int, default=0,
                        help="Hold out this many training images for validation and sampler "
                             "tuning; 0 uses the official 16-image val split")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.latent_dim, args.base_channels, args.cpu_threads) < 1:
        parser.error("epochs, batch-size, latent-dim, base-channels and cpu-threads must be positive")
    if args.learning_rate <= 0 or args.observation_std <= 0:
        parser.error("learning-rate and observation-std must be positive")
    if not 0 <= args.stop_after <= args.epochs:
        parser.error("stop-after must be between zero and epochs")
    if min(args.res_blocks, args.holdout, args.min_learning_rate, args.grad_clip, args.save_every) < 0:
        parser.error("res-blocks, holdout, min-learning-rate and grad-clip must be non-negative")
    return args


if __name__ == "__main__":
    train(parse_args())
