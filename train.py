"""Train the residual VAE with a fixed training holdout."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from data import ChestXrayDataset
from model import ConvVAE, elbo_loss
from validation_metrics import paired_reconstruction_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, validation_seed, optimizer=None, grad_clip=5.0):
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator().manual_seed(validation_seed)
    totals = {key: 0.0 for key in ("loss", "reconstruction", "kl")}
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            epsilon = None if training else torch.randn(
                len(batch), model.latent_dim, generator=generator).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(batch, epsilon)
            losses = elbo_loss(logits, batch, mu, logvar, model.observation_std)
            if training:
                losses["loss"].backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in losses.items():
                totals[key] += value.item() * len(batch)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


@torch.inference_mode()
def save_examples(model, validation_data, output, device, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed + 200000)
    z = torch.randn(64, model.latent_dim, generator=generator).to(device)
    save_image(model.pixel_mean(model.decode(z)), output / "prior_samples.png",
               nrow=8, pad_value=1)
    indices = torch.linspace(0, len(validation_data) - 1, min(16, len(validation_data))).long()
    originals = torch.stack([validation_data[int(i)] for i in indices]).to(device)
    mu, _ = model.encode(originals)
    reconstructions = model.pixel_mean(model.decode(mu))
    pairs = torch.stack((originals, reconstructions), dim=1).flatten(0, 1)
    save_image(pairs, output / "validation_reconstructions.png", nrow=8, pad_value=1)


def train(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Choose a new output directory: {output}")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = ConvVAE(args.latent_dim, args.base_channels, args.observation_std).to(device)

    images = ChestXrayDataset(args.data_root, "train", 64, args.cache_dir)
    if not 0 < args.holdout < len(images):
        raise ValueError("holdout must leave images for both training and validation")
    holdout = sorted(np.random.default_rng(args.seed).permutation(len(images))[:args.holdout].tolist())
    held = set(holdout)
    training_data = Subset(images, [i for i in range(len(images)) if i not in held])
    validation_data = Subset(images, holdout)
    print(f"{len(training_data)} training and {len(validation_data)} holdout images on {device}", flush=True)

    shuffle_generator = torch.Generator()
    training_loader = DataLoader(training_data, batch_size=args.batch_size,
                                 shuffle=True, generator=shuffle_generator)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    best_loss = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        seed_everything(args.seed + 100000 + epoch)
        shuffle_generator.manual_seed(args.seed + epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, training_loader, device, args.seed + 300000,
                             optimizer, args.grad_clip)
        validation = run_epoch(model, validation_loader, device, args.seed + 300000)
        scheduler.step()
        row = {"epoch": epoch,
               **{f"train_{key}": value for key, value in training.items()},
               **{f"val_{key}": value for key, value in validation.items()},
               "seconds": time.perf_counter() - started, "learning_rate": learning_rate}
        rows.append(row)
        with (output / "epoch_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

        checkpoint = {"epoch": epoch, "model_config": model.model_config,
                      "model_state_dict": model.state_dict(), "holdout_indices": holdout,
                      "config": vars(args), "val_loss": validation["loss"]}
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, output / "best.pt")
        torch.save(checkpoint, output / "last.pt")
        print(f"Epoch {epoch:3d}: train {training['loss']:.3f}, holdout {validation['loss']:.3f}, "
              f"lr {learning_rate:.2e}, {row['seconds']:.1f}s", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            examples = output / f"epoch_{epoch:04d}"
            examples.mkdir(exist_ok=True)
            save_examples(model, validation_data, examples, device, args.seed)
            quality = paired_reconstruction_metrics(model, validation_data, device, args.batch_size)
            quality_file = output / "validation_quality.csv"
            fresh = not quality_file.exists()
            with quality_file.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "mse", "ssim"])
                if fresh:
                    writer.writeheader()
                writer.writerow({"epoch": epoch, **quality})
            torch.save({**checkpoint, "validation_quality": quality}, examples / "model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--observation-std", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--holdout", type=int, default=624)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_every, args.cpu_threads,
           args.latent_dim, args.base_channels) < 1 or args.observation_std <= 0:
        parser.error("epochs, batch size, save interval, model sizes and observation std must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
