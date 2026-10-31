"""Compare generated-to-training distances with holdout-to-training distances."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from data import ChestXrayDataset


@torch.inference_mode()
def nearest_mse(queries, references, batch_size=128):
    reference = torch.from_numpy(references.reshape(len(references), -1).astype(np.float32)) / 255
    norms = reference.square().mean(1)
    distances, indices = [], []
    dimensions = reference.shape[1]
    for offset in range(0, len(queries), batch_size):
        pixels = queries[offset:offset + batch_size]
        query = torch.from_numpy(pixels.reshape(len(pixels), -1).astype(np.float32)) / 255
        squared = query.square().mean(1)[:, None] + norms[None, :]
        squared -= 2 * (query @ reference.T) / dimensions
        best = squared.argmin(1).numpy()
        # Recompute selected distances directly in float64 to avoid cancellation.
        difference = (pixels.astype(np.float64) - references[best].astype(np.float64)) / 255
        distances.extend(np.square(difference).reshape(len(pixels), -1).mean(1))
        indices.extend(best)
    return np.asarray(distances), np.asarray(indices)


def grayscale(pixels):
    if pixels.ndim == 4 and pixels.shape[1] == 1:
        pixels = pixels[:, 0]
    if pixels.ndim != 3 or pixels.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale images with shape N,H,W or N,1,H,W")
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--train-cache", type=Path)
    source.add_argument("--cache-dir", default="cache")
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = checkpoint["model_config"]["image_size"]
    if args.train_cache:
        images = np.load(args.train_cache, allow_pickle=False)
    else:
        images = ChestXrayDataset(args.data_root, "train", image_size, args.cache_dir).images
    images = grayscale(np.asarray(images))
    holdout = np.asarray(checkpoint.get("holdout_indices") or [], dtype=int)
    if not len(holdout):
        raise ValueError("A checkpoint with a training holdout is needed for the reference distances")
    seen = np.flatnonzero(~np.isin(np.arange(len(images)), holdout))
    with np.load(args.samples, allow_pickle=False) as saved:
        generated = grayscale(saved["generated"])
    if generated.shape[1:] != images.shape[1:]:
        raise ValueError("Generated images and training cache have different resolutions")
    reference = images[seen]
    generated_distance, nearest = nearest_mse(generated, reference, args.batch_size)
    holdout_distance, _ = nearest_mse(images[holdout], reference, args.batch_size)
    threshold = float(np.quantile(holdout_distance, .05))
    result = {
        "metric": "nearest-training pixel MSE on grayscale [0,1] images",
        "checkpoint_epoch": checkpoint["epoch"],
        "training_images": len(seen), "generated_images": len(generated),
        "holdout_images": len(holdout),
        "generated_exact_pixel_matches": int((generated_distance == 0).sum()),
        "generated_min_mse": float(generated_distance.min()),
        "generated_median_mse": float(np.median(generated_distance)),
        "holdout_min_mse": float(holdout_distance.min()),
        "holdout_median_mse": float(np.median(holdout_distance)),
        "holdout_5th_percentile": threshold,
        "generated_fraction_below_holdout_5th_percentile": float((generated_distance < threshold).mean()),
        "interpretation_limit": "Pixel proximity is a screening diagnostic, not a proof of memorization or its absence.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nn_check.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "nearest_distances.npz",
                        generated_mse=generated_distance, holdout_mse=holdout_distance,
                        nearest_training_indices=seen[nearest])
    count = min(8, len(generated))
    pairs = np.concatenate((generated[:count], reference[nearest[:count]]))
    save_image(torch.from_numpy(pairs.copy()).unsqueeze(1).float().div(255),
               args.output_dir / "nearest_pairs.png", nrow=count, pad_value=1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
