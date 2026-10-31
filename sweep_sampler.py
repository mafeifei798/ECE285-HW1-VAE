"""Compare latent samplers using held-out training images."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for ops MPS lacks

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.utils import save_image

from evaluate import (SharedInception, configure_runtime, dataset_pixels, decode_images, encode_means,
                      extract_features, fid_from_features, inception_score, resolve_device,
                      sample_latents, training_split)
from latent_density import fit_gaussian_mixture
from model import load_vae_checkpoint


def parse_list(text, kind):
    return [kind(value) for value in text.split(",") if value.strip()]


@torch.inference_mode()
def sweep(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, checkpoint = load_vae_checkpoint(args.checkpoint)
    model.to(device)
    seen, holdout = training_split(args.data_root, args.cache_dir, model.image_size, checkpoint)
    if args.reference == "holdout":
        if holdout is None:
            raise SystemExit("This checkpoint has no held-out images: train with --holdout, "
                             "or pass --reference train (in-sample, favours large mixtures)")
        reference = holdout
    else:
        print("Warning: the reference images were used to fit the model and the mixture", flush=True)
        chosen = np.random.default_rng(args.seed).choice(
            len(seen), min(args.reference_count, len(seen)), replace=False)
        reference = Subset(seen, sorted(chosen.tolist()))
    reference_pixels = dataset_pixels(reference)
    count = len(reference_pixels)
    latents = encode_means(model, seen, args.batch_size, device)
    holdout_latents = (encode_means(model, holdout, args.batch_size, device)
                       if holdout is not None else None)
    extractor = SharedInception().to(device).eval()
    reference_probs, reference_pool = extract_features(
        extractor, reference_pixels, args.batch_size, device, verbose=False)
    print(f"{len(latents)} training latents, {count} reference images; "
          f"real-reference IS {inception_score(reference_probs):.3f}", flush=True)

    configurations = []
    if "prior" in args.samplers:
        configurations += [("prior", 0, t) for t in args.temperatures]
    if "gmm" in args.samplers:
        configurations += [("gmm", k, t) for k in args.components for t in args.temperatures]
    mixtures, rows = {}, []
    for sampler, components, temperature in configurations:
        mixture = None
        if sampler == "gmm":
            if components not in mixtures:
                mixtures[components] = fit_gaussian_mixture(latents, components, args.seed)
            mixture = mixtures[components]
        z = sample_latents(model, max(count, args.num_samples), temperature, args.seed, device, mixture)
        images = decode_images(model, z, args.batch_size, device)
        folder = args.output_dir / f"{sampler}_k{components}_t{temperature:g}"
        folder.mkdir(exist_ok=True)
        save_image(images[:64].float().div(255), folder / "generated_samples.png",
                   nrow=8, pad_value=1)
        np.savez_compressed(folder / "generated_pixels.npz", generated=images.numpy())
        if mixture is not None:
            np.savez_compressed(folder / "sampler.npz", weights=mixture.weights_,
                                means=mixture.means_, covariances=mixture.covariances_)
        # Keep all generated pixels for nearest-neighbor checks; FID uses equal counts.
        probs, pool = extract_features(extractor, images[:count], args.batch_size, device, verbose=False)
        row = {"sampler": f"{model.prior} prior" if sampler == "prior" else "gmm",
               "components": components or "", "temperature": temperature,
               "fid_pool3_2048": fid_from_features(reference_pool, pool),
               "fid_probability_1000": fid_from_features(reference_probs, probs),
               "is": inception_score(probs),
               "is_image_count": count,
               "holdout_latent_loglik": (mixture.score(holdout_latents)
                                         if mixture is not None and holdout_latents is not None else "")}
        rows.append(row)
        (folder / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"{row['sampler']:>24} K={str(row['components']):>3} T={temperature:.2f}: "
              f"FID {row['fid_pool3_2048']:.2f}, FID-1000 {row['fid_probability_1000']:.4f}, "
              f"IS {row['is']:.3f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampler_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["fid_pool3_2048"]))
    best = min(rows, key=lambda row: row["fid_pool3_2048"])
    mixture_file = None
    if best["sampler"] == "gmm":
        mixture = mixtures[best["components"]]
        mixture_file = args.output_dir / "best_mixture.npz"
        np.savez_compressed(mixture_file, weights=mixture.weights_, means=mixture.means_,
                            covariances=mixture.covariances_)
    selection = {**best, "sampler": "gmm" if mixture_file is not None else "prior",
                 "seed": args.seed, "num_samples": max(count, args.num_samples),
                 "checkpoint": str(args.checkpoint),
                 "mixture_file": str(mixture_file) if mixture_file is not None else None}
    (args.output_dir / "lowest_fid_candidate.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Lowest held-out FID: {best['sampler']}, K={best['components']}, "
          f"T={best['temperature']} ({best['fid_pool3_2048']:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of generated images to save; FID uses the reference count")
    parser.add_argument("--samplers", nargs="+", choices=("prior", "gmm"),
                        default=("prior", "gmm"))
    parser.add_argument("--components", default="1,5",
                        type=lambda text: parse_list(text, int))
    parser.add_argument("--temperatures", default="0.85,1.0",
                        type=lambda text: parse_list(text, float))
    parser.add_argument("--reference", default="holdout", choices=("holdout", "train"))
    parser.add_argument("--reference-count", type=int, default=624,
                        help="Reference size for --reference train")
    args = parser.parse_args()
    if min(args.components + [args.batch_size, args.cpu_threads]) < 1 or min(args.temperatures) <= 0:
        parser.error("components, batch-size, cpu-threads and temperatures must be positive")
    configure_runtime(args.cpu_threads)
    sweep(args)


if __name__ == "__main__":
    main()
