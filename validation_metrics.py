"""Native and common-resolution metrics for paired held-out reconstructions."""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def paired_reconstruction_metrics(model, dataset, reference64, device, batch_size):
    if len(dataset) != len(reference64):
        raise ValueError("Native and 64-pixel references must contain the same images")
    model.eval()
    values = {key: [] for key in ["mse", "ssim", "mse64", "ssim64"]}
    loaders = [DataLoader(dataset, batch_size=batch_size),
               DataLoader(reference64, batch_size=batch_size)]
    for batch, batch64 in zip(*loaders):
        mu, _ = model.encode(batch.to(device))
        reconstructed = model.pixel_mean(model.decode(mu)).cpu()
        values["mse"].extend((batch - reconstructed).square().mean((1, 2, 3)).tolist())
        for original, output, original64 in zip(batch, reconstructed, batch64):
            a, b = original[0].numpy(), output[0].numpy()
            values["ssim"].append(structural_similarity(
                a, b, data_range=1., gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
            small = (b if b.shape == (64, 64) else np.asarray(
                Image.fromarray(b).resize((64, 64), Image.Resampling.BILINEAR)))
            target64 = original64[0].numpy()
            values["mse64"].append(float(np.mean((target64 - small) ** 2)))
            values["ssim64"].append(structural_similarity(
                target64, small, data_range=1., gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False))
    return {key: float(np.mean(numbers)) for key, numbers in values.items()}
