"""MSE and SSIM for posterior-mean reconstructions."""

import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def paired_reconstruction_metrics(model, dataset, device, batch_size):
    model.eval()
    errors, similarities = [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        mu, _ = model.encode(batch.to(device))
        reconstructed = model.pixel_mean(model.decode(mu)).cpu()
        errors.extend((batch - reconstructed).square().mean((1, 2, 3)).tolist())
        similarities.extend(structural_similarity(
            x[0].numpy(), y[0].numpy(), data_range=1.0, gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False) for x, y in zip(batch, reconstructed))
    return {"mse": float(np.mean(errors)), "ssim": float(np.mean(similarities))}
