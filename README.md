# ECE 285 Homework 1

Residual VAE for the Chest X-Ray Pneumonia dataset. The report uses epoch 300
and a five-component Gaussian mixture sampler with temperature 0.85.
The final [report](ECE285_HW1_Report.pdf) is included in the repository.

## Setup

The experiments used an NVIDIA A30, PyTorch 2.7.1 and CUDA 12.8.

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Put the dataset splits in `data/chest_xray/{train,val,test}`. Images are converted
to grayscale, resized directly to 64 x 64 with bilinear interpolation, and scaled
to [0,1]. Resizing does not preserve aspect ratio. The small resolution keeps
training inexpensive. Training duplicates are removed before making a fixed
624-image holdout; 4,566 images remain for optimization. Caches are saved in `cache/`.

## Saved model

Extract the separate weights archive into this directory, then run:

```bash
python evaluate.py --checkpoint checkpoints/selected.pt --cache-dir cache \
  --device cuda --sampler gmm --mixture-file checkpoints/sampler.npz \
  --temperature .85 --num-samples 5000 --seed 20261023 \
  --reference-baselines --output-dir output/evaluation
```

This saves sample grids, reconstructions, pixel arrays and `metrics.json`.
Test results: pooled FID **45.8**, probability FID **0.0210**, IS **1.96**.
FID uses 624 real and generated images; IS uses all 5,000 generated images.
Both FIDs share one ImageNet Inception-v3: pooled FID uses 2,048 average-pool
features, and probability FID uses 1,000 class probabilities. IS uses Ignite.
Pixels are truncated to uint8, converted to RGB and bilinearly resized to 299 x 299,
following the [assigned evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

## Training

```bash
DEVICE=cuda bash run.sh
```

The model has latent dimension 256 and base width 32. Training uses Gaussian
observation std 0.01, a standard-normal prior, Adam with batch size 64, 300 epochs,
cosine learning-rate decay from 1e-3 to 1e-5, gradient clipping at 5 and seed 2026.
The GMM is fitted afterward to posterior means of the training subset.
`run.sh` trains, compares samplers on the holdout, checks nearest training images,
and evaluates the final configuration on test. Set `CHECK_EPOCHS=1` to include
the report's checkpoint comparisons and extra sampling seeds.

The nearest-training check can also be run separately:

```bash
python nn_check.py --samples output/evaluation/evaluation_samples.npz \
  --checkpoint checkpoints/selected.pt --cache-dir cache --output-dir output/nn_check
```

## Files

`model.py` defines the VAE and negative ELBO. `train.py` handles training;
`data.py` loads images; `evaluate.py` computes scores; `sweep_sampler.py` compares
samplers on the holdout. `latent_density.py` fits and samples the GMM, and
`validation_metrics.py` computes reconstruction MSE/SSIM.

`results/` contains the saved metrics, loss curves, checkpoint comparisons and
nearest-training check. Report Figure 1 is `results/training_curves.png`;
Figure 2 is `results/report_samples.png`; Figure 3 is
`results/report_reconstructions.png`. The complete 64-image random grid is
`results/generated_samples.png`.

Table 1 comes from `results/metrics.json`, and Table 2 from `results/epoch_checks.csv`.
The table's train/holdout MSE uses stochastic posterior samples from the logged
Gaussian reconstruction loss. `validation_quality.csv` instead uses posterior means.
The three-seed results in `sampling_seeds.csv` give 38.1 ± 1.1 at epoch 250 and
36.1 ± 1.4 at epoch 300 (mean and sample SD).

Figure 2 uses generated indices 61, 107, 25, 157, 55, 72, 97 and 131;
Figure 3 uses test indices 0, 20, 180 and 550. These are visually selected examples.
Metrics use the full evaluation batches. Data and weights are excluded from this repository.
