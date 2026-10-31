# ECE 285 Homework 1

A residual VAE trained on both classes of the Chest X-Ray Pneumonia dataset.
The report uses the epoch-300 checkpoint, with a five-component Gaussian mixture
sampler at temperature 0.85. The [report](ECE285_HW1_Report.pdf) is included.

## Setup

For the CUDA Linux environment used in the experiments:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Training used an NVIDIA A30 with PyTorch 2.7.1 and CUDA 12.8.
Extract the dataset so `data/chest_xray/` contains `train/`, `val/` and `test/`.
Images are EXIF-corrected, converted to grayscale, bilinearly resized to 64 x 64,
and divided by 255. Direct resizing does not preserve the original aspect ratio.
This resolution keeps training and evaluation inexpensive. Caches are created automatically.

## Evaluate the saved model

Extract the separate weights archive into this directory, then run:

```bash
python evaluate.py --checkpoint checkpoints/selected.pt --cache-dir cache \
  --device cuda --sampler gmm --mixture-file checkpoints/sampler.npz \
  --temperature .85 --num-samples 5000 --seed 20261023 \
  --reference-baselines --output-dir output/evaluation
```

This writes generated images, reconstructions, pixel arrays and `metrics.json`.
The checkpoint contains model weights and settings; optimizer states are omitted.

Run the nearest-training-image check on all 5,000 generated images with:

```bash
python nn_check.py --samples output/evaluation/evaluation_samples.npz \
  --checkpoint checkpoints/selected.pt --cache-dir cache \
  --output-dir output/nn_check
```

The top row of `nearest_pairs.png` is generated images; the bottom row is their
nearest training images by pixel MSE. Holdout images are excluded from the training
reference. The reported fifth-percentile fraction is a proximity diagnostic, not
a proof of memorization or its absence.

## Reproduce training

```bash
DEVICE=cuda bash run.sh
```

The residual VAE has 256 latent dimensions, base width 32 and about 10.2 million parameters.
Training uses Gaussian observation standard deviation 0.01, KL coefficient one,
a standard-normal training prior, Adam, batch 64, 300 epochs, cosine learning-rate
decay from 1e-3 to 1e-5, norm-5 clipping, and no augmentation.
Removing 26 exact training duplicates leaves 5,190 images. Seed 2026 holds out 624,
leaving 4,566 for optimization. Generation uses a full-covariance Gaussian mixture
fitted only to the training posterior means.
The mixture approximates the distribution of training latent means. It is fitted
after VAE training; the VAE training prior remains standard normal.

`run.sh` trains the model, compares samplers on the holdout, checks nearest training
images, and evaluates the fixed report configuration on test. To also reproduce
the 100/150/200/250/300-epoch checks and extra sampling seeds, use:

```bash
DEVICE=cuda CHECK_EPOCHS=1 bash run.sh
```

The epoch checks fix K=5, temperature 0.85 and seed 20261023, refitting the mixture
for each checkpoint. Extra draws reuse each checkpoint's fitted mixture.
`sweep_sampler.py` saves all 5,000 generated pixels; candidate FID and IS use the
624-image reference count. Final IS in `evaluate.py` uses all 5,000 images.

## Results

Test pooled FID: **45.8**; Ignite probability FID: **0.0210**; IS: **1.96**.
The JSON and CSV files retain the full precision.
FID uses 624 test images and the first 624 of 5,000 generated images; IS uses all
5,000. Decoder means are floor-quantized to uint8, repeated to RGB, PIL-bilinearly
resized to 299 x 299, and divided by 255. ImageNet1K-v1 Inception-v3 uses
`transform_input=True`. Probability FID uses 1,000 Softmax features; pooled FID uses
2,048 average-pool features. The metric APIs and PIL resizing follow the
[assigned Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

`results/` includes per-epoch losses, negative-ELBO/KL plots,
generated examples and a fixed 64-image random grid. Reconstruction MSE
in `reconstruction_mse.csv` is derived from the logged Gaussian reconstruction
term and uses the same stochastic posterior samples as training/holdout losses.
Figure 2 shows eight visually selected images from the fixed-seed generated batch.
Figure 3 shows four selected test reconstructions. Metrics cover
the complete evaluation batches. Nearest-neighbor enlargement is for display.

`epoch_checks.csv` keeps stochastic train/holdout MSE separate from posterior-mean
holdout MSE. Report Table 2 uses the stochastic train/holdout pair from the logged
Gaussian reconstruction terms.
`sampler_sweep.csv` contains the epoch-300 sampler comparison;
`sampling_seeds.csv` contains three fixed draws for epochs 250 and 300.
Their pooled FID means and sample standard deviations are 38.1 ± 1.1 and
36.1 ± 1.4. With the previously chosen 1.5-FID tolerance for preferring an
earlier checkpoint, epoch 300 remains selected among the checked epochs.

The nearest-training check on all 5,000 final samples finds no exact pixel matches.
77/5,000 (1.54%) have nearest-training MSE below the holdout fifth percentile;
see `nn_check.json`. This check does not rule out other forms of memorization.

The repository excludes datasets, cached Inception features and model weights.
