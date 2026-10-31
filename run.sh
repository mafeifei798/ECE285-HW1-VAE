#!/usr/bin/env bash
set -euo pipefail

CACHE=${CACHE:-cache}
DEVICE=${DEVICE:-auto}
CHECK_EPOCHS=${CHECK_EPOCHS:-0}

python train.py --cache-dir "$CACHE" --device "$DEVICE" \
  --output-dir output/native300 --base-channels 32 \
  --latent-dim 256 --observation-std .01 \
  --epochs 300 --batch-size 64 --holdout 624 \
  --learning-rate .001 --min-learning-rate .00001 \
  --grad-clip 5 --save-every 25 --seed 2026

python sweep_sampler.py --checkpoint output/native300/last.pt --cache-dir "$CACHE" \
  --device "$DEVICE" --components 1,5 --temperatures .85,1.0 \
  --num-samples 5000 --seed 20261023 --batch-size 16 \
  --output-dir output/sampler_sweep

if [[ "$CHECK_EPOCHS" == "1" ]]; then
  for epoch in 100 150 200 250; do
    printf -v padded '%04d' "$epoch"
    python sweep_sampler.py --checkpoint "output/native300/epoch_${padded}/model.pt" \
      --cache-dir "$CACHE" --device "$DEVICE" --samplers gmm --components 5 \
      --temperatures .85 --num-samples 5000 --seed 20261023 --batch-size 16 \
      --output-dir "output/epoch_sweep/e${padded}"
  done
  for epoch in 250 300; do
    if [[ "$epoch" == "300" ]]; then
      checkpoint=output/native300/last.pt
      mixture=output/sampler_sweep/gmm_k5_t0.85/sampler.npz
    else
      checkpoint=output/native300/epoch_0250/model.pt
      mixture=output/epoch_sweep/e0250/gmm_k5_t0.85/sampler.npz
    fi
    for seed in 20261024 20261025; do
      python evaluate.py --checkpoint "$checkpoint" --cache-dir "$CACHE" \
        --device "$DEVICE" --split holdout --sampler gmm --mixture-file "$mixture" \
        --temperature .85 --num-samples 5000 --seed "$seed" --batch-size 16 \
        --output-dir "output/sampling_seeds/e${epoch}_s${seed}"
    done
  done
fi

python nn_check.py --checkpoint output/native300/last.pt --cache-dir "$CACHE" \
  --samples output/sampler_sweep/gmm_k5_t0.85/generated_pixels.npz \
  --output-dir output/nn_check

python evaluate.py --checkpoint output/native300/last.pt --cache-dir "$CACHE" \
  --device "$DEVICE" --sampler gmm \
  --mixture-file output/sampler_sweep/gmm_k5_t0.85/sampler.npz --temperature .85 \
  --num-samples 5000 --seed 20261023 --reference-baselines --batch-size 16 \
  --output-dir output/evaluation
