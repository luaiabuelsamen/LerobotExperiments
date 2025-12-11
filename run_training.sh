#!/bin/bash
#  Residual RL training script
# Run with DISPLAY=:1 for rendering

cd "$(dirname "$0")"

DISPLAY=:1 python scripts/train_far_residual.py \
    --checkpoint outputs/train/act_dataset_20251203/checkpoints/020000 \
    --dataset recordings/dataset_20251203_000617 \
    --output-dir outputs/train/far_residual_$(date +%Y%m%d_%H%M%S) \
    --max-steps 30000 \
    --max-episode-steps 500 \
    --render \
    --action-scale 0.03 \
    --exploration-noise 0.01 \
    --min-noise 0.005 \
    --batch-size 256 \
    --utd-ratio 2 \
    --n-step 3 \
    --num-q 10 \
    --actor-lr 1e-5 \
    --critic-lr 1e-4 \
    --tau 0.005 \
    --action-l2-reg 0.05 \
    --lr-warmup-steps 2000 \
    --reward-scale 100 \
    --save-freq 10000 \
    "$@"
