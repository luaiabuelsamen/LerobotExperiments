# Residual RL for Robot Manipulation

![act robot demo fully autonomous](./act020000_viewer.gif)

Fine-tune a pre-trained ACT (Action Chunking Transformer) policy using residual reinforcement learning.

## Overview

This project combines imitation learning with reinforcement learning:
1. **Base Policy**: Pre-trained ACT policy from demonstrations
2. **Residual Policy**: Small RL-learned corrections on top of base actions

The residual policy learns tiny adjustments (±3% of action range) that improve task performance while preserving the stable behavior of the base policy.

Based on the approach from [Fine-Tuning Offline Policies with Optimistic Action Selection](https://arxiv.org/abs/2211.12167).

## Training

```bash
# Train residual RL policy
./run_training.sh
```

Key hyperparameters:
- `--action-scale 0.03`: Max residual magnitude (3% of action range)
- `--exploration-noise 0.01`: Low noise to preserve base policy behavior
- `--action-l2-reg 0.05`: Penalizes large residuals

## Evaluation

```bash
python scripts/evaluate_far_residual.py \
    --residual-checkpoint outputs/train/far_residual_v4_no_demos/final.pt \
    --action-scale 0.03
```

## Results

| Policy | Reward | Success Rate |
|--------|--------|--------------|
| Base ACT | 1462 ± 284 | 100% |
| Base + Residual | 4815 ± 532 | 100% |

The residual policy improves reward by **+229%** while maintaining 100% success.

## Project Structure

```
scripts/
├── train_far_residual.py      # Residual RL training
├── evaluate_far_residual.py   # Evaluation
└── run_act_policy.py          # Run base policy only

src/
├── networks/                  # RL networks (actor, critic, replay buffer)
├── residual_env_wrapper.py    # Wraps env with base policy
├── normalization_utils.py     # Action/state normalization
└── so_arm_env.py              # MuJoCo environment
```