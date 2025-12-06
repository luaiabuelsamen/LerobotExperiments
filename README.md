# SO-ARM100 ACT Training & Residual RL Project

A complete robotics project for training Action Chunking Transformer (ACT) policies and improving them with Residual Reinforcement Learning on the SO-ARM100 robot.

## Overview

This project implements the **"BC → RL"** paradigm for robotic manipulation:

1. **Imitation Learning (IL)**: Train ACT policies from human demonstrations
2. **Residual RL**: Improve IL policies with reinforcement learning  
3. **Evaluation**: Compare and analyze performance improvements

## Project Structure

```
├── src/                    # Core Python modules  
│   ├── lerobot_dataset.py  # LeRobot v3.0 dataset recorder
│   ├── so_arm_env.py       # MuJoCo environment with rewards
│   ├── train_act.py        # ACT training implementation  
│   └── residual_rl.py      # Residual RL implementation
├── scripts/                # Executable scripts
│   ├── calibrate_arm.py    # Arm calibration  
│   ├── simple_control.py   # Manual control & demo recording
│   ├── run_act_policy.py   # Run trained ACT policy
│   ├── evaluate_policy.py  # Evaluate policy performance
│   └── run_residual_rl.py  # Train/evaluate residual RL
├── models/                 # Trained models
│   └── act_policy/         # ACT model checkpoints
├── outputs/                # RL training outputs
│   └── residual_rl/        # Residual RL checkpoints & logs
├── docs/                   # Documentation
│   └── RESIDUAL_RL.md      # Detailed RL guide
└── env/                    # MuJoCo environment files
```

## Quick Start Guide

### 🎯 Phase 1: Imitation Learning (IL)

Record demonstrations and train base ACT policy:

```bash
# 1. Record teleoperated demonstrations  
./run.sh control

# 2. Train ACT policy from demos
./run.sh train --dataset recordings/dataset_20251130_151734 --output models/act_policy

# 3. Test trained policy
./run.sh run --model models/act_policy/checkpoint_epoch_20.pt
```

### 🚀 Phase 2: Reinforcement Learning (RL)

Improve ACT policy with residual RL:

```bash
# 1. Evaluate base ACT policy performance
./run.sh eval --model models/act_policy/checkpoint_epoch_20.pt

# 2. Train residual RL on top of ACT
./run.sh rl-train --base_policy models/act_policy/checkpoint_epoch_20.pt --episodes 1000

# 3. Compare base vs residual policies  
./run.sh rl-eval --base_policy models/act_policy/checkpoint_epoch_20.pt \
                 --residual_policy outputs/residual_rl/best_model.pt

# 4. Interactive demo
./run.sh rl-demo --base_policy models/act_policy/checkpoint_epoch_20.pt \
                 --residual_policy outputs/residual_rl/best_model.pt
```

## Key Features

### 🤖 Imitation Learning
- **ACT (Action Chunking Transformer)**: State-of-the-art IL for robotics
- **LeRobot Compatible**: Standard dataset format for reproducibility  
- **MuJoCo Simulation**: Physics-based training environment
- **Real Robot Support**: Optional hardware integration

### 🎯 Reinforcement Learning  
- **Residual RL**: Conservative approach that improves IL policies safely
- **TD3+BC Algorithm**: Off-policy RL with behavior cloning regularization
- **Bounded Residuals**: Small, safe corrections to base policy
- **Environment Rewards**: Pick-and-place task with shaped rewards

### 📊 Evaluation & Analysis
- **Performance Metrics**: Success rate, reward, efficiency analysis
- **Failure Analysis**: Understanding where policies fail
- **Comparative Evaluation**: Before/after RL improvement
- **TensorBoard Logging**: Training progress visualization

## Command Reference

### Imitation Learning
```bash
./run.sh control      # Record demonstrations
./run.sh train        # Train ACT policy  
./run.sh run          # Run ACT policy
./run.sh eval         # Evaluate ACT performance
```

### Reinforcement Learning  
```bash
./run.sh rl-train     # Train residual RL
./run.sh rl-eval      # Evaluate RL vs base
./run.sh rl-demo      # Interactive demo
```

## Expected Results

With residual RL, you can expect:

| Metric | Base ACT | After RL | Improvement |
|--------|----------|----------|-------------|
| Success Rate | 45-70% | 70-90% | **+15-25%** |
| Average Reward | -20 to +20 | +30 to +60 | **+50-80%** |
| Episode Length | 300-500 steps | 200-350 steps | **20-30% faster** |

## Installation

```bash
# Clone repository
git clone <repository-url>
cd RL

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run.sh
```

## Dependencies

### Core Dependencies
- PyTorch ≥ 2.0
- MuJoCo ≥ 3.0  
- Gymnasium
- NumPy, Pandas
- OpenCV, FFmpeg

### RL Dependencies
- TensorBoard
- Stable-Baselines3
- Weights & Biases (optional)

## Hardware Requirements

- **Training**: CUDA-compatible GPU recommended
- **Real Robot**: SO-ARM100 with Feetech motors (optional)
- **Display**: For visualization (optional with headless mode)

## Documentation

- **[Residual RL Guide](docs/RESIDUAL_RL.md)**: Detailed documentation on residual reinforcement learning
- **Algorithm Details**: TD3+BC, residual architectures, training tips
- **Troubleshooting**: Common issues and solutions

## Research Background

This implementation is based on cutting-edge robotics research:

1. **"What Matters in Learning from Offline Human Demonstrations for Robot Manipulation"** - TD3+BC for robotics
2. **"Pre-Training for Robots: Offline RL Enables Learning New Tasks"** - Offline-to-online RL
3. **"Residual Off-Policy RL for Finetuning Behavior Cloning Policies"** - Residual RL approach

## Current Model Status

**Base ACT Policy**:
- Dataset: 1 episode, 849 frames  
- Model: 17.8M parameters, chunk_size=50
- Training: 50 epochs, loss ~0.19
- Location: `models/act_policy/checkpoint_epoch_20.pt`

**Ready for RL Improvement**! 🎯