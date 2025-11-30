# SO-ARM100 ACT Training Project
A complete robotics project for training Action Chunking Transformer (ACT) policies on the SO-ARM100 robot using MuJoCo simulation.

## Project Structure

```
├── src/                    # Core Python modules
│   ├── lerobot_dataset.py  # LeRobot v3.0 dataset recorder
│   ├── so_arm_env.py       # MuJoCo environment with dataset recording
│   └── train_act.py        # ACT training implementation
├── scripts/                # Executable scripts
│   ├── calibrate_arm.py    # Arm calibration script
│   ├── record_demos.py     # High-level demo recording
│   ├── run_act_policy.py   # Run trained ACT policy
│   └── simple_control.py   # Manual control interface
├── configs/                # Configuration files
│   └── calibration.json    # Arm calibration data
├── data/                   # Recorded datasets
│   └── recordings/         # LeRobot v3.0 format datasets
├── models/                 # Trained models and outputs
│   └── act_test/           # ACT model checkpoints
├── env/                    # Environment files
│   ├── scene.xml           # Main MuJoCo scene
│   ├── so_arm100.xml       # Robot model
│   └── lerobot-mujoco/     # Additional MuJoCo assets
├── logs/                   # Log files
│   └── MUJOCO_LOG.TXT      # MuJoCo simulation logs
└── assets/                 # Additional assets
```

## Quick Start

### Using the Convenience Script
```bash
# Record teleoperated demonstrations
./run.sh record

# Train ACT policy on recorded dataset
./run.sh train --dataset data/recordings/dataset_20251130_142003 --output models/act_test

# Run trained policy in simulation
./run.sh run --model models/act_test/final_model.pt
```

### Manual Commands
```bash
# Record teleoperated demonstrations
python scripts/simple_control.py

# Or use the high-level recording script
python scripts/record_demos.py

# Train on recorded dataset
python src/train_act.py --dataset data/recordings/dataset_20251130_142003 --output models/act_test

# Run policy in simulation
python scripts/run_act_policy.py --model models/act_test/final_model.pt
```

## Features

- **Standalone Implementation**: No dependency on LeRobot library
- **LeRobot v3.0 Compatible**: Dataset format matches official LeRobot datasets
- **ACT Policy**: Action Chunking Transformer with VAE for action prediction
- **Real-time Control**: Support for Feetech motor serial communication
- **MuJoCo Simulation**: Physics-based robot simulation
- **Dataset Recording**: Record demonstrations in proper format for training

## Dependencies

- PyTorch
- MuJoCo
- NumPy
- Pandas
- PyArrow
- OpenCV
- FFmpeg (for video encoding)

## Hardware Requirements

- SO-ARM100 robot with Feetech motors (optional)
- CUDA-compatible GPU (recommended for training)
- Display for visualization (optional)

## Training Results

Current trained model:
- **Dataset**: 1 episode, 849 frames, 6-DOF robot
- **Model**: ACT with 17.8M parameters, chunk_size=50
- **Training**: 50 epochs, final loss ~0.19
- **Location**: `models/act_test/final_model.pt`