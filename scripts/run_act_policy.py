#!/usr/bin/env python3
"""
Run trained ACT policy on SO-ARM100 robot in MuJoCo simulation.

Usage:
    python run_act_policy.py --model outputs/act_test/final_model.pt
"""

import argparse
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer as viewer
from pathlib import Path

# Import ACT model from training script
from src.train_act import ACTPolicy


class PolicyRunner:
    """Run ACT policy on SO-ARM100 robot."""
    
    def __init__(
        self,
        model_path: str,
        scene_path: str = "scene.xml",
        chunk_size: int = 50,
        action_horizon: int = 10,
        device: str = None
    ):
        """
        Initialize policy runner.
        
        Args:
            model_path: Path to trained model checkpoint
            scene_path: Path to MuJoCo scene XML
            chunk_size: Action chunk size (must match training)
            action_horizon: How many actions to execute before re-querying
            device: Device to run on (cuda/cpu)
        """
        self.chunk_size = chunk_size
        self.action_horizon = action_horizon
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Get joint indices
        self.joint_names = [
            "Rotation", "Pitch", "Elbow", 
            "Wrist_Pitch", "Wrist_Roll", "Jaw"
        ]
        self.joint_indices = []
        for name in self.joint_names:
            idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx >= 0:
                self.joint_indices.append(idx)
        
        print(f"Found {len(self.joint_indices)} joints")
        
        # Action buffer for chunked execution
        self.action_buffer = None
        self.buffer_idx = 0
        
    def _load_model(self, model_path: str) -> ACTPolicy:
        """Load trained ACT model."""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config
        if "config" in checkpoint:
            config = checkpoint["config"]
            state_dim = config.get("state_dim", 6)
            action_dim = config.get("action_dim", 6)
            chunk_size = config.get("chunk_size", self.chunk_size)
            hidden_dim = config.get("hidden_dim", 512)
            
            # Update chunk size if different
            if chunk_size != self.chunk_size:
                print(f"Note: Model was trained with chunk_size={chunk_size}, using that")
                self.chunk_size = chunk_size
        else:
            # Default config
            state_dim = 6
            action_dim = 6
            hidden_dim = 512
        
        # Create model
        model = ACTPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=self.chunk_size,
            hidden_dim=hidden_dim
        )
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded: state_dim={state_dim}, action_dim={action_dim}, chunk_size={self.chunk_size}")
        return model
    
    def get_state(self) -> np.ndarray:
        """Get current robot state (joint positions)."""
        state = np.zeros(len(self.joint_indices), dtype=np.float32)
        for i, joint_idx in enumerate(self.joint_indices):
            qpos_idx = self.mj_model.jnt_qposadr[joint_idx]
            state[i] = self.mj_data.qpos[qpos_idx]
        return state
    
    def set_action(self, action: np.ndarray):
        """Set robot action (joint positions)."""
        # Flatten action if needed
        action = np.asarray(action).flatten()
        
        for i, joint_idx in enumerate(self.joint_indices):
            if i >= len(action):
                break
            # Clamp to joint limits
            jnt_range = self.mj_model.jnt_range[joint_idx]
            action_val = float(np.clip(float(action[i]), jnt_range[0], jnt_range[1]))
            self.mj_data.ctrl[i] = action_val
    
    def predict_action(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next action using ACT policy.
        
        Uses action chunking: predicts chunk_size actions but only
        executes action_horizon of them before re-querying.
        """
        # Check if we need to re-query the model
        if self.action_buffer is None or self.buffer_idx >= self.action_horizon:
            # Query model for new action chunk
            state_tensor = torch.from_numpy(state).float().to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dim
            
            with torch.no_grad():
                # ACT returns (actions, info_dict) where actions is [batch, chunk_size, action_dim]
                output = self.model(state_tensor)
                if isinstance(output, tuple):
                    action_chunk = output[0]  # Get just the actions tensor
                else:
                    action_chunk = output
            
            # Store action buffer: [chunk_size, action_dim]
            self.action_buffer = action_chunk[0].cpu().numpy()
            self.buffer_idx = 0
        
        # Get next action from buffer
        action = self.action_buffer[self.buffer_idx]
        self.buffer_idx += 1
        
        return action
    
    def reset(self, initial_position=None):
        """Reset the environment and action buffer."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        
        # Set initial joint positions if provided
        if initial_position is not None:
            for i, joint_idx in enumerate(self.joint_indices):
                if i < len(initial_position):
                    qpos_idx = self.mj_model.jnt_qposadr[joint_idx]
                    self.mj_data.qpos[qpos_idx] = initial_position[i]
            # Forward simulation to update the state
            mujoco.mj_forward(self.mj_model, self.mj_data)
        
        self.action_buffer = None
        self.buffer_idx = 0
    
    def run_episode(self, max_steps: int = 1000, render: bool = True, initial_position=None):
        """
        Run one episode with the policy.
        
        Args:
            max_steps: Maximum number of steps
            render: Whether to render with viewer
            initial_position: Initial joint positions to start from
        """
        # Default initial position for ACT policy
        if initial_position is None:
            initial_position = [0.00397189, -1.6223897, 1.6252737, 1.521133, -1.5169878, -0.01448443]
        
        self.reset(initial_position)
        
        if render:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as v:
                print("Running policy... Press Ctrl+C to stop")
                print("Controls: Space=pause, R=reset, Q=quit")
                
                paused = False
                step = 0
                
                while v.is_running() and step < max_steps:
                    if not paused:
                        # Get current state
                        state = self.get_state()
                        
                        # Predict action
                        action = self.predict_action(state)
                        
                        # Execute action
                        self.set_action(action)
                        
                        # Step simulation
                        mujoco.mj_step(self.mj_model, self.mj_data)
                        step += 1
                        
                        if step % 100 == 0:
                            print(f"Step {step}/{max_steps}")
                    
                    # Update viewer
                    v.sync()
                    time.sleep(1/60)  # ~60fps rendering
                    
                print(f"Episode finished after {step} steps")
        else:
            # Run without rendering
            for step in range(max_steps):
                state = self.get_state()
                action = self.predict_action(state)
                self.set_action(action)
                mujoco.mj_step(self.mj_model, self.mj_data)
                
                if step % 100 == 0:
                    print(f"Step {step}/{max_steps}")
    
    def run_continuous(self):
        """Run policy continuously with interactive viewer."""
        # Default initial position for ACT policy
        initial_position = [0.00397189, -1.6223897, 1.6252737, 1.521133, -1.5169878, -0.01448443]
        self.reset(initial_position)
        
        def controller(model, data):
            """Controller callback for viewer."""
            state = self.get_state()
            action = self.predict_action(state)
            self.set_action(action)
        
        print("Running policy with interactive viewer...")
        print("Press 'R' to reset, 'Q' to quit")
        
        # Launch viewer with controller
        viewer.launch(self.mj_model, self.mj_data)


def main():
    parser = argparse.ArgumentParser(description="Run trained ACT policy")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--scene", "-s",
        type=str,
        default="env/scene.xml",
        help="Path to MuJoCo scene XML"
    )
    parser.add_argument(
        "--chunk_size", "-c",
        type=int,
        default=50,
        help="Action chunk size (should match training)"
    )
    parser.add_argument(
        "--action_horizon", "-a",
        type=int,
        default=10,
        help="Number of actions to execute before re-querying"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously with interactive viewer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = PolicyRunner(
        model_path=args.model,
        scene_path=args.scene,
        chunk_size=args.chunk_size,
        action_horizon=args.action_horizon,
        device=args.device
    )
    
    if args.continuous:
        runner.run_continuous()
    else:
        runner.run_episode(max_steps=args.steps)


if __name__ == "__main__":
    main()
