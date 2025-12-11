#!/usr/bin/env python3
"""Environment wrapper for residual RL training with base policy.

This wrapper enables residual RL by:
1. Running a frozen base policy (ACT) to get base actions
2. Augmenting observations with base actions
3. Combining base + residual actions before environment steps
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

# Ensure LeRobot is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
LERO_SRC = REPO_ROOT / "lerobot" / "src"
if str(LERO_SRC) not in sys.path:
    sys.path.insert(0, str(LERO_SRC))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


class ResidualRLEnvWrapper:
    """
    Wraps an environment with a frozen base policy for residual RL training.
    
    This wrapper:
    - Runs base_policy on observations to get base actions
    - Augments state observations with base actions for residual policy
    - Combines base + residual actions before stepping
    - Handles policy resets automatically
    """
    
    def __init__(
        self,
        env,
        base_policy: ACTPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        action_scaler,
        state_standardizer,
        initial_position=None,
        camera_name: str = "gripper_fpv",
        image_width: int = 480,
        image_height: int = 480,
    ):
        """
        Args:
            env: MuJoCo environment instance
            base_policy: Frozen ACT policy for base actions
            preprocessor: LeRobot preprocessor for observations
            postprocessor: LeRobot postprocessor for actions
            action_scaler: ActionScaler for normalizing actions
            state_standardizer: StateStandardizer for normalizing states
            initial_position: Optional initial joint positions for reset
            camera_name: Camera name for rendering
            image_width: Image width for rendering
            image_height: Image height for rendering
        """
        self.env = env
        self.base_policy = base_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.initial_position = initial_position
        
        # Get device from policy parameters
        self.device = next(base_policy.parameters()).device
        
        # Get action dimension
        self.action_dim = env.action_space.shape[0]
        
        # Store observation and action spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Setup rendering for base policy
        self.mj_model = env.unwrapped.model
        self.mj_data = env.unwrapped.data
        camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera_id = camera_id if camera_id >= 0 else -1
        self.renderer = mujoco.Renderer(self.mj_model, image_height, image_width)
        
        # Track last base action
        self._last_base_naction = None
        
    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset environment and base policy."""
        # Use initial position if provided
        if self.initial_position is not None:
            if 'options' not in kwargs:
                kwargs['options'] = {}
            if 'initial_position' not in kwargs['options']:
                kwargs['options']['initial_position'] = self.initial_position
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        
        # Convert to torch tensor
        obs_torch = self._obs_to_torch(obs)
        
        # Reset base policy
        self.base_policy.reset()
        
        # Get base action from frozen policy using processors (like run.sh)
        with torch.no_grad():
            batch = self.preprocessor(obs_torch)
            base_action = self.base_policy.select_action(batch)
            base_action = self.postprocessor(base_action)
        
        # Scale to [-1, 1]
        base_naction = self.action_scaler.scale(base_action.squeeze(0))
        
        # Augment observation with base action
        augmented_obs = self._augment_obs(obs_torch, base_naction)
        
        # Store for next step
        self._last_base_naction = base_naction
        
        return augmented_obs, info
    
    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict]:
        """
        Step environment with residual action.
        
        Args:
            residual_naction: Residual action from residual policy (normalized, in [-action_scale, action_scale])
            
        Returns:
            augmented_obs: Observations with base actions
            reward: Reward
            terminated: Episode terminated flag
            truncated: Episode truncated flag
            info: Info dict
        """
        # Combine base and residual actions (both in normalized [-1, 1] space)
        # residual_naction is scaled by actor.action_scale (e.g., 0.2), so in range [-0.2, 0.2]
        # base_naction is in full [-1, 1] range
        
        # Ensure residual_naction is on same device as base_naction
        residual_naction = residual_naction.to(self._last_base_naction.device)
        
        combined_naction = self._last_base_naction + residual_naction
        
        # Unscale to original action space for environment
        env_action = self.action_scaler.unscale(combined_naction)
        
        # Convert to numpy for environment
        env_action_np = env_action.cpu().numpy()
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(env_action_np)
        
        # Convert observation to torch
        obs_torch = self._obs_to_torch(obs)
        
        # Get next base action using processors
        with torch.no_grad():
            batch = self.preprocessor(obs_torch)
            base_action = self.base_policy.select_action(batch)
            base_action = self.postprocessor(base_action)
        
        base_naction = self.action_scaler.scale(base_action.squeeze(0))
        
        # Augment observation
        augmented_obs = self._augment_obs(obs_torch, base_naction)
        
        # Store for next step
        self._last_base_naction = base_naction
        
        # Store combined action in info for logging
        info["combined_action"] = combined_naction.cpu()
        info["base_action"] = base_action.cpu()
        info["residual_action"] = residual_naction.cpu()
        
        return augmented_obs, reward, terminated, truncated, info
        
    def _obs_to_torch(self, obs: dict[str, np.ndarray] | np.ndarray) -> dict[str, torch.Tensor]:
        """Convert numpy observations to torch tensors for ACT policy.
        
        Renders image from MuJoCo and creates proper observation dict.
        """
        # Handle numpy array (Box observation space)
        if isinstance(obs, np.ndarray):
            # Extract first 6 elements (joint positions)
            if obs.shape[-1] > 6:
                obs = obs[..., :6]
            
            state = obs.copy()
        else:
            state = obs.get("observation.state", obs)
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if state.shape[-1] > 6:
                state = state[..., :6]
        
        # Render image from MuJoCo (like run.sh does)
        self.renderer.update_scene(self.mj_data, camera=self.camera_id)
        frame = self.renderer.render().copy().astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        image = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        
        # Create observation dict
        obs_torch = {
            "observation.state": torch.from_numpy(state).float(),
            "observation.images.front": image,
        }
        
        return obs_torch
    
    def _augment_obs(
        self, obs: dict[str, torch.Tensor], base_naction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Augment observation with base action for residual policy."""
        augmented_obs = obs.copy()
        
        # Add base action as separate key
        augmented_obs["observation.base_action"] = base_naction
        
        # Standardize state if present
        if "observation.state" in augmented_obs:
            augmented_obs["observation.state"] = self.state_standardizer.standardize(
                augmented_obs["observation.state"]
            )
        
        return augmented_obs
    
    def render(self):
        """Pass through to environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        return self.env.close()
    
    def __getattr__(self, name: str):
        """Delegate unknown attributes to environment."""
        return getattr(self.env, name)
