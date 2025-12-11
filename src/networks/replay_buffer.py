#!/usr/bin/env python3
"""
N-step replay buffer for  residual RL.

Based on the FAR paper implementation with:
- N-step returns for lower variance TD targets
- Support for symmetric offline/online sampling
- Image storage as uint8 for memory efficiency
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class NStepReplayBuffer:
    """
    Replay buffer with N-step returns support.
    
    Features:
    - N-step return computation
    - Symmetric sampling from offline demos + online data
    - Memory-efficient uint8 image storage
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 200000,
        device: str = "cuda",
        n_step: int = 3,
        gamma: float = 0.99,
        image_shape: tuple[int, int, int] | None = None,
    ):
        """
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            max_size: Maximum buffer size
            device: Device for sampling
            n_step: N-step return horizon
            gamma: Discount factor
            image_shape: Optional (C, H, W) for image storage
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.image_shape = image_shape
        
        # Main storage (CPU for memory efficiency)
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.base_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.residual_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.next_base_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32)
        
        # Image storage (uint8 for memory efficiency)
        if image_shape is not None:
            self.images = torch.zeros((max_size,) + image_shape, dtype=torch.uint8)
            self.next_images = torch.zeros((max_size,) + image_shape, dtype=torch.uint8)
        else:
            self.images = None
            self.next_images = None
        
        # N-step buffer for accumulating transitions
        self.n_step_buffer: deque[dict[str, Any]] = deque(maxlen=n_step)
        
        # Offline demo storage (separate)
        self.offline_size = 0
        self._offline_data: dict[str, torch.Tensor] = {}
    
    def _compute_n_step_return(self) -> float:
        """Compute n-step discounted return from buffer."""
        reward = 0.0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition["reward"]
        return reward
    
    def add(
        self,
        state: np.ndarray,
        base_action: np.ndarray,
        residual_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_base_action: np.ndarray,
        done: bool,
        image: np.ndarray | None = None,
        next_image: np.ndarray | None = None,
    ):
        """Add transition with n-step handling."""
        transition = {
            "state": state,
            "base_action": base_action,
            "residual_action": residual_action,
            "reward": reward,
            "next_state": next_state,
            "next_base_action": next_base_action,
            "done": done,
            "image": image,
            "next_image": next_image,
        }
        
        self.n_step_buffer.append(transition)
        
        # Store when we have n transitions or episode ends
        if len(self.n_step_buffer) == self.n_step or done:
            first = self.n_step_buffer[0]
            last = self.n_step_buffer[-1]
            
            # N-step return
            n_step_reward = self._compute_n_step_return()
            
            # Store
            self.states[self.ptr] = torch.from_numpy(first["state"]).float()
            self.base_actions[self.ptr] = torch.from_numpy(first["base_action"]).float()
            self.residual_actions[self.ptr] = torch.from_numpy(first["residual_action"]).float()
            self.rewards[self.ptr] = n_step_reward
            self.next_states[self.ptr] = torch.from_numpy(last["next_state"]).float()
            self.next_base_actions[self.ptr] = torch.from_numpy(last["next_base_action"]).float()
            
            # Done if any transition in n-step was terminal
            any_done = any(t["done"] for t in self.n_step_buffer)
            self.dones[self.ptr] = float(any_done)
            
            # Images
            if self.images is not None and first["image"] is not None:
                img = first["image"]
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                self.images[self.ptr] = torch.from_numpy(img)
                
                next_img = last["next_image"]
                if next_img.dtype == np.float32:
                    next_img = (next_img * 255).astype(np.uint8)
                self.next_images[self.ptr] = torch.from_numpy(next_img)
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
            if done:
                self.n_step_buffer.clear()
    
    def add_offline_demos(
        self,
        dataset,
        action_scaler,
        state_standardizer,
        image_key: str | None = None,
    ):
        """
        Load offline demonstrations into separate buffer.
        
        In demos, expert action is the base action, residual = 0.
        This teaches the critic that base + zero residual = good.
        
        Args:
            dataset: LeRobotDataset
            action_scaler: ActionScaler for normalizing actions
            state_standardizer: StateStandardizer for normalizing states
            image_key: Optional key for images in dataset
        """
        print("Loading offline demonstrations...")
        
        # Access hf_dataset directly to avoid video decoding
        hf_data = dataset.hf_dataset
        from_indices = dataset.meta.episodes["dataset_from_index"]
        to_indices = dataset.meta.episodes["dataset_to_index"]
        num_episodes = len(from_indices)
        
        transitions: list[dict] = []
        
        for ep_idx in range(num_episodes):
            start_idx = from_indices[ep_idx]
            end_idx = to_indices[ep_idx]
            
            for i in range(start_idx, end_idx - 1):
                state = np.array(hf_data[i]["observation.state"])
                action = np.array(hf_data[i]["action"])
                next_state = np.array(hf_data[i + 1]["observation.state"])
                next_action = np.array(hf_data[i + 1]["action"])
                
                # Standardize state
                state_t = state_standardizer.standardize(
                    torch.from_numpy(state).float()
                ).numpy()
                next_state_t = state_standardizer.standardize(
                    torch.from_numpy(next_state).float()
                ).numpy()
                
                # Scale action to [-1, 1]
                action_t = action_scaler.scale(
                    torch.from_numpy(action).float()
                ).numpy()
                next_action_t = action_scaler.scale(
                    torch.from_numpy(next_action).float()
                ).numpy()
                
                # In demos: base = expert, residual = 0
                transitions.append({
                    "state": state_t,
                    "base_action": action_t,
                    "residual_action": np.zeros_like(action_t),
                    "reward": 10.0,  # High reward for demos
                    "next_state": next_state_t,
                    "next_base_action": next_action_t,
                    "done": float(i == end_idx - 2),
                })
        
        n = len(transitions)
        if n == 0:
            print("  Warning: No transitions loaded from dataset")
            return
        
        state_dim = transitions[0]["state"].shape[-1]
        action_dim = transitions[0]["base_action"].shape[-1]
        
        self._offline_data = {
            "states": torch.zeros((n, state_dim), dtype=torch.float32),
            "base_actions": torch.zeros((n, action_dim), dtype=torch.float32),
            "residual_actions": torch.zeros((n, action_dim), dtype=torch.float32),
            "rewards": torch.zeros((n, 1), dtype=torch.float32),
            "next_states": torch.zeros((n, state_dim), dtype=torch.float32),
            "next_base_actions": torch.zeros((n, action_dim), dtype=torch.float32),
            "dones": torch.zeros((n, 1), dtype=torch.float32),
        }
        
        for i, t in enumerate(transitions):
            self._offline_data["states"][i] = torch.from_numpy(t["state"]).float()
            self._offline_data["base_actions"][i] = torch.from_numpy(t["base_action"]).float()
            self._offline_data["residual_actions"][i] = torch.from_numpy(t["residual_action"]).float()
            self._offline_data["rewards"][i] = t["reward"]
            self._offline_data["next_states"][i] = torch.from_numpy(t["next_state"]).float()
            self._offline_data["next_base_actions"][i] = torch.from_numpy(t["next_base_action"]).float()
            self._offline_data["dones"][i] = t["done"]
        
        self.offline_size = n
        print(f"  Loaded {n} offline demo transitions from {num_episodes} episodes")
    
    def sample(
        self,
        batch_size: int,
        offline_fraction: float = 0.5,
    ) -> tuple[torch.Tensor, ...]:
        """
        Sample batch with symmetric offline/online sampling.
        
        Args:
            batch_size: Total batch size
            offline_fraction: Fraction of batch from offline demos
            
        Returns:
            Tuple of (states, base_actions, residual_actions, rewards,
                     next_states, next_base_actions, dones)
        """
        device = self.device
        
        if self.offline_size > 0 and self.size > 0:
            # Symmetric sampling
            offline_batch = int(batch_size * offline_fraction)
            online_batch = batch_size - offline_batch
            
            offline_idx = torch.randint(0, self.offline_size, (offline_batch,))
            online_idx = torch.randint(0, self.size, (online_batch,))
            
            return (
                torch.cat([
                    self._offline_data["states"][offline_idx],
                    self.states[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["base_actions"][offline_idx],
                    self.base_actions[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["residual_actions"][offline_idx],
                    self.residual_actions[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["rewards"][offline_idx],
                    self.rewards[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["next_states"][offline_idx],
                    self.next_states[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["next_base_actions"][offline_idx],
                    self.next_base_actions[online_idx]
                ]).to(device),
                torch.cat([
                    self._offline_data["dones"][offline_idx],
                    self.dones[online_idx]
                ]).to(device),
            )
        
        elif self.offline_size > 0:
            # Only offline data
            idx = torch.randint(0, self.offline_size, (batch_size,))
            return (
                self._offline_data["states"][idx].to(device),
                self._offline_data["base_actions"][idx].to(device),
                self._offline_data["residual_actions"][idx].to(device),
                self._offline_data["rewards"][idx].to(device),
                self._offline_data["next_states"][idx].to(device),
                self._offline_data["next_base_actions"][idx].to(device),
                self._offline_data["dones"][idx].to(device),
            )
        
        else:
            # Only online data
            idx = torch.randint(0, self.size, (batch_size,))
            return (
                self.states[idx].to(device),
                self.base_actions[idx].to(device),
                self.residual_actions[idx].to(device),
                self.rewards[idx].to(device),
                self.next_states[idx].to(device),
                self.next_base_actions[idx].to(device),
                self.dones[idx].to(device),
            )
    
    def sample_with_images(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        """Sample with images (online data only for now)."""
        if self.images is None:
            raise ValueError("Buffer was not initialized with image storage")
        
        idx = torch.randint(0, self.size, (batch_size,))
        device = self.device
        
        return (
            self.states[idx].to(device),
            self.base_actions[idx].to(device),
            self.residual_actions[idx].to(device),
            self.rewards[idx].to(device),
            self.next_states[idx].to(device),
            self.next_base_actions[idx].to(device),
            self.dones[idx].to(device),
            self.images[idx].to(device).float() / 255.0,
            self.next_images[idx].to(device).float() / 255.0,
        )


if __name__ == "__main__":
    # Test buffer
    buffer = NStepReplayBuffer(
        state_dim=6,
        action_dim=6,
        max_size=1000,
        n_step=3,
        gamma=0.99,
    )
    
    # Add some transitions
    for ep in range(5):
        for step in range(100):
            buffer.add(
                state=np.random.randn(6).astype(np.float32),
                base_action=np.random.randn(6).astype(np.float32),
                residual_action=np.random.randn(6).astype(np.float32) * 0.1,
                reward=np.random.rand(),
                next_state=np.random.randn(6).astype(np.float32),
                next_base_action=np.random.randn(6).astype(np.float32),
                done=(step == 99),
            )
    
    print(f"Buffer size: {buffer.size}")
    
    # Sample
    batch = buffer.sample(32)
    print(f"Sampled batch shapes: {[b.shape for b in batch]}")
