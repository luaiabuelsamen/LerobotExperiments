#!/usr/bin/env python3
"""
Residual RL for ACT Policy

Implements residual reinforcement learning on top of a trained ACT (IL) policy.
The approach follows the "ResFiT" style: π_total(s) = π_BC(s) + π_residual(s)

This allows fine-tuning with RL while keeping the benefits of the IL policy.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

from train_act import ACTPolicy
from so_arm_env import SoArm100Env


class ResidualPolicy(nn.Module):
    """
    Small MLP that outputs action residuals to add to the base ACT policy.
    
    Design principles:
    - Small network to ensure stability
    - Bounded outputs to prevent large deviations from BC policy
    - Operates on single timesteps (not chunks) for real-time control
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_residual: float = 0.1,  # Clamp residual magnitude
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.max_residual = max_residual
        
        layers = []
        prev_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer - tanh to bound outputs
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Output bounded residual actions."""
        residual = self.network(state)
        return residual * self.max_residual


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done),
        )
    
    def __len__(self):
        return len(self.buffer)


class TD3BC:
    """
    TD3+BC (Twin Delayed Deep Deterministic Policy Gradient + Behavior Cloning)
    
    A conservative off-policy RL algorithm that's good for fine-tuning BC policies.
    Adds behavior cloning loss to standard TD3 to stay close to demonstration data.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,  # BC regularization weight
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Networks
        self.actor = ResidualPolicy(state_dim, action_dim).to(self.device)
        self.actor_target = ResidualPolicy(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin critics
        self.critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic1_target = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2_target = self._build_critic(state_dim, action_dim).to(self.device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Hyperparameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        
        self.total_it = 0
    
    def _build_critic(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build a critic network Q(s, a)."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        residual = self.actor(state).cpu().data.numpy().flatten()
        return residual
    
    def train(
        self,
        replay_buffer: ReplayBuffer,
        bc_buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Train the agent for one step."""
        self.total_it += 1
        
        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).unsqueeze(1)
        next_state = next_state.to(self.device)
        done = done.to(self.device).unsqueeze(1)
        
        # Critic loss
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Twin Q targets
            target_q1 = self.critic1_target(torch.cat([next_state, next_action], 1))
            target_q2 = self.critic2_target(torch.cat([next_state, next_action], 1))
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1 - done.float()) * target_q
        
        # Current Q estimates
        current_q1 = self.critic1(torch.cat([state, action], 1))
        current_q2 = self.critic2(torch.cat([state, action], 1))
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        bc_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_actions = self.actor(state)
            actor_loss = -self.critic1(torch.cat([state, actor_actions], 1)).mean()
            
            # BC loss (regularization)
            if bc_buffer is not None and len(bc_buffer) > 0:
                bc_state, bc_action, _, _, _ = bc_buffer.sample(min(batch_size, len(bc_buffer)))
                bc_state = bc_state.to(self.device)
                bc_action = bc_action.to(self.device)
                
                predicted_residual = self.actor(bc_state)
                # For BC loss, we want residual to be close to zero (no change from base policy)
                bc_loss = F.mse_loss(predicted_residual, torch.zeros_like(predicted_residual))
                
                total_actor_loss = actor_loss + self.alpha * bc_loss
            else:
                total_actor_loss = actor_loss
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._update_targets()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
        }
    
    def _update_targets(self):
        """Soft update of target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ResidualACTAgent:
    """
    Agent that combines ACT base policy with residual RL.
    
    π_total(s) = π_BC(s) + π_residual(s)
    """
    
    def __init__(
        self,
        base_policy_path: str,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 50,
        action_horizon: int = 10,
        device: str = "auto",
        **td3bc_kwargs,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.chunk_size = chunk_size
        self.action_horizon = action_horizon
        
        # Load base ACT policy (frozen)
        self.base_policy = self._load_base_policy(base_policy_path, state_dim, action_dim)
        self.base_policy.eval()
        
        # Initialize residual RL agent
        self.rl_agent = TD3BC(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **td3bc_kwargs,
        )
        
        # Action chunking state
        self.action_buffer = None
        self.buffer_idx = 0
        
    def _load_base_policy(self, model_path: str, state_dim: int, action_dim: int) -> ACTPolicy:
        """Load the base ACT policy."""
        print(f"Loading base ACT policy from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            chunk_size = config.get("chunk_size", self.chunk_size)
            hidden_dim = config.get("hidden_dim", 512)
        else:
            chunk_size = self.chunk_size
            hidden_dim = 512
        
        # Create base policy
        base_policy = ACTPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
        )
        
        # Load weights
        if "model_state_dict" in checkpoint:
            base_policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            base_policy.load_state_dict(checkpoint)
        
        base_policy = base_policy.to(self.device)
        
        print(f"Base policy loaded: chunk_size={chunk_size}")
        return base_policy
    
    def get_base_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from base ACT policy using chunking."""
        # Check if we need to re-query the base policy
        if self.action_buffer is None or self.buffer_idx >= self.action_horizon:
            # Query base policy for new action chunk
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_chunk, _ = self.base_policy(state_tensor)
                self.action_buffer = action_chunk.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
                self.buffer_idx = 0
        
        # Get current action from buffer
        action = self.action_buffer[self.buffer_idx].copy()
        self.buffer_idx += 1
        
        return action
    
    def select_action(self, state: np.ndarray, exploration: bool = False) -> np.ndarray:
        """
        Select action using base policy + residual.
        
        Args:
            state: Current robot state
            exploration: Whether to add exploration noise
            
        Returns:
            Combined action: base_action + residual_action
        """
        # Get base action (with chunking)
        base_action = self.get_base_action(state)
        
        # Get residual from RL policy
        residual = self.rl_agent.select_action(state)
        
        # Add exploration noise if requested
        if exploration:
            noise = np.random.normal(0, 0.01, size=residual.shape)
            residual = residual + noise
        
        # Combine actions
        combined_action = base_action + residual
        
        # Ensure action is in valid range (will be clipped by environment)
        return combined_action
    
    def reset_action_buffer(self):
        """Reset action chunking buffer (call at episode start)."""
        self.action_buffer = None
        self.buffer_idx = 0


def collect_episode(
    env: SoArm100Env,
    agent: ResidualACTAgent,
    max_steps: int = 500,
    exploration: bool = True,
) -> Tuple[List[Tuple], float, bool]:
    """
    Collect one episode of experience.
    
    Returns:
        transitions: List of (s, a, r, s', done) tuples
        total_reward: Episode return
        success: Whether episode was successful
    """
    transitions = []
    
    state, _ = env.reset()
    agent.reset_action_buffer()
    
    total_reward = 0.0
    success = False
    
    for step in range(max_steps):
        action = agent.select_action(state, exploration=exploration)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        transitions.append((state.copy(), action.copy(), reward, next_state.copy(), done))
        
        total_reward += reward
        state = next_state
        
        if info.get('success', False) or info.get('block_in_box', False):
            success = True
        
        if done:
            break
    
    return transitions, total_reward, success


def train_residual_rl(
    base_policy_path: str,
    env: SoArm100Env,
    output_dir: str = "./outputs/residual_rl",
    num_episodes: int = 1000,
    max_episode_steps: int = 500,
    batch_size: int = 256,
    replay_buffer_size: int = 100000,
    learning_starts: int = 1000,
    train_freq: int = 1,
    save_freq: int = 100,
    eval_freq: int = 50,
    eval_episodes: int = 10,
    device: str = "auto",
    **kwargs,
):
    """
    Train residual RL on top of ACT base policy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: state_dim={state_dim}, action_dim={action_dim}")
    
    # Initialize agent
    agent = ResidualACTAgent(
        base_policy_path=base_policy_path,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **kwargs,
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    # Training loop
    episode = 0
    total_steps = 0
    best_eval_reward = float('-inf')
    
    print("Starting residual RL training...")
    print(f"Base policy: {base_policy_path}")
    print(f"Output directory: {output_dir}")
    
    while episode < num_episodes:
        # Collect episode
        transitions, episode_reward, success = collect_episode(
            env, agent, max_episode_steps, exploration=True
        )
        
        # Add to replay buffer
        for transition in transitions:
            replay_buffer.push(*transition)
        
        total_steps += len(transitions)
        episode += 1
        
        # Log episode stats
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/episode_length", len(transitions), episode)
        writer.add_scalar("train/success", float(success), episode)
        writer.add_scalar("train/total_steps", total_steps, episode)
        
        print(f"Episode {episode}: reward={episode_reward:.3f}, steps={len(transitions)}, success={success}")
        
        # Training
        if total_steps >= learning_starts and episode % train_freq == 0:
            # Train for multiple steps per episode
            train_steps = max(1, len(transitions) // 4)
            
            for _ in range(train_steps):
                if len(replay_buffer) < batch_size:
                    break
                    
                losses = agent.rl_agent.train(replay_buffer, batch_size=batch_size)
                
                # Log training losses
                for key, value in losses.items():
                    writer.add_scalar(f"losses/{key}", value, total_steps)
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_reward, eval_success_rate = evaluate_agent(env, agent, eval_episodes)
            
            writer.add_scalar("eval/episode_reward", eval_reward, episode)
            writer.add_scalar("eval/success_rate", eval_success_rate, episode)
            
            print(f"Eval Episode {episode}: reward={eval_reward:.3f}, success_rate={eval_success_rate:.3f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_checkpoint(agent, output_dir / "best_model.pt", episode, eval_reward)
        
        # Save checkpoint
        if episode % save_freq == 0:
            save_checkpoint(agent, output_dir / f"checkpoint_episode_{episode}.pt", episode, episode_reward)
    
    # Final save
    save_checkpoint(agent, output_dir / "final_model.pt", episode, episode_reward)
    writer.close()
    
    print(f"Training completed! Best eval reward: {best_eval_reward:.3f}")
    return agent


def evaluate_agent(env: SoArm100Env, agent: ResidualACTAgent, num_episodes: int = 10) -> Tuple[float, float]:
    """Evaluate the agent performance."""
    total_reward = 0.0
    success_count = 0
    
    for _ in range(num_episodes):
        _, episode_reward, success = collect_episode(env, agent, exploration=False)
        total_reward += episode_reward
        success_count += int(success)
    
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    
    return avg_reward, success_rate


def save_checkpoint(agent: ResidualACTAgent, path: Path, episode: int, reward: float):
    """Save model checkpoint."""
    torch.save({
        'episode': episode,
        'reward': reward,
        'residual_policy_state_dict': agent.rl_agent.actor.state_dict(),
        'critic1_state_dict': agent.rl_agent.critic1.state_dict(),
        'critic2_state_dict': agent.rl_agent.critic2.state_dict(),
        'actor_optimizer': agent.rl_agent.actor_optimizer.state_dict(),
        'critic1_optimizer': agent.rl_agent.critic1_optimizer.state_dict(),
        'critic2_optimizer': agent.rl_agent.critic2_optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Residual RL on ACT policy")
    parser.add_argument("--base_policy", type=str, required=True,
                       help="Path to trained ACT model")
    parser.add_argument("--output_dir", type=str, default="./outputs/residual_rl",
                       help="Output directory")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Max steps per episode")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cpu/cuda/auto)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    
    args = parser.parse_args()
    
    # Create environment
    env = SoArm100Env(render_mode=None)  # Headless for faster training
    
    # Train agent
    agent = train_residual_rl(
        base_policy_path=args.base_policy,
        env=env,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        max_episode_steps=args.max_steps,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()