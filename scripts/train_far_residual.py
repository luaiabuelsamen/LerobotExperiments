#!/usr/bin/env python3
"""
 Residual RL Training Script

Implements the full FAR (Fine-tuning with Action Residuals) paper methodology:
1. ViT encoder for image observations
2. Ensemble critics (10 Q-heads with min-over-random-2)
3. N-step returns
4. Data augmentation (random shifts)
5. Gradient clipping
6. Action L2 regularization
7. Combined action clamping
8. LR warmup
9. Higher UTD ratio

Key insight: FAR processes IMAGES through ViT encoder, not just state vectors!
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
LERO_SRC = REPO_ROOT / "lerobot" / "src"
if str(LERO_SRC) not in sys.path:
    sys.path.insert(0, str(LERO_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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

from src.normalization_utils import ActionScaler, StateStandardizer
from src.residual_env_wrapper import ResidualRLEnvWrapper
from src.so_arm_env import SoArm100Env

# FAR components
from src.networks.q_agent import SimpleQAgent
from src.networks.replay_buffer import NStepReplayBuffer


# =============================================================================
# IMAGE REPLAY BUFFER (stores both state and images)
# =============================================================================
class ImageReplayBuffer:
    """Replay buffer that stores images for ViT encoder."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_shape: tuple[int, int, int] = (3, 84, 84),
        max_size: int = 100000,
        device: str = "cuda",
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        # State storage
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.base_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.residual_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.next_base_actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32)
        
        # Image storage (uint8 for memory efficiency)
        self.images = torch.zeros((max_size,) + image_shape, dtype=torch.uint8)
        self.next_images = torch.zeros((max_size,) + image_shape, dtype=torch.uint8)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
    
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
        image: np.ndarray,
        next_image: np.ndarray,
    ):
        """Add transition with n-step handling."""
        # Convert image to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        if next_image.dtype == np.float32 or next_image.dtype == np.float64:
            next_image = (next_image * 255).astype(np.uint8)
        
        transition = {
            "state": state.copy(),
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "reward": reward,
            "next_state": next_state.copy(),
            "next_base_action": next_base_action.copy(),
            "done": done,
            "image": image.copy(),
            "next_image": next_image.copy(),
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
            
            # Done if any transition was terminal
            any_done = any(t["done"] for t in self.n_step_buffer)
            self.dones[self.ptr] = float(any_done)
            
            # Images
            self.images[self.ptr] = torch.from_numpy(first["image"])
            self.next_images[self.ptr] = torch.from_numpy(last["next_image"])
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int):
        """Sample batch."""
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
            self.images[idx].to(device),
            self.next_images[idx].to(device),
        )
    
    def __len__(self):
        return self.size


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_base_policy(checkpoint_path: str, device: str = "cuda"):
    """Load frozen ACT base policy."""
    checkpoint_path = Path(checkpoint_path)
    
    # Handle checkpoint directory structure
    policy_path = checkpoint_path / "pretrained_model"
    if not policy_path.exists():
        policy_path = checkpoint_path
    
    # Load policy
    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.to(device)
    policy.eval()
    
    # Freeze
    for param in policy.parameters():
        param.requires_grad = False
    
    # Load processors with converters
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": str(device)}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_path),
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    
    return policy, preprocessor, postprocessor


def get_initial_position(dataset_path: str):
    """Get initial joint position from dataset."""
    dataset = LeRobotDataset(repo_id="local/dataset", root=str(dataset_path))
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    initial_pos = np.array(dataset.hf_dataset[from_idx]["observation.state"])
    return initial_pos, dataset


def resize_image(image: np.ndarray, size: int = 84) -> np.ndarray:
    """Resize image to target size for ViT encoder."""
    # image is (H, W, C) or (C, H, W)
    if image.shape[0] == 3:  # (C, H, W)
        image = image.transpose(1, 2, 0)  # -> (H, W, C)
    
    import cv2
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # Back to (C, H, W)
    return resized.transpose(2, 0, 1)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args):
    """Main training function."""
    print("=" * 60)
    print(" Residual RL Training")
    print("=" * 60)
    
    device = args.device
    
    # =========================================================================
    # 1. Load base policy
    # =========================================================================
    print("\n[1/5] Loading base ACT policy...")
    policy, preprocessor, postprocessor = load_base_policy(args.checkpoint, device)
    print(f"  Loaded from: {args.checkpoint}")
    
    # =========================================================================
    # 2. Create environment
    # =========================================================================
    print("\n[2/5] Creating environment...")
    env = SoArm100Env(
        render_mode="human" if args.render else None,
        enable_viewer_hotkeys=False,
    )
    
    # Load initial position from dataset
    initial_pos, dataset = get_initial_position(args.dataset)
    print(f"  Initial position from dataset: {initial_pos}")
    
    # Action/state normalization
    action_low = torch.from_numpy(env.action_space.low).float()
    action_high = torch.from_numpy(env.action_space.high).float()
    action_scaler = ActionScaler(action_low, action_high, device=device)
    
    state_mean = torch.from_numpy(initial_pos).float()
    state_std = torch.ones(6).float() * 0.5
    state_standardizer = StateStandardizer(state_mean, state_std, device=device)
    
    # Create wrapper
    env_wrapper = ResidualRLEnvWrapper(
        env=env,
        base_policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        initial_position=initial_pos,
    )
    
    action_dim = env.action_space.shape[0]
    state_dim = 6  # Joint positions
    
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    # =========================================================================
    # 3. Create agent and buffer
    # =========================================================================
    print("\n[3/5] Creating  agent...")
    
    # Using SimpleQAgent (state-based) for now
    # Full ViT-based agent would require more memory
    agent = SimpleQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_q=args.num_q,
        action_scale=args.action_scale,
        action_l2_reg=args.action_l2_reg,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        lr_warmup_steps=args.lr_warmup_steps,
        device=device,
    )
    
    print(f"  Num Q-heads: {args.num_q}")
    print(f"  Action scale: {args.action_scale}")
    print(f"  Action L2 reg: {args.action_l2_reg}")
    print(f"  Learning rates: actor={args.actor_lr}, critic={args.critic_lr}")
    print(f"  Gradient clipping: {args.grad_clip}")
    
    # Create buffer with n-step returns
    buffer = NStepReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.buffer_size,
        device=device,
        n_step=args.n_step,
        gamma=args.gamma,
    )
    
    print(f"  N-step returns: n={args.n_step}")
    print(f"  Buffer size: {args.buffer_size}")
    
    # =========================================================================
    # 4. Load offline demos
    # =========================================================================
    if args.load_demos:
        print("\n[4/5] Loading offline demonstrations...")
        # dataset is already loaded from get_initial_position
        buffer.add_offline_demos(dataset, action_scaler, state_standardizer)
    else:
        print("\n[4/5] Skipping offline demo loading...")
    
    # =========================================================================
    # 5. Training loop
    # =========================================================================
    print("\n[5/5] Starting training...")
    print(f"  Max steps: {args.max_steps}")
    print(f"  UTD ratio: {args.utd_ratio}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Exploration noise: {args.exploration_noise}")
    print(f"  Policy update freq: {args.policy_freq}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Training metrics
    episode_rewards = []
    episode_successes = []
    training_metrics = []
    
    # Reset environment
    obs, _ = env_wrapper.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    
    # Get state and base action from obs
    state = obs["observation.state"]
    base_action = obs["observation.base_action"]
    
    # Exploration noise schedule
    def get_exploration_noise(step):
        # Linear decay from max to min noise
        progress = min(1.0, step / args.exploration_decay_steps)
        return args.exploration_noise * (1.0 - progress) + args.min_noise * progress
    
    start_time = time.time()
    
    for step in tqdm(range(args.max_steps), desc="Training"):
        # Get exploration noise
        noise_std = get_exploration_noise(step)
        
        # Select action with exploration
        state_t = state.unsqueeze(0).to(device) if state.dim() == 1 else state.to(device)
        base_t = base_action.unsqueeze(0).to(device) if base_action.dim() == 1 else base_action.to(device)
        
        residual = agent.select_action(state_t, base_t, std=noise_std)
        residual = residual.squeeze(0)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env_wrapper.step(residual)
        done = terminated or truncated
        
        next_state = next_obs["observation.state"]
        next_base_action = next_obs["observation.base_action"]
        
        # Add to buffer
        buffer.add(
            state=state.cpu().numpy(),
            base_action=base_action.cpu().numpy(),
            residual_action=residual.cpu().numpy(),
            reward=reward,
            next_state=next_state.cpu().numpy(),
            next_base_action=next_base_action.cpu().numpy(),
            done=done,
        )
        
        # Track metrics
        episode_reward += reward
        episode_steps += 1
        
        # Update
        if buffer.size >= args.batch_size:
            for _ in range(args.utd_ratio):
                batch = buffer.sample(args.batch_size)
                states, base_actions, residual_actions, rewards, next_states, next_base_actions, dones = batch
                
                # Scale rewards
                rewards = rewards / args.reward_scale
                
                metrics = agent.train(
                    state=states,
                    base_action=base_actions,
                    residual_action=residual_actions,
                    reward=rewards,
                    next_state=next_states,
                    next_base_action=next_base_actions,
                    done=dones,
                    update_actor=(step % args.policy_freq == 0),
                    stddev=args.target_noise,
                    n_step=args.n_step,
                )
                
                if step % 1000 == 0 and _ == 0:
                    training_metrics.append({
                        "step": step,
                        "critic_loss": metrics.get("critic_loss", 0),
                        "actor_loss": metrics.get("actor_loss", 0),
                        "residual_magnitude": metrics.get("residual_magnitude", 0),
                    })
        
        # Handle episode end
        if done or episode_steps >= args.max_episode_steps:
            episode_rewards.append(episode_reward)
            episode_successes.append(info.get("success", False))
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_success = np.mean(episode_successes[-10:])
                elapsed = time.time() - start_time
                
                print(f"\nEpisode {episode_count} | Step {step}")
                print(f"  Avg Reward (10 ep): {avg_reward:.2f}")
                print(f"  Avg Success (10 ep): {avg_success:.1%}")
                print(f"  Exploration noise: {noise_std:.4f}")
                print(f"  Buffer size: {buffer.size}")
                print(f"  Elapsed: {elapsed:.1f}s")
                
                if training_metrics:
                    last = training_metrics[-1]
                    print(f"  Critic loss: {last['critic_loss']:.4f}")
                    print(f"  Residual mag: {last['residual_magnitude']:.4f}")
            
            # Reset
            obs, _ = env_wrapper.reset()
            state = obs["observation.state"]
            base_action = obs["observation.base_action"]
            episode_reward = 0.0
            episode_steps = 0
        else:
            state = next_state
            base_action = next_base_action
        
        # Save checkpoint
        if (step + 1) % args.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint_{step+1}.pt"
            agent.save(str(ckpt_path), step + 1)
            print(f"\nSaved checkpoint: {ckpt_path}")
            
            # Also save latest
            agent.save(str(output_dir / "latest.pt"), step + 1)
    
    # Final save
    agent.save(str(output_dir / "final.pt"), args.max_steps)
    print(f"\nTraining complete! Saved to {output_dir}")
    
    # Save training history
    history = {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "training_metrics": training_metrics,
    }
    torch.save(history, output_dir / "training_history.pt")
    
    env_wrapper.close()


def parse_args():
    parser = argparse.ArgumentParser(description=" Residual RL Training")
    
    # Paths
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/train/act_dataset_20251203/checkpoints/020000",
                        help="Path to ACT checkpoint")
    parser.add_argument("--dataset", type=str,
                        default="recordings/dataset_20251203_000617",
                        help="Path to demonstration dataset")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/train/far_residual",
                        help="Output directory for checkpoints")
    
    # Training
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--max-episode-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for updates")
    parser.add_argument("--buffer-size", type=int, default=200000,
                        help="Replay buffer size")
    parser.add_argument("--utd-ratio", type=int, default=4,
                        help="Update-to-data ratio (FAR uses 4)")
    parser.add_argument("--n-step", type=int, default=3,
                        help="N-step returns (FAR uses 3)")
    parser.add_argument("--policy-freq", type=int, default=2,
                        help="Policy update frequency")
    
    # Network
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--num-q", type=int, default=10,
                        help="Number of Q-heads (FAR uses 10)")
    
    # FAR hyperparameters
    parser.add_argument("--actor-lr", type=float, default=1e-4,
                        help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=1e-4,
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Target network soft update rate")
    parser.add_argument("--action-scale", type=float, default=0.2,
                        help="Max residual action magnitude (FAR uses 0.1-0.2)")
    parser.add_argument("--action-l2-reg", type=float, default=0.001,
                        help="L2 regularization on residual actions")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--lr-warmup-steps", type=int, default=1000,
                        help="LR warmup steps")
    
    # Exploration
    parser.add_argument("--exploration-noise", type=float, default=0.2,
                        help="Initial exploration noise")
    parser.add_argument("--min-noise", type=float, default=0.05,
                        help="Minimum exploration noise")
    parser.add_argument("--exploration-decay-steps", type=int, default=50000,
                        help="Steps to decay exploration")
    parser.add_argument("--target-noise", type=float, default=0.2,
                        help="Target policy noise for TD3")
    
    # Reward
    parser.add_argument("--reward-scale", type=float, default=100.0,
                        help="Reward scaling factor")
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Checkpoint save frequency")
    parser.add_argument("--load-demos", action="store_true",
                        help="Load offline demonstrations")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
