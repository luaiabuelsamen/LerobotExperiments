#!/usr/bin/env python3
"""
Evaluate  Residual RL Policy

Compares:
1. Base policy only (zero residual)
2. Base + learned residual
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
from src.networks.q_agent import SimpleQAgent


def load_base_policy(checkpoint_path: str, device: str = "cuda"):
    """Load frozen ACT base policy."""
    checkpoint_path = Path(checkpoint_path)
    
    # Handle checkpoint directory structure
    policy_path = checkpoint_path / "pretrained_model"
    if not policy_path.exists():
        policy_path = checkpoint_path
    
    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.to(device)
    policy.eval()
    
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
    return initial_pos


def evaluate(args):
    """Evaluate residual policy."""
    print("=" * 60)
    print(" Residual RL Evaluation")
    print("=" * 60)
    
    device = args.device
    
    # Load base policy
    print("\n[1/4] Loading base ACT policy...")
    policy, preprocessor, postprocessor = load_base_policy(args.base_checkpoint, device)
    
    # Create environment
    print("\n[2/4] Creating environment...")
    env = SoArm100Env(
        render_mode="human",
        enable_viewer_hotkeys=False,
    )
    
    # Get initial position
    initial_pos = get_initial_position(args.dataset)
    
    # Normalization
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
    
    # Load residual agent if provided
    print("\n[3/4] Loading residual policy...")
    residual_agent = None
    if args.residual_checkpoint:
        state_dim = 6
        action_dim = env.action_space.shape[0]
        
        # Create agent with same config
        residual_agent = SimpleQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_layers=2,
            num_q=10,
            action_scale=args.action_scale,
            action_l2_reg=0.001,
            actor_lr=1e-4,
            critic_lr=1e-4,
            tau=0.01,
            gamma=0.99,
            grad_clip=1.0,
            lr_warmup_steps=0,
            device=device,
        )
        
        step = residual_agent.load(args.residual_checkpoint)
        print(f"  Loaded residual checkpoint from step {step}")
    else:
        print("  No residual checkpoint - evaluating base policy only")
    
    # Evaluation
    print("\n[4/4] Running evaluation...")
    
    results = {
        "base_only": {"rewards": [], "successes": [], "steps": []},
        "residual": {"rewards": [], "successes": [], "steps": []},
    }
    
    for mode in ["base_only", "residual"]:
        if mode == "residual" and residual_agent is None:
            continue
        
        print(f"\n--- Evaluating: {mode} ---")
        
        for ep in range(args.num_episodes):
            obs, _ = env_wrapper.reset()
            episode_reward = 0.0
            episode_steps = 0
            success = False
            
            for step in range(args.max_steps):
                state = obs["observation.state"]
                base_action = obs["observation.base_action"]
                
                if mode == "base_only":
                    # Zero residual
                    residual = torch.zeros(6, device=device)
                else:
                    # Use learned residual
                    state_t = state.unsqueeze(0).to(device)
                    base_t = base_action.unsqueeze(0).to(device)
                    residual = residual_agent.select_action(state_t, base_t, std=0.0)
                    residual = residual.squeeze(0)
                
                obs, reward, terminated, truncated, info = env_wrapper.step(residual)
                episode_reward += reward
                episode_steps += 1
                
                if info.get("success", False):
                    success = True
                
                if terminated or truncated:
                    break
            
            results[mode]["rewards"].append(episode_reward)
            results[mode]["successes"].append(success)
            results[mode]["steps"].append(episode_steps)
            
            print(f"  Episode {ep+1}: reward={episode_reward:.2f}, success={success}, steps={episode_steps}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for mode in ["base_only", "residual"]:
        if len(results[mode]["rewards"]) == 0:
            continue
        
        avg_reward = np.mean(results[mode]["rewards"])
        std_reward = np.std(results[mode]["rewards"])
        success_rate = np.mean(results[mode]["successes"])
        avg_steps = np.mean(results[mode]["steps"])
        
        print(f"\n{mode.upper()}:")
        print(f"  Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Steps: {avg_steps:.1f}")
    
    # Compare
    if len(results["residual"]["rewards"]) > 0:
        base_r = np.mean(results["base_only"]["rewards"])
        res_r = np.mean(results["residual"]["rewards"])
        improvement = res_r - base_r
        
        print(f"\nIMPROVEMENT: {improvement:+.2f} reward ({improvement/abs(base_r)*100:+.1f}%)")
    
    env_wrapper.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FAR Residual Policy")
    
    parser.add_argument("--base-checkpoint", type=str,
                        default="outputs/train/act_dataset_20251203/checkpoints/020000",
                        help="Path to base ACT checkpoint")
    parser.add_argument("--residual-checkpoint", type=str, default=None,
                        help="Path to residual policy checkpoint")
    parser.add_argument("--dataset", type=str,
                        default="recordings/dataset_20251203_000617",
                        help="Path to dataset for initial position")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--action-scale", type=float, default=0.2,
                        help="Residual action scale (must match training)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
