#!/usr/bin/env python3
"""
Run Residual RL training or evaluation.

Usage:
    # Train residual RL on top of existing ACT policy
    python run_residual_rl.py train --base_policy models/act_policy/checkpoint_epoch_20.pt --episodes 1000
    
    # Evaluate trained residual policy  
    python run_residual_rl.py eval --base_policy models/act_policy/checkpoint_epoch_20.pt --residual_policy outputs/residual_rl/best_model.pt
    
    # Interactive demo with residual policy
    python run_residual_rl.py demo --base_policy models/act_policy/checkpoint_epoch_20.pt --residual_policy outputs/residual_rl/best_model.pt
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.residual_rl import ResidualACTAgent, train_residual_rl, evaluate_agent, collect_episode
from src.so_arm_env import SoArm100Env


def train_command(args):
    """Train residual RL."""
    print("Starting Residual RL Training")
    print("=" * 50)
    print(f"Base policy: {args.base_policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Create environment
    env = SoArm100Env(render_mode=None)  # Headless for training
    
    try:
        # Train
        agent = train_residual_rl(
            base_policy_path=args.base_policy,
            env=env,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            max_episode_steps=args.max_steps,
            device=args.device,
            lr=args.lr,
            batch_size=args.batch_size,
            alpha=args.bc_weight,
            max_action=args.max_residual,
        )
        
        print("\n✓ Training completed successfully!")
        print(f"✓ Models saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise
    finally:
        env.close()


def eval_command(args):
    """Evaluate residual RL policy."""
    print("Evaluating Residual RL Policy")
    print("=" * 50)
    
    if not args.residual_policy:
        print("Error: --residual_policy required for evaluation")
        return
    
    # Create environment
    env = SoArm100Env(render_mode=None)
    
    try:
        # Load agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = ResidualACTAgent(
            base_policy_path=args.base_policy,
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
        )
        
        # Load residual policy weights
        print(f"Loading residual policy: {args.residual_policy}")
        checkpoint = torch.load(args.residual_policy, map_location=agent.device)
        agent.rl_agent.actor.load_state_dict(checkpoint['residual_policy_state_dict'])
        
        # Evaluate
        print(f"Evaluating for {args.eval_episodes} episodes...")
        avg_reward, success_rate = evaluate_agent(env, agent, args.eval_episodes)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        # Compare with base policy only
        print(f"\nComparing with base policy only...")
        
        # Create base-only agent (zero residuals)
        class BaseOnlyAgent:
            def __init__(self, residual_agent):
                self.residual_agent = residual_agent
            
            def select_action(self, state, exploration=False):
                return self.residual_agent.get_base_action(state)  # Only base action
            
            def reset_action_buffer(self):
                self.residual_agent.reset_action_buffer()
        
        base_agent = BaseOnlyAgent(agent)
        base_reward, base_success = evaluate_agent(env, base_agent, args.eval_episodes)
        
        print(f"   Base Policy Only:")
        print(f"     Average Reward: {base_reward:.3f}")
        print(f"     Success Rate: {base_success:.1%}")
        
        print(f"\n🚀 Residual RL Improvement:")
        print(f"   Reward: {avg_reward - base_reward:+.3f} ({((avg_reward / base_reward - 1) * 100):+.1f}%)")
        print(f"   Success: {success_rate - base_success:+.3f} ({((success_rate - base_success) * 100):+.1f} pts)")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        raise
    finally:
        env.close()


def demo_command(args):
    """Interactive demo with residual RL policy."""
    print("Residual RL Interactive Demo")
    print("=" * 50)
    
    if not args.residual_policy:
        print("Running base policy only (no residual)")
        use_residual = False
    else:
        use_residual = True
    
    # Create environment with rendering
    env = SoArm100Env(render_mode="human")
    
    try:
        # Load agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = ResidualACTAgent(
            base_policy_path=args.base_policy,
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
        )
        
        if use_residual:
            # Load residual policy weights
            print(f"Loading residual policy: {args.residual_policy}")
            checkpoint = torch.load(args.residual_policy, map_location=agent.device)
            agent.rl_agent.actor.load_state_dict(checkpoint['residual_policy_state_dict'])
        
        print("\n🎮 Controls:")
        print("   R - Reset episode")
        print("   Z - Randomize block position")
        print("   Space - Start/stop autonomous control")
        print("   Q - Quit")
        
        running = False
        episode = 0
        
        while True:
            if not running:
                print(f"\n🔄 Episode {episode + 1}")
                print("Press Space to start autonomous control...")
                time.sleep(0.1)
                continue
            
            # Run episode
            print(f"🚀 Running episode {episode + 1}...")
            
            transitions, reward, success = collect_episode(
                env, agent, max_steps=args.max_steps, exploration=False
            )
            
            episode += 1
            
            print(f"✓ Episode completed:")
            print(f"   Reward: {reward:.3f}")
            print(f"   Steps: {len(transitions)}")
            print(f"   Success: {success}")
            
            if use_residual:
                # Also show what base policy would have done
                print("   (Running base policy only for comparison...)")
                
                class BaseOnlyAgent:
                    def __init__(self, residual_agent):
                        self.residual_agent = residual_agent
                    
                    def select_action(self, state, exploration=False):
                        return self.residual_agent.get_base_action(state)
                    
                    def reset_action_buffer(self):
                        self.residual_agent.reset_action_buffer()
                
                base_agent = BaseOnlyAgent(agent)
                _, base_reward, base_success = collect_episode(
                    env, base_agent, max_steps=args.max_steps, exploration=False
                )
                
                print(f"   Base only: reward={base_reward:.3f}, success={base_success}")
                print(f"   Improvement: {reward - base_reward:+.3f}")
            
            running = False  # Stop after each episode
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        raise
    finally:
        env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Residual RL for ACT policies")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train residual RL')
    train_parser.add_argument('--base_policy', type=str, required=True,
                             help='Path to base ACT policy')
    train_parser.add_argument('--output_dir', type=str, default='./outputs/residual_rl',
                             help='Output directory')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes')
    train_parser.add_argument('--max_steps', type=int, default=500,
                             help='Max steps per episode')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device (cpu/cuda/auto)')
    train_parser.add_argument('--lr', type=float, default=3e-4,
                             help='Learning rate')
    train_parser.add_argument('--batch_size', type=int, default=256,
                             help='Batch size')
    train_parser.add_argument('--bc_weight', type=float, default=2.5,
                             help='Behavior cloning regularization weight')
    train_parser.add_argument('--max_residual', type=float, default=0.1,
                             help='Maximum residual action magnitude')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate residual RL policy')
    eval_parser.add_argument('--base_policy', type=str, required=True,
                            help='Path to base ACT policy')
    eval_parser.add_argument('--residual_policy', type=str, required=True,
                            help='Path to trained residual policy')
    eval_parser.add_argument('--eval_episodes', type=int, default=20,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--device', type=str, default='auto',
                            help='Device (cpu/cuda/auto)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--base_policy', type=str, required=True,
                            help='Path to base ACT policy')
    demo_parser.add_argument('--residual_policy', type=str, default=None,
                            help='Path to residual policy (optional)')
    demo_parser.add_argument('--max_steps', type=int, default=500,
                            help='Max steps per episode')
    demo_parser.add_argument('--device', type=str, default='auto',
                            help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()