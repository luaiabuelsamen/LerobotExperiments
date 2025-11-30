#!/usr/bin/env python3
"""
Record Demonstrations Script

This script provides a simple interface to record robot demonstrations 
in the LeRobot v3.0 format for training ACT or other imitation learning policies.

Usage:
    python record_demos.py --output ./my_dataset --task "Pick up the block"
    
Controls (in terminal):
    D - Start recording an episode
    F - Finish/save current episode
    R - Reset environment
    Z - Randomize block position
    Q - Quit and finalize dataset

Controls (in viewer window):
    Same as above - keyboard callbacks work in viewer too
"""

import argparse
import os
import sys
import time
import threading
import termios
import tty

# Import the environment and dataset recorder
import src.so_arm_env as so_arm_env
from src.lerobot_dataset import LeRobotDatasetRecorder


def get_key():
    """Get a single keypress from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class DemoRecorder:
    """High-level interface for recording demonstrations."""
    
    def __init__(
        self,
        output_dir: str,
        task: str = "Pick up the block",
        fps: int = 30,
        robot_type: str = "so100_mujoco",
    ):
        self.output_dir = output_dir
        self.task = task
        self.fps = fps
        self.robot_type = robot_type
        self.running = True
        self.env = None
        
    def setup(self):
        """Initialize environment and recorder."""
        # Create environment
        try:
            self.env = so_arm_env.SoArm100Env(
                render_mode="human", 
                camera_mode="first_person"
            )
        except Exception as e:
            if "GLFW" in str(e) or "DISPLAY" in str(e):
                print("No display available, cannot record demonstrations in headless mode")
                return False
            raise
        
        # Initialize dataset recorder
        success = self.env.init_dataset_recorder(
            dataset_path=self.output_dir,
            task=self.task,
            fps=self.fps,
            robot_type=self.robot_type,
        )
        
        if not success:
            print("Failed to initialize dataset recorder")
            return False
        
        return True
    
    def print_status(self):
        """Print current recording status."""
        if self.env is None:
            return
        
        recording = "🔴 RECORDING" if self.env.dataset_recording else "⚪ NOT RECORDING"
        episodes = self.env.dataset_recorder.current_episode_index if self.env.dataset_recorder else 0
        frames = self.env.dataset_recorder.total_frames if self.env.dataset_recorder else 0
        
        print(f"\r{recording} | Episodes: {episodes} | Total Frames: {frames}", end="", flush=True)
    
    def keyboard_handler(self):
        """Handle keyboard input in a separate thread."""
        print("\n" + "=" * 60)
        print("DEMONSTRATION RECORDING")
        print("=" * 60)
        print(f"Output: {self.output_dir}")
        print(f"Task: {self.task}")
        print("=" * 60)
        print("Controls:")
        print("  D - Start recording episode")
        print("  F - Finish/save episode") 
        print("  R - Reset environment")
        print("  Z - Randomize block position")
        print("  Q - Quit and save dataset")
        print("=" * 60 + "\n")
        
        while self.running:
            try:
                key = get_key().lower()
                
                if key == 'd':
                    if not self.env.dataset_recording:
                        print("\n>>> Starting episode recording...")
                        self.env.start_dataset_recording(self.task)
                        
                elif key == 'f':
                    if self.env.dataset_recording:
                        print("\n>>> Saving episode...")
                        self.env.save_dataset_episode()
                        
                elif key == 'r':
                    print("\n>>> Resetting environment...")
                    self.env.reset()
                    
                elif key == 'z':
                    print("\n>>> Randomizing block position...")
                    self.env._randomize_environment()
                    
                elif key == 'q':
                    print("\n>>> Quitting...")
                    self.running = False
                    break
                    
                self.print_status()
                
            except Exception as e:
                print(f"\nKeyboard error: {e}")
                break
    
    def run_with_simulated_leader(self):
        """Run demo recording with simulated leader arm (for testing)."""
        if not self.setup():
            return
        
        # Start keyboard thread
        keyboard_thread = threading.Thread(target=self.keyboard_handler, daemon=True)
        keyboard_thread.start()
        
        # Reset environment
        obs, info = self.env.reset()
        
        try:
            t = 0
            while self.running:
                # Generate smooth oscillating movements (simulated teleoperation)
                action = self._generate_demo_action(t)
                t += 0.05
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Control loop rate
                time.sleep(0.01)
                
                if terminated or truncated:
                    obs, info = self.env.reset()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.env.close()
            print(f"\nDataset saved to: {self.output_dir}")
    
    def run_with_leader_arm(self, leader_port: str = '/dev/ttyACM1'):
        """Run demo recording with real leader arm."""
        # Import the leader arm reader from simple_control
        from simple_control import SO100LeaderReader
        
        if not self.setup():
            return
        
        # Connect to leader arm
        leader_arm = SO100LeaderReader(port=leader_port)
        if not leader_arm.connect():
            print("Failed to connect to leader arm")
            return
        
        # Start keyboard thread
        keyboard_thread = threading.Thread(target=self.keyboard_handler, daemon=True)
        keyboard_thread.start()
        
        # Reset environment
        obs, info = self.env.reset()
        
        try:
            while self.running:
                # Read leader arm positions
                leader_action = leader_arm.get_action()
                action = leader_arm.apply_calibration(leader_action)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Control loop rate
                time.sleep(0.01)
                
                if terminated or truncated:
                    obs, info = self.env.reset()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            leader_arm.disconnect()
            self.env.close()
            print(f"\nDataset saved to: {self.output_dir}")
    
    def _generate_demo_action(self, t: float):
        """Generate smooth demo actions for testing."""
        import numpy as np
        
        # Create smooth oscillating movements
        return np.array([
            0.5 * np.sin(t * 0.5),        # shoulder_pan
            0.3 * np.sin(t * 0.7 + 1),    # shoulder_lift  
            0.4 * np.sin(t * 0.6 + 2),    # elbow_flex
            0.2 * np.sin(t * 0.8 + 3),    # wrist_flex
            0.6 * np.sin(t * 0.4 + 4),    # wrist_roll
            0.1 * np.sin(t * 0.3 + 5) + 0.5  # gripper
        ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Record robot demonstrations in LeRobot format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record with simulated leader (for testing)
    python record_demos.py --output ./my_dataset --simulated
    
    # Record with real leader arm
    python record_demos.py --output ./my_dataset --port /dev/ttyACM1
    
    # Record with custom task description
    python record_demos.py --output ./my_dataset --task "Place block in box"
"""
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/recordings/demo_dataset",
        help="Output directory for the dataset"
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="Pick up the block",
        help="Task description for the demonstrations"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording framerate"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for leader arm"
    )
    
    parser.add_argument(
        "--simulated", "-s",
        action="store_true",
        help="Use simulated leader arm for testing"
    )
    
    args = parser.parse_args()
    
    recorder = DemoRecorder(
        output_dir=args.output,
        task=args.task,
        fps=args.fps,
    )
    
    if args.simulated:
        print("Running with simulated leader arm (for testing)")
        recorder.run_with_simulated_leader()
    else:
        print(f"Running with real leader arm on {args.port}")
        recorder.run_with_leader_arm(args.port)


if __name__ == "__main__":
    main()
