#!/usr/bin/env python3
"""
Calibration script for SO-ARM100 leader arm and Mujoco simulation.

This script helps calibrate the mapping between the real leader arm positions
and the Mujoco simulated robot positions to reduce oscillation and improve control.
"""

import time
import numpy as np
import json
import os
from simple_control import SO100LeaderReader
import src.so_arm_env as so_arm_env


class ArmCalibrator:
    """Calibrates the mapping between leader arm and Mujoco robot."""

    def __init__(self, calibration_file="calibration.json"):
        self.calibration_file = calibration_file
        self.leader_reader = SO100LeaderReader(port='/dev/ttyACM1')  # Use the correct port
        self.env = None
        self.calibration_data = {
            'home_positions': {},
            'joint_mappings': {},
            'offsets': {},
            'scales': {}
        }

    def connect_hardware(self):
        """Connect to leader arm and initialize Mujoco environment."""
        print("Connecting to leader arm...")
        if not self.leader_reader.connect():
            print("Failed to connect to leader arm!")
            return False

        print("Initializing Mujoco environment...")
        try:
            self.env = so_arm_env.SoArm100Env(render_mode=None)  # Headless for calibration
        except Exception as e:
            print(f"Failed to initialize Mujoco: {e}")
            return False

        return True

    def record_home_position(self):
        """Record the home/rest position of both leader arm and Mujoco."""
        print("\n=== Recording Home Position ===")
        print("Please place the leader arm in its HOME/REST position.")
        print("This should match the Mujoco home position defined in so_arm100.xml")
        input("Press Enter when ready...")

        # Record leader arm positions
        leader_home = self.leader_reader.get_action()
        self.calibration_data['home_positions']['leader'] = leader_home

        # Record Mujoco home position (from keyframe)
        mujoco_home = self.env._home_qpos.copy()
        self.calibration_data['home_positions']['mujoco'] = mujoco_home.tolist()

        print(f"Leader home positions: {leader_home}")
        print(f"Mujoco home positions: {mujoco_home}")

    def record_joint_ranges(self):
        """Record joint ranges by moving leader arm to extremes."""
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

        print("\n=== Recording Joint Ranges ===")
        print("For each joint, move the leader arm to MIN and MAX positions.")
        print("The script will record the positions automatically.")

        for i, joint_name in enumerate(joint_names):
            print(f"\n--- Calibrating {joint_name} ---")
            print(f"Move leader arm to MINIMUM position for {joint_name}")
            input("Press Enter when ready...")

            # Record minimum position
            min_pos = self.leader_reader.get_action()[f'{joint_name}.pos']
            print(f"Min position recorded: {min_pos}")

            print(f"Move leader arm to MAXIMUM position for {joint_name}")
            input("Press Enter when ready...")

            # Record maximum position
            max_pos = self.leader_reader.get_action()[f'{joint_name}.pos']
            print(f"Max position recorded: {max_pos}")

            # Store the mapping
            self.calibration_data['joint_mappings'][joint_name] = {
                'leader_min': float(min_pos),
                'leader_max': float(max_pos),
                'leader_range': float(max_pos - min_pos)
            }

    def compute_calibration(self):
        """Compute offset and scaling factors from collected data."""
        print("\n=== Computing Calibration Parameters ===")

        # Get Mujoco joint ranges from the model
        mujoco_ranges = {}
        for i in range(self.env.model.nq):
            joint_name = self.env.model.joint(i).name
            if joint_name:
                joint_name = joint_name.decode() if isinstance(joint_name, bytes) else joint_name
                range_min = self.env.model.joint(i).range[0]
                range_max = self.env.model.joint(i).range[1]
                mujoco_ranges[joint_name] = {
                    'min': float(range_min),
                    'max': float(range_max),
                    'range': float(range_max - range_min)
                }

        # Map leader joints to Mujoco joints
        joint_mapping = {
            'shoulder_pan': 'Rotation',
            'shoulder_lift': 'Pitch',
            'elbow_flex': 'Elbow',
            'wrist_flex': 'Wrist_Pitch',
            'wrist_roll': 'Wrist_Roll',
            'gripper': 'Jaw'
        }

        for leader_joint, mujoco_joint in joint_mapping.items():
            if leader_joint in self.calibration_data['joint_mappings'] and mujoco_joint in mujoco_ranges:
                leader_data = self.calibration_data['joint_mappings'][leader_joint]
                mujoco_data = mujoco_ranges[mujoco_joint]

                # Compute scaling factor (Mujoco range / Leader range)
                scale = mujoco_data['range'] / leader_data['leader_range'] if leader_data['leader_range'] != 0 else 1.0

                # Compute offset to align home positions
                leader_home = self.calibration_data['home_positions']['leader'][f'{leader_joint}.pos']
                mujoco_home = self.calibration_data['home_positions']['mujoco'][self.env.model.joint(mujoco_joint).qposadr[0]]

                # Offset = Mujoco_home - (Leader_home * scale)
                offset = mujoco_home - (leader_home * scale)

                self.calibration_data['offsets'][leader_joint] = float(offset)
                self.calibration_data['scales'][leader_joint] = float(scale)

                print(f"{leader_joint} -> {mujoco_joint}:")
                print(f"  Scale: {scale:.4f}, Offset: {offset:.4f}")

    def save_calibration(self):
        """Save calibration data to file."""
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"\nCalibration saved to {self.calibration_file}")

    def load_calibration(self):
        """Load existing calibration data."""
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration loaded from {self.calibration_file}")
            return True
        return False

    def apply_calibration(self, leader_positions):
        """Apply calibration to leader arm positions to get Mujoco positions."""
        calibrated = []
        joint_order = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

        for joint in joint_order:
            raw_pos = leader_positions[f'{joint}.pos']
            if joint in self.calibration_data['scales'] and joint in self.calibration_data['offsets']:
                scale = self.calibration_data['scales'][joint]
                offset = self.calibration_data['offsets'][joint]
                calibrated_pos = (raw_pos * scale) + offset
                calibrated.append(calibrated_pos)
            else:
                # Fallback to raw position if no calibration
                calibrated.append(raw_pos)

        return np.array(calibrated, dtype=np.float32)

    def run_calibration(self):
        """Run the complete calibration process."""
        if not self.connect_hardware():
            return False

        try:
            self.record_home_position()
            self.record_joint_ranges()
            self.compute_calibration()
            self.save_calibration()
            print("\nCalibration completed successfully!")
            return True

        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

        finally:
            self.leader_reader.disconnect()
            if self.env:
                self.env.close()


def main():
    calibrator = ArmCalibrator()

    print("SO-ARM100 Calibration Tool")
    print("This will help calibrate your leader arm with the Mujoco simulation.")
    print("Make sure your leader arm is connected and powered on.")

    if input("Start calibration? (y/n): ").lower() == 'y':
        success = calibrator.run_calibration()
        if success:
            print("\nTo use the calibration in your control script:")
            print("1. Copy calibration.json to your working directory")
            print("2. Modify simple_control.py to load and apply calibration")
        else:
            print("Calibration failed. Please try again.")


if __name__ == "__main__":
    main()