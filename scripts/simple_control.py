#!/usr/bin/env python3
"""
Simple script to control Mujoco SO-ARM100 robot using lerobot leader arm.
Standalone implementation with direct serial communication to Feetech motors.
"""

import time
import serial
import struct
import numpy as np
import json
import os
import threading
import sys
import termios
import tty

# Import the Mujoco environment
import src.so_arm_env as so_arm_env

# Global lock for thread-safe mujoco operations
env_lock = threading.Lock()


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


def randomize_block_position(env):
    """Randomize block position anywhere on the table, avoiding box and robot areas."""
    # Import mujoco from the environment
    import mujoco
    
    # Table bounds (based on scene.xml: table size 0.3x0.4 at pos 0, -0.15, 0.07)
    table_x_min, table_x_max = -0.28, 0.28  # Leave some margin from edges
    table_y_min, table_y_max = -0.53, 0.23  # Leave some margin from edges
    table_z = 0.115  # Height on table surface (table top at 0.09 + block height)
    
    # Box bounds to avoid (box at pos -0.1, -0.35, with size 0.06x0.06)
    box_x_min, box_x_max = -0.18, -0.02  # Box area with safety margin
    box_y_min, box_y_max = -0.43, -0.27  # Box area with safety margin
    
    # Robot base area to avoid (around origin)
    robot_x_min, robot_x_max = -0.12, 0.12  # Robot base area with margin
    robot_y_min, robot_y_max = -0.05, 0.15  # Robot base area with margin
    
    max_attempts = 50
    for attempt in range(max_attempts):
        # Generate random position on table
        block_x = np.random.uniform(table_x_min, table_x_max)
        block_y = np.random.uniform(table_y_min, table_y_max)
        
        # Check if position conflicts with box area
        in_box_area = (box_x_min <= block_x <= box_x_max and 
                       box_y_min <= block_y <= box_y_max)
        
        # Check if position conflicts with robot area
        in_robot_area = (robot_x_min <= block_x <= robot_x_max and 
                         robot_y_min <= block_y <= robot_y_max)
        
        # If no conflicts, use this position
        if not in_box_area and not in_robot_area:
            # THREAD SAFE: Lock before modifying mujoco state
            with env_lock:
                # Set block position (free joint body)
                block_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "block")
                if block_id >= 0:
                    # For free joint bodies, qpos has 7 values: 3 position + 4 quaternion
                    qpos_start = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "block")]
                    env.data.qpos[qpos_start:qpos_start+3] = [block_x, block_y, table_z]
                    env.data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # quaternion identity
                    
                    # Reset velocities
                    qvel_start = env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "block")]
                    env.data.qvel[qvel_start:qvel_start+6] = 0.0
                
                # Forward kinematics to update derived quantities
                mujoco.mj_forward(env.model, env.data)
            
            print(f"Block randomized to position: ({block_x:.3f}, {block_y:.3f}, {table_z:.3f})")
            return
    
    print(f"Warning: Could not find valid random position after {max_attempts} attempts. Using default position.")
    # Fallback to a safe default position
    block_x, block_y = 0.15, -0.25
    block_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "block")
    if block_id >= 0:
        qpos_start = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "block")]
        env.data.qpos[qpos_start:qpos_start+3] = [block_x, block_y, table_z]
        env.data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]
        qvel_start = env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "block")]
        env.data.qvel[qvel_start:qvel_start+6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def keyboard_listener(env):
    """Listen for keyboard input in a separate thread."""
    print("\n" + "="*60)
    print("KEYBOARD CONTROLS:")
    print("="*60)
    print("  R - Reset episode (reset robot to home position)")
    print("  Z - Randomize block position on table")
    print("  S - Start recording (press when ready to demonstrate)")
    print("  F - Finish/save recording (press when task complete)")
    print("  Q - Quit program")
    print("="*60)
    print("\nTIP: Press 'Z' to randomize block, 'R' to reset robot,")
    print("     then 'S' to start recording your demonstration.")
    print("     Complete the task, then press 'F' to save.\n")
    
    while True:
        try:
            key = get_key().lower()
            if key == 'r':
                print("\n[RESET] Resetting robot to home position...")
                with env_lock:
                    obs, info = env.reset()
                print("[RESET] Robot reset complete.\n")
            elif key == 'z':
                print("\n[RANDOMIZE] Randomizing block position...")
                randomize_block_position(env)
            elif key == 's':
                # Start both video and dataset recording
                if env.dataset_recording:
                    print("\n[WARNING] Already recording! Press 'F' to finish current episode first.\n")
                else:
                    env.start_dataset_recording()
            elif key == 'f':
                # Stop both video and dataset recording
                if env.dataset_recording:
                    env.save_dataset_episode()
            elif key == 'q':
                print("\n[QUIT] Shutting down...")
                if env.dataset_recording:
                    print("[QUIT] Saving current recording before exit...")
                    env.save_dataset_episode()
                break
        except Exception as e:
            print(f"\n[ERROR] Keyboard error: {e}\n")
            break


class SO100LeaderReader:
    """Direct serial communication with SO100 leader arm Feetech motors."""
    
    def __init__(self, port='/dev/ttyACM1', baudrate=1000000, calibration_file="configs/calibration.json"):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.calibration_file = calibration_file
        self.calibration_data = None
        
        # SO100 motor IDs (based on lerobot configuration)
        self.motors = {
            'shoulder_pan': 1,
            'shoulder_lift': 2, 
            'elbow_flex': 3,
            'wrist_flex': 4,
            'wrist_roll': 5,
            'gripper': 6
        }
        
        # Position scaling (approximate, may need calibration)
        self.position_scale = 2047.5  # 12-bit resolution centered at 0
        
        # Load calibration if available
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data from file if it exists."""
        if not os.path.isfile(self.calibration_file):
            print(f"No calibration file found at {self.calibration_file}, using default values.")
            return
        
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            print(f"Loaded calibration data from {self.calibration_file}")
        except Exception as e:
            print(f"Error loading calibration file: {e}. Using default values.")
            self.calibration_data = None
    
    def connect(self):
        """Connect to the serial port."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(0.1)  # Allow time for connection
            print(f"Connected to SO100 leader arm at {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to leader arm: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial:
            self.serial.close()
            self.serial = None
    
    @property
    def is_connected(self):
        return self.serial is not None and self.serial.is_open
    
    def read_motor_position(self, motor_id):
        """Read position from a single Feetech motor."""
        if not self.serial:
            return None
            
        # SCServo protocol for reading present position
        servo_id = motor_id
        address = 56  # Present_Position register
        length = 2    # 2 bytes for position
        
        # Build read command packet
        packet = bytearray()
        packet.append(0xFF)  # Header 1
        packet.append(0xFF)  # Header 2
        packet.append(servo_id)  # Motor ID
        packet.append(0x04)  # Length (instruction + params)
        packet.append(0x02)  # Read instruction
        packet.append(address)  # Address
        packet.append(length)   # Data length
        
        # Calculate checksum
        checksum = servo_id + 0x04 + 0x02 + address + length
        checksum = ~checksum & 0xFF
        packet.append(checksum)
        
        try:
            # Send command
            self.serial.write(packet)
            self.serial.flush()
            
            # Read response (simplified - may need adjustment based on exact protocol)
            response = self.serial.read(8)
            
            if len(response) >= 8 and response[0] == 0xFF and response[1] == 0xFF:
                # Extract position (little endian, 2 bytes)
                pos_low = response[5]
                pos_high = response[6]
                position = pos_low + (pos_high << 8)
                return position
        except:
            pass
        
        return None
    
    def get_action(self):
        """Get positions from all motors and return as action dict."""
        positions = {}
        
        for name, motor_id in self.motors.items():
            raw_pos = self.read_motor_position(motor_id)
            if raw_pos is not None:
                # Convert to normalized range (-1 to 1)
                # Feetech motors typically use 0-4095 range
                normalized = (raw_pos / self.position_scale) - 1.0
                normalized = np.clip(normalized, -1.0, 1.0)
                positions[f'{name}.pos'] = normalized
            else:
                # Use zero if reading fails
                positions[f'{name}.pos'] = 0.0
        
        return positions
    
    def load_calibration(self):
        """Load calibration data from file."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"Loaded calibration from {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Failed to load calibration: {e}")
        else:
            print(f"No calibration file found at {self.calibration_file}")
        return False
    
    def apply_calibration(self, raw_positions):
        """Apply calibration to raw leader positions to get calibrated Mujoco positions."""
        if not self.calibration_data:
            # No calibration, return raw positions
            joint_order = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            return np.array([raw_positions[f'{joint}.pos'] for joint in joint_order], dtype=np.float32)
        
        # Mujoco joint limits (from so_arm100.xml)
        mujoco_limits = {
            'shoulder_pan': (-2.2, 2.2),      # Rotation
            'shoulder_lift': (-3.14158, 0.2), # Pitch
            'elbow_flex': (0, 3.14158),       # Elbow
            'wrist_flex': (-2.0, 1.8),        # Wrist_Pitch
            'wrist_roll': (-3.14158, 3.14158), # Wrist_Roll
            'gripper': (-0.2, 2.0)            # Jaw
        }
        
        calibrated = []
        joint_order = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        joint_mappings = self.calibration_data.get('joint_mappings', {})
        
        for joint in joint_order:
            raw_pos = raw_positions[f'{joint}.pos']
            
            if joint in joint_mappings:
                # Get leader arm range from calibration
                leader_min = joint_mappings[joint]['leader_min']
                leader_max = joint_mappings[joint]['leader_max']
                
                # Get Mujoco range
                mujoco_min, mujoco_max = mujoco_limits[joint]
                
                # Normalize leader position to 0-1 range
                leader_range = leader_max - leader_min
                if abs(leader_range) > 0.001:  # Avoid division by zero
                    normalized = (raw_pos - leader_min) / leader_range
                else:
                    normalized = 0.5
                
                # Map to Mujoco range
                mujoco_range = mujoco_max - mujoco_min
                calibrated_pos = mujoco_min + (normalized * mujoco_range)
                
                # Clamp to Mujoco joint limits
                calibrated_pos = np.clip(calibrated_pos, mujoco_min, mujoco_max)
                
                calibrated.append(calibrated_pos)
            else:
                # Fallback to raw position if no calibration for this joint
                calibrated.append(raw_pos)
        
        return np.array(calibrated, dtype=np.float32)
        

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SO100 Leader Arm Control for Mujoco")
    parser.add_argument("--new", action="store_true", help="Create new dataset folder instead of continuing existing")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Explicit recordings path (default: auto-continue last dataset in ./recordings)",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("SO100 Leader Arm Control for Mujoco SO-ARM100")
    print("QUICK DATA COLLECTION MODE")
    if args.new:
        print("[NEW DATASET MODE - Will create new folder]")
    if args.dataset_path:
        print(f"[DATASET PATH] {args.dataset_path}")
    print("="*60)
    
    # Create leader arm reader and Mujoco environment
    leader_arm = SO100LeaderReader(port='/dev/ttyACM1')
    
    try:
        env = so_arm_env.SoArm100Env(
            render_mode="human",
            camera_mode="first_person",
            force_new_dataset=args.new,
            dataset_path=args.dataset_path,
            enable_viewer_hotkeys=False,
        )
    except Exception as e:
        if "GLFW" in str(e) or "DISPLAY" in str(e):
            print("No display available, using headless mode")
            env = so_arm_env.SoArm100Env(
                render_mode=None,
                camera_mode="first_person",
                force_new_dataset=args.new,
                dataset_path=args.dataset_path,
                enable_viewer_hotkeys=False,
            )
        else:
            raise
    
    try:
        # Connect to leader arm
        print("\nConnecting to leader arm...")
        if not leader_arm.connect():
            print("Failed to connect to leader arm. Using simulated control instead.")
            # Fall back to simulated control
            leader_arm = SimulatedLeaderArm()
            leader_arm.connect()
        
        if not leader_arm.is_connected:
            print("No leader arm available!")
            return
        
        print("Connected! Move the leader arm to control the Mujoco robot.")
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=keyboard_listener, args=(env,), daemon=True)
        keyboard_thread.start()
        
        # Reset Mujoco environment
        obs, info = env.reset()
        
        # Control loop
        print("\nReady! Waiting for your commands...\n")
        while True:
            # Read leader arm action
            leader_action = leader_arm.get_action()
            
            # Apply calibration to get Mujoco-compatible positions
            action = leader_arm.apply_calibration(leader_action)
            
            # Step the Mujoco environment with the calibrated action (thread-safe)
            with env_lock:
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Small delay to control loop rate
            time.sleep(0.01)
            
            if terminated or truncated:
                print("\n[INFO] Episode ended, resetting...\n")
                with env_lock:
                    obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n\n[EXIT] Stopping control...")
        if env.dataset_recording:
            print("[EXIT] Saving current recording before exit...")
            try:
                env.save_dataset_episode()
            except:
                pass
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    finally:
        # Clean up
        leader_arm.disconnect()
        env.close()
        print("Disconnected and cleaned up.")


class SimulatedLeaderArm:
    """Simulated leader arm for testing when real hardware is not available."""
    
    def __init__(self):
        self.connected = False
        self.time = 0
        
    def connect(self):
        self.connected = True
        print("Using simulated leader arm (no hardware connected)")
        return True
        
    def disconnect(self):
        self.connected = False
        
    @property
    def is_connected(self):
        return self.connected
        
    def get_action(self):
        """Generate simulated joint movements."""
        self.time += 0.05
        
        # Create smooth oscillating movements for each joint
        return {
            'shoulder_pan.pos': 0.5 * np.sin(self.time * 0.5),
            'shoulder_lift.pos': 0.3 * np.sin(self.time * 0.7 + 1),
            'elbow_flex.pos': 0.4 * np.sin(self.time * 0.6 + 2),
            'wrist_flex.pos': 0.2 * np.sin(self.time * 0.8 + 3),
            'wrist_roll.pos': 0.6 * np.sin(self.time * 0.4 + 4),
            'gripper.pos': 0.1 * np.sin(self.time * 0.3 + 5) + 0.5
        }


if __name__ == "__main__":
    main()