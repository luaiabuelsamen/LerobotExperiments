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
from src.so_arm_env import HAS_DATASET_RECORDER


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


def keyboard_listener(env):
    """Listen for keyboard input in a separate thread."""
    print("Keyboard controls:")
    print("  R - Reset episode")
    print("  Z - Randomize block position")
    print("  S - Start recording (video + dataset)")
    print("  F - Finish/save recording")
    print("  Q - Quit")
    
    while True:
        try:
            key = get_key().lower()
            if key == 'r':
                print("Resetting episode...")
                env.reset()
            elif key == 'z':
                print("Randomizing block position...")
                env._randomize_environment()
            elif key == 's':
                # Start both video and dataset recording
                if not env.recording:
                    env.start_recording()
                if HAS_DATASET_RECORDER and not env.dataset_recording:
                    env.start_dataset_recording()
            elif key == 'f':
                # Stop both video and dataset recording
                if env.recording:
                    env.stop_recording()
                if HAS_DATASET_RECORDER and env.dataset_recording:
                    env.save_dataset_episode()
            elif key == 'q':
                print("Quitting...")
                break
        except Exception as e:
            print(f"Keyboard error: {e}")
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
    print("SO100 Leader Arm Control for Mujoco SO-ARM100")
    
    # Create leader arm reader and Mujoco environment
    leader_arm = SO100LeaderReader(port='/dev/ttyACM1')
    
    try:
        env = so_arm_env.SoArm100Env(render_mode="human", camera_mode="first_person")
    except Exception as e:
        if "GLFW" in str(e) or "DISPLAY" in str(e):
            print("No display available, using headless mode")
            env = so_arm_env.SoArm100Env(render_mode=None, camera_mode="first_person")
        else:
            raise
    
    try:
        # Connect to leader arm
        print("Connecting to leader arm...")
        if not leader_arm.connect():
            print("Failed to connect to leader arm. Using simulated control instead.")
            # Fall back to simulated control
            leader_arm = SimulatedLeaderArm()
            leader_arm.connect()
        
        if not leader_arm.is_connected:
            print("No leader arm available!")
            return
        
        print("Connected! Move the leader arm to control the Mujoco robot.")
        print("Press Ctrl+C to exit.")
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=keyboard_listener, args=(env,), daemon=True)
        keyboard_thread.start()
        
        # Reset Mujoco environment
        obs, info = env.reset()
        
        # Control loop
        while True:
            # Read leader arm action
            leader_action = leader_arm.get_action()
            
            # Apply calibration to get Mujoco-compatible positions
            action = leader_arm.apply_calibration(leader_action)
            
            # Step the Mujoco environment with the calibrated action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print action for debugging
            # print(f"Action: after {action} before {leader_action}")
            # Small delay to control loop rate
            time.sleep(0.01)
            
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping control...")
    
    except Exception as e:
        print(f"Error: {e}")
    
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