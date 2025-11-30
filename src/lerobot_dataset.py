#!/usr/bin/env python3
"""
Standalone LeRobot Dataset Format Writer

This module provides a minimal implementation for recording robot data in the LeRobot v3.0 format,
compatible with ACT and other policies. No lerobot dependencies required.

Dataset Structure:
    dataset_name/
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet
    └── videos/
        └── observation.images.front/
            └── chunk-000/
                └── file-000.mp4
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Optional imports - fail gracefully if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Install with: pip install pandas pyarrow")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv not available. Install with: pip install opencv-python")


# Constants matching LeRobot v3.0
CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_MB = 100
DEFAULT_VIDEO_FILE_SIZE_MB = 500

# Path patterns
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file."""
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(fpath: Path) -> dict:
    """Load data from a JSON file."""
    with open(fpath) as f:
        return json.load(f)


def serialize_stats(stats: dict) -> dict:
    """Convert numpy arrays to lists for JSON serialization."""
    result = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            result[key] = serialize_stats(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def compute_feature_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute min, max, mean, std statistics for a feature array."""
    return {
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "count": np.array([len(data)]),
    }


def compute_image_stats(images: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute statistics for images (sample and normalize to 0-1)."""
    if len(images) == 0:
        return {}
    
    # Sample images if too many
    num_samples = min(len(images), 100)
    indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)
    sampled = [images[i] for i in indices]
    
    # Stack and normalize to 0-1
    stacked = np.stack(sampled).astype(np.float32) / 255.0
    
    # Compute stats per channel (C, H, W format)
    if stacked.ndim == 4:  # (N, H, W, C)
        stacked = stacked.transpose(0, 3, 1, 2)  # to (N, C, H, W)
    
    # Per-channel stats
    return {
        "min": np.min(stacked, axis=(0, 2, 3), keepdims=True).squeeze((2, 3)),
        "max": np.max(stacked, axis=(0, 2, 3), keepdims=True).squeeze((2, 3)),
        "mean": np.mean(stacked, axis=(0, 2, 3), keepdims=True).squeeze((2, 3)),
        "std": np.std(stacked, axis=(0, 2, 3), keepdims=True).squeeze((2, 3)),
        "count": np.array([len(images)]),
    }


def aggregate_stats(stats_list: List[Dict]) -> Dict:
    """Aggregate statistics from multiple episodes using parallel algorithm."""
    if not stats_list:
        return {}
    if len(stats_list) == 1:
        return stats_list[0]
    
    result = {}
    all_keys = set()
    for stats in stats_list:
        all_keys.update(stats.keys())
    
    for key in all_keys:
        feat_stats = [s[key] for s in stats_list if key in s]
        if not feat_stats:
            continue
            
        means = np.stack([s["mean"] for s in feat_stats])
        variances = np.stack([s["std"] ** 2 for s in feat_stats])
        counts = np.stack([s["count"] for s in feat_stats])
        total_count = counts.sum(axis=0)
        
        # Match dimensions for weighted computation
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        
        weighted_means = means * counts
        total_mean = weighted_means.sum(axis=0) / total_count
        
        delta_means = means - total_mean
        weighted_variances = (variances + delta_means ** 2) * counts
        total_variance = weighted_variances.sum(axis=0) / total_count
        
        result[key] = {
            "min": np.min(np.stack([s["min"] for s in feat_stats]), axis=0),
            "max": np.max(np.stack([s["max"] for s in feat_stats]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_variance),
            "count": np.array([int(total_count.flat[0])]),
        }
    
    return result


def encode_video_ffmpeg(image_dir: Path, output_path: Path, fps: int = 30):
    """Encode images to video using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use glob pattern for ffmpeg
    input_pattern = str(image_dir / "frame-*.png")
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", input_pattern,
        "-c:v", "libx264",  # H.264 codec (widely compatible)
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install ffmpeg.")
        return False


class LeRobotDatasetRecorder:
    """
    Standalone recorder for LeRobot v3.0 format datasets.
    
    Usage:
        recorder = LeRobotDatasetRecorder.create(
            root="./my_dataset",
            fps=30,
            robot_type="so100",
            features={
                "action": {"dtype": "float32", "shape": [6], "names": ["j1", "j2", ...]},
                "observation.state": {"dtype": "float32", "shape": [6], "names": ["j1", "j2", ...]},
                "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            }
        )
        
        # Start recording an episode
        recorder.start_episode(task="Pick up the block")
        
        # Add frames
        recorder.add_frame({
            "action": np.array([...]),
            "observation.state": np.array([...]),
            "observation.images.front": image_array,
        })
        
        # Save episode
        recorder.save_episode()
        
        # Finalize dataset
        recorder.finalize()
    """
    
    def __init__(
        self,
        root: Path,
        fps: int,
        features: Dict[str, Dict],
        robot_type: str = "unknown",
    ):
        self.root = Path(root)
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        
        # Episode state
        self.episode_buffer: Dict[str, List] = {}
        self.episode_images: Dict[str, List[np.ndarray]] = {}
        self.current_episode_index = 0
        self.total_frames = 0
        self.all_episode_stats: List[Dict] = []
        self.tasks: Dict[str, int] = {}  # task_name -> task_index
        self.episodes_metadata: List[Dict] = []
        
        # Identify video/image keys
        self.video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        self.image_keys = [k for k, v in features.items() if v.get("dtype") == "image"]
        self.camera_keys = self.video_keys + self.image_keys
        
    @classmethod
    def create(
        cls,
        root: str | Path,
        fps: int,
        features: Dict[str, Dict],
        robot_type: str = "unknown",
    ) -> "LeRobotDatasetRecorder":
        """Create a new dataset recorder."""
        root = Path(root)
        
        if root.exists():
            raise ValueError(f"Dataset directory already exists: {root}")
        
        root.mkdir(parents=True)
        
        # Add default features
        default_features = {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        
        # Merge features
        all_features = {**features, **default_features}
        
        # Add video info for video features
        for key, feat in all_features.items():
            if feat.get("dtype") == "video":
                shape = feat["shape"]
                feat["names"] = feat.get("names", ["height", "width", "channels"])
                feat["info"] = {
                    "video.height": shape[0] if len(shape) >= 2 else 480,
                    "video.width": shape[1] if len(shape) >= 2 else 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": shape[2] if len(shape) >= 3 else 3,
                    "has_audio": False,
                }
        
        # Create info.json
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_MB,
            "fps": fps,
            "splits": {"train": "0:0"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": all_features,
        }
        
        write_json(info, root / INFO_PATH)
        
        return cls(root, fps, all_features, robot_type)
    
    @classmethod
    def load(cls, root: str | Path) -> "LeRobotDatasetRecorder":
        """Load an existing dataset to continue recording."""
        root = Path(root)
        info = load_json(root / INFO_PATH)
        
        recorder = cls(
            root=root,
            fps=info["fps"],
            features=info["features"],
            robot_type=info.get("robot_type", "unknown"),
        )
        
        recorder.current_episode_index = info["total_episodes"]
        recorder.total_frames = info["total_frames"]
        
        # Load existing tasks
        tasks_path = root / TASKS_PATH
        if tasks_path.exists() and HAS_PANDAS:
            tasks_df = pd.read_parquet(tasks_path)
            recorder.tasks = {row.name: row.task_index for _, row in tasks_df.iterrows()}
        
        # Load existing stats
        stats_path = root / STATS_PATH
        if stats_path.exists():
            recorder.all_episode_stats = [load_json(stats_path)]
        
        return recorder
    
    def start_episode(self, task: str = "default_task"):
        """Start recording a new episode."""
        self.current_task = task
        
        # Register task if new
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        
        # Initialize buffers
        self.episode_buffer = {
            "timestamp": [],
            "frame_index": [],
            "action": [],
            "observation.state": [],
            "task_index": [],
        }
        
        self.episode_images = {key: [] for key in self.camera_keys}
        self.episode_frame_count = 0
        
    def add_frame(
        self,
        action: np.ndarray,
        state: np.ndarray,
        images: Optional[Dict[str, np.ndarray]] = None,
        timestamp: Optional[float] = None,
    ):
        """Add a frame to the current episode."""
        if timestamp is None:
            timestamp = self.episode_frame_count / self.fps
        
        # Add to buffer
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["frame_index"].append(self.episode_frame_count)
        self.episode_buffer["action"].append(action.astype(np.float32))
        self.episode_buffer["observation.state"].append(state.astype(np.float32))
        self.episode_buffer["task_index"].append(self.tasks[self.current_task])
        
        # Store images
        if images:
            for key, img in images.items():
                if key in self.episode_images:
                    self.episode_images[key].append(img)
        
        self.episode_frame_count += 1
    
    def save_episode(self) -> int:
        """Save the current episode to disk."""
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required to save episodes")
        
        episode_index = self.current_episode_index
        episode_length = self.episode_frame_count
        
        if episode_length == 0:
            print("Warning: Empty episode, skipping save")
            return -1
        
        # Create data arrays
        actions = np.stack(self.episode_buffer["action"])
        states = np.stack(self.episode_buffer["observation.state"])
        timestamps = np.array(self.episode_buffer["timestamp"], dtype=np.float32)
        frame_indices = np.array(self.episode_buffer["frame_index"], dtype=np.int64)
        task_indices = np.array(self.episode_buffer["task_index"], dtype=np.int64)
        
        # Global indices
        global_indices = np.arange(self.total_frames, self.total_frames + episode_length, dtype=np.int64)
        episode_indices = np.full(episode_length, episode_index, dtype=np.int64)
        
        # Build dataframe
        data = {
            "index": global_indices,
            "episode_index": episode_indices,
            "frame_index": frame_indices,
            "timestamp": timestamps,
            "action": list(actions),  # Store arrays as objects
            "observation.state": list(states),
            "task_index": task_indices,
        }
        
        df = pd.DataFrame(data)
        
        # Save parquet
        data_path = self.root / DATA_PATH.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to existing or create new
        if data_path.exists():
            existing_df = pd.read_parquet(data_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(data_path, index=False)
        
        # Save images and encode video
        video_metadata = {}
        for video_key in self.video_keys:
            if video_key in self.episode_images and self.episode_images[video_key]:
                video_metadata.update(
                    self._save_episode_video(video_key, episode_index)
                )
        
        # Compute episode statistics
        ep_stats = {
            "action": compute_feature_stats(actions),
            "observation.state": compute_feature_stats(states),
        }
        
        for video_key in self.video_keys:
            if video_key in self.episode_images and self.episode_images[video_key]:
                ep_stats[video_key] = compute_image_stats(self.episode_images[video_key])
        
        self.all_episode_stats.append(ep_stats)
        
        # Save episode metadata
        episode_meta = {
            "episode_index": episode_index,
            "tasks": [self.current_task],
            "length": episode_length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": self.total_frames,
            "dataset_to_index": self.total_frames + episode_length,
        }
        episode_meta.update(video_metadata)
        self.episodes_metadata.append(episode_meta)
        
        self._save_episodes_metadata()
        
        # Update totals
        self.total_frames += episode_length
        self.current_episode_index += 1
        
        # Update info.json
        self._update_info()
        
        # Update tasks
        self._save_tasks()
        
        # Update stats
        self._save_stats()
        
        print(f"Episode {episode_index} saved: {episode_length} frames")
        return episode_index
    
    def _save_episode_video(self, video_key: str, episode_index: int) -> Dict:
        """Save episode images as video."""
        images = self.episode_images[video_key]
        if not images:
            return {}
        
        # Create temp directory for images
        temp_dir = self.root / "temp_images" / video_key / f"episode-{episode_index:06d}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images as PNGs
        for i, img in enumerate(images):
            img_path = temp_dir / f"frame-{i:06d}.png"
            if HAS_CV2:
                # Convert RGB to BGR for OpenCV
                if img.ndim == 3 and img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
                cv2.imwrite(str(img_path), img_bgr)
        
        # Encode video
        video_path = self.root / VIDEO_PATH.format(
            video_key=video_key, chunk_index=0, file_index=0
        )
        
        # For simplicity, we create one video per episode and concatenate later
        # In production, you'd concatenate to existing video file
        temp_video = temp_dir.parent / f"episode-{episode_index:06d}.mp4"
        
        if encode_video_ffmpeg(temp_dir, temp_video, self.fps):
            # Move/concatenate to final location
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            if video_path.exists():
                # Concatenate with existing video
                self._concatenate_videos([video_path, temp_video], video_path)
                temp_video.unlink()
            else:
                shutil.move(str(temp_video), str(video_path))
        
        # Clean up temp images
        shutil.rmtree(temp_dir)
        
        # Get video duration
        video_duration = len(images) / self.fps
        
        return {
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
            f"videos/{video_key}/from_timestamp": 0.0,  # Simplified
            f"videos/{video_key}/to_timestamp": video_duration,
        }
    
    def _concatenate_videos(self, input_paths: List[Path], output_path: Path):
        """Concatenate multiple videos using ffmpeg."""
        # Create concat file
        concat_file = output_path.parent / "concat_list.txt"
        with open(concat_file, "w") as f:
            for path in input_paths:
                f.write(f"file '{path}'\n")
        
        temp_output = output_path.parent / "temp_concat.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(temp_output)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            shutil.move(str(temp_output), str(output_path))
        except subprocess.CalledProcessError as e:
            print(f"Video concatenation failed: {e}")
        finally:
            concat_file.unlink(missing_ok=True)
    
    def _save_episodes_metadata(self):
        """Save episodes metadata to parquet."""
        if not HAS_PANDAS or not self.episodes_metadata:
            return
        
        df = pd.DataFrame(self.episodes_metadata)
        
        # Add stats columns (flattened)
        for i, ep_stats in enumerate(self.all_episode_stats):
            for feat_key, stats in ep_stats.items():
                for stat_name, stat_val in stats.items():
                    col_name = f"stats/{feat_key}/{stat_name}"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = stat_val.tolist() if isinstance(stat_val, np.ndarray) else stat_val
        
        episodes_path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(episodes_path, index=False)
    
    def _save_tasks(self):
        """Save tasks to parquet."""
        if not HAS_PANDAS or not self.tasks:
            return
        
        df = pd.DataFrame({
            "task_index": list(self.tasks.values()),
        }, index=list(self.tasks.keys()))
        
        tasks_path = self.root / TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tasks_path)
    
    def _save_stats(self):
        """Save aggregated statistics."""
        if not self.all_episode_stats:
            return
        
        aggregated = aggregate_stats(self.all_episode_stats)
        serialized = serialize_stats(aggregated)
        write_json(serialized, self.root / STATS_PATH)
    
    def _update_info(self):
        """Update info.json with current totals."""
        info = load_json(self.root / INFO_PATH)
        info["total_episodes"] = self.current_episode_index
        info["total_frames"] = self.total_frames
        info["total_tasks"] = len(self.tasks)
        info["splits"] = {"train": f"0:{self.current_episode_index}"}
        write_json(info, self.root / INFO_PATH)
    
    def finalize(self):
        """Finalize the dataset (call when done recording)."""
        self._save_stats()
        self._update_info()
        print(f"Dataset finalized: {self.current_episode_index} episodes, {self.total_frames} frames")


# Convenience function for simple usage
def create_dataset_recorder(
    root: str,
    fps: int = 30,
    robot_type: str = "so100",
    action_dim: int = 6,
    state_dim: int = 6,
    image_height: int = 480,
    image_width: int = 640,
    action_names: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
) -> LeRobotDatasetRecorder:
    """
    Create a dataset recorder with common defaults for SO-ARM100.
    
    Args:
        root: Directory to save dataset
        fps: Frames per second
        robot_type: Robot type identifier
        action_dim: Dimension of action vector
        state_dim: Dimension of state vector
        image_height: Height of camera images
        image_width: Width of camera images
        action_names: Names for action dimensions
        state_names: Names for state dimensions
    
    Returns:
        LeRobotDatasetRecorder instance
    """
    if action_names is None:
        action_names = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ][:action_dim]
    
    if state_names is None:
        state_names = action_names.copy()
    
    features = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
        },
    }
    
    return LeRobotDatasetRecorder.create(
        root=root,
        fps=fps,
        features=features,
        robot_type=robot_type,
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LeRobot dataset recorder")
    parser.add_argument("--output", default="./test_dataset", help="Output directory")
    args = parser.parse_args()
    
    # Create recorder
    recorder = create_dataset_recorder(
        root=args.output,
        fps=30,
        robot_type="so100",
    )
    
    # Simulate recording an episode
    recorder.start_episode(task="Pick up the block")
    
    for i in range(100):
        action = np.random.randn(6).astype(np.float32)
        state = np.random.randn(6).astype(np.float32)
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        
        recorder.add_frame(
            action=action,
            state=state,
            images={"observation.images.front": image},
        )
    
    recorder.save_episode()
    recorder.finalize()
    
    print(f"Test dataset created at: {args.output}")
