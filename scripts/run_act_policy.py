#!/usr/bin/env python3
"""Run a trained ACT policy checkpoint on the SO-ARM100 MuJoCo scene."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
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
    PRETRAINED_MODEL_DIR,
)


LOGGER = logging.getLogger(__name__)


class PolicyRunner:
    """Run an ACT policy on the SO-ARM100 MuJoCo simulation."""

    def __init__(
        self,
        model_path: str,
        scene_path: str = "scene.xml",
        chunk_size: int | None = None,
        n_action_steps: int | None = None,
        device: str | None = None,
        camera_name: str = "gripper_fpv",
        image_key: str = "observation.images.front",
        image_width: int = 480,
        image_height: int = 480,
    ) -> None:
        self.image_key = image_key
        self.image_width = image_width
        self.image_height = image_height
        self.pretrained_dir = self._resolve_pretrained_dir(Path(model_path))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        LOGGER.info("Using device: %s", self.device)

        self.policy = self._load_policy(chunk_size, n_action_steps)
        self.preprocessor = self._load_processor(
            POLICY_PREPROCESSOR_DEFAULT_NAME,
            overrides={"device_processor": {"device": str(self.device)}},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = self._load_processor(
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            overrides=None,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.camera_id = self._resolve_camera(camera_name)
        self.offscreen_renderer: mujoco.Renderer | None = None

        self.joint_names = [
            "Rotation",
            "Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
            "Jaw",
        ]
        self.joint_indices = [
            idx
            for name in self.joint_names
            if (idx := mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)) >= 0
        ]
        LOGGER.info("Found %d actuated joints", len(self.joint_indices))

    def _resolve_pretrained_dir(self, path: Path) -> Path:
        if path.is_file():
            raise ValueError("Expected a directory path, got a file: %s" % path)
        if (path / SAFETENSORS_SINGLE_FILE).is_file():
            return path
        candidate = path / PRETRAINED_MODEL_DIR
        if (candidate / SAFETENSORS_SINGLE_FILE).is_file():
            return candidate
        raise FileNotFoundError(
            f"Could not find '{SAFETENSORS_SINGLE_FILE}' in {path} or {candidate}."
        )

    def _load_policy(self, chunk_size: int | None, n_action_steps: int | None) -> ACTPolicy:
        config = PreTrainedConfig.from_pretrained(self.pretrained_dir, device=str(self.device))
        if not isinstance(config, ACTConfig):
            raise TypeError(f"Expected ACTConfig, got {type(config)}")
        if chunk_size is not None:
            LOGGER.info("Overriding chunk_size to %d", chunk_size)
            config.chunk_size = chunk_size
        if n_action_steps is not None:
            LOGGER.info("Overriding n_action_steps to %d", n_action_steps)
            config.n_action_steps = n_action_steps
        return ACTPolicy.from_pretrained(self.pretrained_dir, config=config)

    def _load_processor(
        self,
        name: str,
        overrides: dict[str, dict[str, object]] | None,
        *,
        to_transition,
        to_output,
    ) -> PolicyProcessorPipeline:
        cfg_name = f"{name}.json"
        return PolicyProcessorPipeline.from_pretrained(
            self.pretrained_dir,
            config_filename=cfg_name,
            overrides=overrides or {},
            to_transition=to_transition,
            to_output=to_output,
        )

    def _resolve_camera(self, camera_name: str) -> int:
        if not camera_name:
            return -1
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            LOGGER.warning("Camera '%s' not found, falling back to default view", camera_name)
            return -1
        return cam_id

    def get_state(self) -> np.ndarray:
        values = np.zeros(len(self.joint_indices), dtype=np.float32)
        for i, joint_idx in enumerate(self.joint_indices):
            qpos_idx = self.mj_model.jnt_qposadr[joint_idx]
            values[i] = self.mj_data.qpos[qpos_idx]
        return values

    def set_action(self, action: np.ndarray) -> None:
        action = np.asarray(action).flatten()
        for i, joint_idx in enumerate(self.joint_indices):
            if i >= len(action):
                break
            jnt_range = self.mj_model.jnt_range[joint_idx]
            value = float(np.clip(float(action[i]), jnt_range[0], jnt_range[1]))
            self.mj_data.ctrl[i] = value

    def _render_policy_image(self) -> torch.Tensor:
        if self.offscreen_renderer is None:
            self.offscreen_renderer = mujoco.Renderer(
                self.mj_model,
                self.image_height,
                self.image_width,
            )
        self.offscreen_renderer.update_scene(self.mj_data, camera=self.camera_id)
        frame = self.offscreen_renderer.render().copy()
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        return tensor

    def _capture_rgb_frame(
        self,
        renderer: mujoco.Renderer,
        camera_source: int | mujoco.MjvCamera | None,
    ) -> np.ndarray:
        renderer.update_scene(self.mj_data, camera=camera_source)
        frame = renderer.render().copy()
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _save_gif(frames: list[np.ndarray], path: Path, fps: int, speed: float) -> None:
        if not frames:
            return
        duration_ms = max(1, int(1000 / (fps * speed)))
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        LOGGER.info(
            "Saved GIF to %s (%d frames, %.2fx speed)",
            path,
            len(frames),
            speed,
        )

    def _build_policy_observation(self) -> dict[str, torch.Tensor]:
        obs = {
            "observation.state": torch.from_numpy(self.get_state()).float(),
            self.image_key: self._render_policy_image(),
        }
        return obs

    def _predict_action(self) -> np.ndarray:
        batch = self.preprocessor(self._build_policy_observation())
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy()
    
    def reset(self, initial_position=None) -> None:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if initial_position is not None:
            for i, joint_idx in enumerate(self.joint_indices):
                if i >= len(initial_position):
                    break
                qpos_idx = self.mj_model.jnt_qposadr[joint_idx]
                self.mj_data.qpos[qpos_idx] = initial_position[i]
            mujoco.mj_forward(self.mj_model, self.mj_data)
        self.policy.reset()
    
    def run_episode(
        self,
        max_steps: int = 1000,
        render: bool = True,
        initial_position=None,
        *,
        record_gif_path: str | None = None,
        record_fps: int = 30,
        record_speed: float = 1.0,
        record_camera_name: str | None = None,
        record_from_viewer: bool = False,
        record_width: int | None = None,
        record_height: int | None = None,
    ) -> None:
        """
        Run one episode with the policy.
        
        Args:
            max_steps: Maximum number of steps
            render: Whether to render with viewer
            initial_position: Initial joint positions to start from
        """
        # Default initial position for ACT policy
        if initial_position is None:
            initial_position = [0.3, -1.6223897, 1.6252737, 1.521133, -1.5169878, -0.01448443]
        
        self.reset(initial_position)

        record_renderer = None
        record_frames: list[np.ndarray] = []
        record_camera_id: int | None = None
        if record_gif_path is not None:
            if record_fps <= 0:
                raise ValueError("record_fps must be positive")
            if record_speed <= 0:
                raise ValueError("record_speed must be positive")
            width = record_width or self.image_width
            height = record_height or self.image_height
            record_renderer = mujoco.Renderer(self.mj_model, height, width)
            if record_camera_name:
                record_camera_id = self._resolve_camera(record_camera_name)
            else:
                record_camera_id = self.camera_id
            if record_from_viewer and not render:
                raise ValueError("record_from_viewer requires the viewer to be enabled")

        if render:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as v:
                print("Running policy... Press Ctrl+C to stop")

                step = 0
                while v.is_running() and step < max_steps:
                    action = self._predict_action()
                    self.set_action(action)
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    step += 1
                    if step % 100 == 0:
                        print(f"Step {step}/{max_steps}")
                    v.sync()
                    time.sleep(1 / 60)
                    if record_renderer is not None:
                        camera_source: int | mujoco._structs.MjvCamera | None = (
                            v.cam if record_from_viewer else record_camera_id
                        )
                        record_frames.append(
                            self._capture_rgb_frame(record_renderer, camera_source)
                        )

                print(f"Episode finished after {step} steps")
        else:
            for step in range(max_steps):
                action = self._predict_action()
                self.set_action(action)
                mujoco.mj_step(self.mj_model, self.mj_data)
                if step % 100 == 0:
                    print(f"Step {step}/{max_steps}")
                if record_renderer is not None:
                    record_frames.append(
                        self._capture_rgb_frame(record_renderer, record_camera_id)
                    )

        if record_gif_path and record_frames:
            output_path = Path(record_gif_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_gif(record_frames, output_path, record_fps, record_speed)
    
    def run_continuous(self) -> None:
        """Run policy continuously with interactive viewer."""
        # Default initial position for ACT policy
        initial_position = [0.00397189, -1.6223897, 1.6252737, 1.521133, -1.5169878, -0.01448443]
        self.reset(initial_position)
        
        def controller(model, data):
            _ = model, data
            action = self._predict_action()
            self.set_action(action)
        
        print("Running policy with interactive viewer (press Ctrl+C to exit)...")

        viewer.launch(self.mj_model, self.mj_data, controller=controller)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ACT policy in MuJoCo")
    parser.add_argument("--model", "-m", required=True, help="Checkpoint directory or pretrained_model path")
    parser.add_argument("--scene", "-s", default="env/scene.xml", help="MuJoCo scene XML path")
    parser.add_argument("--chunk_size", "-c", type=int, default=None, help="Optional chunk size override")
    parser.add_argument(
        "--action_horizon",
        "-a",
        dest="n_action_steps",
        type=int,
        default=None,
        help="Optional n_action_steps override",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Maximum simulation steps")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous viewer mode")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run the policy on")
    parser.add_argument("--camera-name", type=str, default="gripper_fpv", help="Camera name used for policy input")
    parser.add_argument("--image-key", type=str, default="observation.images.front", help="Dataset key for the camera")
    parser.add_argument("--image-width", type=int, default=480, help="Offscreen render width")
    parser.add_argument("--image-height", type=int, default=480, help="Offscreen render height")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the interactive viewer")
    parser.add_argument("--record-gif", type=str, default=None, help="Path to save a rendered GIF")
    parser.add_argument("--record-fps", type=int, default=30, help="Frame rate used for GIF rendering")
    parser.add_argument(
        "--record-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier applied to the GIF (e.g., 2.0 for 2x)",
    )
    parser.add_argument(
        "--record-camera-name",
        type=str,
        default=None,
        help="Camera name to render for the GIF (defaults to policy camera)",
    )
    parser.add_argument(
        "--record-viewer-angle",
        action="store_true",
        help="Capture frames using the interactive viewer angle each step",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=None,
        help="Optional GIF width override (defaults to policy image width)",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=None,
        help="Optional GIF height override (defaults to policy image height)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    runner = PolicyRunner(
        model_path=args.model,
        scene_path=args.scene,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        device=args.device,
        camera_name=args.camera_name,
        image_key=args.image_key,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    if args.continuous:
        runner.run_continuous()
    else:
        runner.run_episode(
            max_steps=args.steps,
            render=not args.no_viewer,
            record_gif_path=args.record_gif,
            record_fps=args.record_fps,
            record_speed=args.record_speed,
            record_camera_name=args.record_camera_name,
            record_from_viewer=args.record_viewer_angle,
            record_width=args.record_width,
            record_height=args.record_height,
        )


if __name__ == "__main__":
    main()
