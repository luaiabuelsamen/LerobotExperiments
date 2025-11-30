from __future__ import annotations

import os
import pathlib
from typing import Any, Tuple, Optional
from datetime import datetime

import numpy as np

import gymnasium as gym

# Import the standalone dataset recorder
try:
    from lerobot_dataset import LeRobotDatasetRecorder, create_dataset_recorder
    HAS_DATASET_RECORDER = True
except ImportError:
    HAS_DATASET_RECORDER = False
    print("Warning: lerobot_dataset.py not found. Dataset recording disabled.")


class SoArm100Env(gym.Env):
    """Env. Gymnasium que encapsula a descrição MJCF do SO-ARM100."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # ---------------------------------------------------------------------
    # Métodos auxiliares de construção
    # ---------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | os.PathLike | None = "scene.xml",
        frame_skip: int = 10,
        render_mode: str | None = "human",
        camera_mode: str = "third_person",  # "third_person" or "first_person"
    ) -> None:
        """Cria uma nova instância do ambiente.

        Parâmetros
        ----------
        model_path
            Caminho para o *scene.xml* que inclui o arquivo do robô
            *so_arm100.xml*. Se preferir carregar o robô diretamente, basta
            passar ``"so_arm100.xml"``.
        frame_skip
            Quantidade de sub-passos da simulação realizados a cada chamada de
            :py:meth:`step`.
        render_mode
            ``None`` para não renderizar, ``"human"`` para abrir a janela GLFW
            interativa ou ``"rgb_array"`` para obter um *frame* renderizado
            fora da tela.
        camera_mode
            ``"third_person"`` para vista externa padrão ou ``"first_person"``
            para vista em primeira pessoa do gripper.
        """

        super().__init__()

        # Importação em tempo de execução para evitar dependência dura no
        # momento de importação do módulo.
        try:
            import mujoco as _mj  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mujoco' python package is required but not installed. "
                "Please follow the installation instructions at "
                "https://github.com/google-deepmind/mujoco and ensure MuJoCo ≥3.1.6." 
            ) from exc

        # Guardamos a referência para não precisar importar novamente a cada
        # uso. Também injetamos o módulo no espaço global para que os demais
        # métodos (definidos fora do __init__) o acessem sem repetição de
        # código.
        global mujoco  # noqa: PLW0603 – we deliberately inject the symbol.
        mujoco = _mj
        self._mujoco = _mj

        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.camera_mode = camera_mode

        if render_mode == "human":
            print("Keyboard controls:")
            print("  R - Reset episode")
            print("  Z - Randomize block position")
            print("  S - Start video recording (MP4)")
            print("  X - Stop video recording")
            print("  D - Start dataset recording (LeRobot format)")
            print("  F - Finish/save dataset episode")

        # -----------------------------------------------------------------
        # Carrega o modelo MJCF e cria o objeto MjData correspondente
        # -----------------------------------------------------------------
        model_path = pathlib.Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find MJCF model: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # -----------------------------------------------------------------
        # Define espaços de ação e observação
        # -----------------------------------------------------------------
        self._setup_spaces()

        # -----------------------------------------------------------------
        # Viewer opcional (instanciado sob demanda na primeira renderização)
        # -----------------------------------------------------------------
        self._viewer: "mujoco.viewer.Viewer | None" = None

        # Data recording state
        self.recording = False
        self.recorded_data = []
        self.video_recording = False
        self.recorded_frames = []
        
        # LeRobot dataset recording
        self.dataset_recorder: Optional[LeRobotDatasetRecorder] = None
        self.dataset_recording = False
        self.current_action = None  # Store last action for dataset recording

        # Pré-calcula mapeamento *nome do atuador → índice* para conveniência.
        self._actuator_name_to_id = self._build_actuator_index()

        # Armazena uma cópia da posição *home* definida em *so_arm100.xml*.
        # Caso o keyframe seja removido do XML, caimos para vetores de zeros
        # sem quebrar a execução.
        self._home_qpos = np.zeros(self.model.nq, dtype=np.float32)
        self._home_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        self._extract_home_keyframe()

    def _keyboard_callback(self, keycode):
        """Handle keyboard input for reset and randomization."""
        if keycode == 82:  # 'R' key for reset
            print("Resetting episode...")
            self.reset()
        elif keycode == 90:  # 'Z' key for randomize
            print("Randomizing environment...")
            self._randomize_environment()
        elif keycode == 83:  # 'S' key for start recording
            if not self.recording:
                self.start_recording()
        elif keycode == 88:  # 'X' key for stop recording
            if self.recording:
                self.stop_recording()
        elif keycode == 68:  # 'D' key for dataset recording
            if not self.dataset_recording:
                self.start_dataset_recording()
        elif keycode == 70:  # 'F' key for finish dataset episode
            if self.dataset_recording:
                self.save_dataset_episode()

    def _randomize_environment(self):
        """Randomize block position within bounds."""
        # Randomize block position
        block_x = np.random.uniform(-0.15, 0.15)  # front/back
        block_y = np.random.uniform(-0.4, -0.3)   # left/right
        block_z = 0.15  # fixed height on table

        # Set block position (free joint body)
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        if block_id >= 0:
            # For free joint bodies, qpos has 7 values: 3 position + 4 quaternion
            qpos_start = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block")]
            self.data.qpos[qpos_start:qpos_start+3] = [block_x, block_y, block_z]
            self.data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # quaternion identity
            
            # Reset velocities
            qvel_start = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block")]
            self.data.qvel[qvel_start:qvel_start+6] = 0.0

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------------------
    # LeRobot Dataset Recording Methods
    # -------------------------------------------------------------------------
    
    def init_dataset_recorder(
        self,
        dataset_path: str = None,
        task: str = "Pick up the block",
        fps: int = 30,
        robot_type: str = "so100_mujoco",
    ):
        """Initialize the LeRobot dataset recorder.
        
        Args:
            dataset_path: Path to save the dataset. If None, auto-generates a name.
            task: Task description for this dataset.
            fps: Recording framerate.
            robot_type: Robot type identifier.
        """
        if not HAS_DATASET_RECORDER:
            print("Error: lerobot_dataset module not available")
            return False
        
        if dataset_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_path = f"./recordings/dataset_{timestamp}"
        
        try:
            # Check if dataset exists
            if os.path.exists(dataset_path):
                print(f"Loading existing dataset: {dataset_path}")
                self.dataset_recorder = LeRobotDatasetRecorder.load(dataset_path)
            else:
                print(f"Creating new dataset: {dataset_path}")
                self.dataset_recorder = create_dataset_recorder(
                    root=dataset_path,
                    fps=fps,
                    robot_type=robot_type,
                    action_dim=self.model.nu,  # Number of actuators
                    state_dim=self.model.nu,
                    image_height=480,
                    image_width=480,
                )
            
            self.dataset_task = task
            print(f"Dataset recorder initialized: {dataset_path}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize dataset recorder: {e}")
            self.dataset_recorder = None
            return False
    
    def start_dataset_recording(self, task: str = None):
        """Start recording an episode for the LeRobot dataset."""
        if self.dataset_recorder is None:
            # Auto-initialize with defaults
            if not self.init_dataset_recorder():
                return
        
        task = task or getattr(self, 'dataset_task', "Pick up the block")
        self.dataset_recorder.start_episode(task=task)
        self.dataset_recording = True
        print(f"Dataset episode recording started: {task}")
    
    def save_dataset_episode(self):
        """Save the current episode to the dataset."""
        if not self.dataset_recording or self.dataset_recorder is None:
            print("No dataset recording in progress")
            return
        
        episode_idx = self.dataset_recorder.save_episode()
        self.dataset_recording = False
        print(f"Dataset episode {episode_idx} saved")
    
    def finalize_dataset(self):
        """Finalize and close the dataset recorder."""
        if self.dataset_recorder is not None:
            if self.dataset_recording:
                self.save_dataset_episode()
            self.dataset_recorder.finalize()
            print("Dataset finalized")
    
    def _record_dataset_frame(self, action: np.ndarray):
        """Record a frame to the dataset (called internally during step)."""
        if not self.dataset_recording or self.dataset_recorder is None:
            return
        
        # Get current state (joint positions)
        state = self.data.qpos[:self.model.nu].copy().astype(np.float32)
        
        # Get first person camera image
        image = self.render(mode="rgb_array")
        
        if image is not None:
            self.dataset_recorder.add_frame(
                action=action.astype(np.float32),
                state=state,
                images={"observation.images.front": image},
            )

    # -------------------------------------------------------------------------
    # Original Recording Methods (MP4 only)
    # -------------------------------------------------------------------------

    def start_recording(self):
        """Start recording episode data and video."""
        self.recording = True
        self.recorded_data = []
        self.video_recording = True
        self.recorded_frames = []
        print("Recording started - capturing data and video frames")

    def stop_recording(self):
        """Stop recording episode data and save video."""
        self.recording = False
        self.video_recording = False
        
        # Save video if frames were recorded
        if self.recorded_frames:
            self._save_video()
        
        print(f"Recording stopped. Recorded {len(self.recorded_data)} data points and {len(self.recorded_frames)} video frames")

    def _save_video(self, filename: str = "demo_recording.mp4", fps: int = 30):
        """Save recorded frames as a video file."""
        if not self.recorded_frames:
            print("No frames to save")
            return
            
        try:
            import cv2
            import numpy as np
            
            # Get frame dimensions
            height, width = self.recorded_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            # Write frames
            for frame in self.recorded_frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            video_writer.release()
            print(f"Video saved as {filename}")
            
        except ImportError:
            print("OpenCV not available for video saving. Install with: pip install opencv-python")
        except Exception as e:
            print(f"Error saving video: {e}")

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinicia qpos/qvel e também os comandos dos atuadores (ctrl).
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self._home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._home_ctrl

        # Garante que a dinâmica interna esteja consistente com o novo estado.
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()

        info: dict[str, Any] = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: np.ndarray):
        # Limita a ação ao intervalo permitido pelo atuador.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        # Store current action for dataset recording
        self.current_action = action.copy()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # Compute reward based on task (pick and place)
        reward = self._compute_reward()
        
        # Check if task is complete
        terminated = self._check_success()
        truncated = False
        info: dict[str, Any] = {
            'block_pos': self._get_block_pos(),
            'box_pos': self._get_box_pos(),
            'gripper_pos': self._get_gripper_pos(),
            'distance_to_block': self._get_distance_to_block(),
            'block_in_box': self._is_block_in_box(),
            'success': terminated
        }

        # Record data if recording is enabled
        if self.recording:
            self.recorded_data.append({
                'observation': observation.copy(),
                'action': action.copy(),
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info.copy()
            })

        # Record video frame if video recording is enabled
        if self.video_recording and self.render_mode == "human":
            try:
                # Temporarily switch to first-person camera for video recording
                original_camera_mode = self.camera_mode
                self.camera_mode = "first_person"
                frame = self.render(mode="rgb_array")
                self.camera_mode = original_camera_mode  # Restore original
                
                if frame is not None:
                    self.recorded_frames.append(frame.copy())
            except Exception as e:
                print(f"Error capturing video frame: {e}")
        
        # Record to LeRobot dataset if dataset recording is enabled
        if self.dataset_recording:
            self._record_dataset_frame(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    
    def _get_block_pos(self):
        """Get the position of the block."""
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        if block_id >= 0:
            return self.data.xpos[block_id].copy()
        return np.zeros(3)
    
    def _get_box_pos(self):
        """Get the position of the box center."""
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "box_center")
        if box_id >= 0:
            return self.data.site_xpos[box_id].copy()
        return np.array([-0.1, -0.35, 0.125])
    
    def _get_gripper_pos(self):
        """Get the position of the gripper (Fixed_Jaw body)."""
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        if gripper_id >= 0:
            return self.data.xpos[gripper_id].copy()
        return np.zeros(3)
    
    def _get_distance_to_block(self):
        """Calculate distance from gripper to block."""
        gripper_pos = self._get_gripper_pos()
        block_pos = self._get_block_pos()
        return np.linalg.norm(gripper_pos - block_pos)
    
    def _is_block_in_box(self):
        """Check if block is inside the box."""
        block_pos = self._get_block_pos()
        box_pos = self._get_box_pos()
        
        # Check if block is within box horizontal bounds (6cm x 6cm box)
        dx = abs(block_pos[0] - box_pos[0])
        dy = abs(block_pos[1] - box_pos[1])
        
        # Check if block is at appropriate height (on bottom of box)
        dz = abs(block_pos[2] - 0.115)  # Box bottom + block half-height
        
        return dx < 0.05 and dy < 0.05 and dz < 0.02
    
    def _compute_reward(self):
        """Compute reward for pick and place task."""
        reward = 0.0
        
        block_pos = self._get_block_pos()
        box_pos = self._get_box_pos()
        gripper_pos = self._get_gripper_pos()
        
        # Distance from gripper to block
        dist_to_block = np.linalg.norm(gripper_pos - block_pos)
        
        # Distance from block to box
        dist_block_to_box = np.linalg.norm(block_pos - box_pos)
        
        # Reward for getting gripper close to block
        reward += -dist_to_block * 5.0
        
        # Reward for getting block close to box
        reward += -dist_block_to_box * 10.0
        
        # Big reward for block in box
        if self._is_block_in_box():
            reward += 100.0
        
        # Small penalty for time (encourages faster completion)
        reward -= 0.1
        
        return reward
    
    def _check_success(self):
        """Check if the task is successfully completed."""
        # Disabled automatic termination - controlled by keyboard now
        return False

    def render(self, mode: str | None = None):  # type: ignore[override]
        mode = mode or self.render_mode
        if mode is None:
            return  # Sem renderização.

        if mode == "human":
            # Instancia o viewer (janela GLFW) apenas na primeira chamada.
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                # O *viewer passivo* não assume o contexto via ``with``; ele
                # permanece vivo até que ``.close()`` seja chamado.
                self._viewer = mj_viewer.launch_passive(
                    self.model, 
                    self.data,
                    key_callback=self._keyboard_callback
                )

            # Mantém a janela sincronizada com a simulação.
            self._viewer.sync()
        elif mode == "rgb_array":
            # Renderização fora da tela – aloca na primeira chamada e reutiliza.
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, 480, 480)

            # Set camera based on mode
            camera = -1  # Default camera
            if self.camera_mode == "first_person":
                camera = self._setup_first_person_camera()

            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render().copy()
        else:
            raise NotImplementedError(f"Unsupported render mode '{mode}'")

    def close(self):  # type: ignore[override]
        # Finalize dataset if recording
        if hasattr(self, 'dataset_recorder') and self.dataset_recorder is not None:
            self.finalize_dataset()
        
        # Fecha a janela do viewer caso ela tenha sido criada.
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Métodos auxiliares públicos
    # ------------------------------------------------------------------

    def _setup_first_person_camera(self):
        """Setup first person camera using the predefined camera in XML."""
        # Look for the camera in XML
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "first_person")
        if camera_id >= 0:
            return camera_id
        
        # If not found, try gripper_fpv
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_fpv")
        if camera_id >= 0:
            return camera_id
        
        print("WARNING: No first-person camera found in XML, using default")
        return -1  # Use default camera

    def save_image(self, filename: str, camera_mode: str | None = None):
        """Save a rendered image to a PNG file.
        
        Parameters
        ----------
        filename
            Path to save the PNG file (should end with .png)
        camera_mode
            Override camera mode for this image ("third_person" or "first_person")
        """
        # Temporarily change camera mode if specified
        original_camera_mode = self.camera_mode
        if camera_mode is not None:
            self.camera_mode = camera_mode
            
        try:
            # Render the image
            image = self.render(mode="rgb_array")
            if image is not None:
                # Save using PIL or matplotlib
                try:
                    from PIL import Image
                    img = Image.fromarray(image)
                    img.save(filename)
                    print(f"Image saved to {filename}")
                except ImportError:
                    # Fallback to matplotlib
                    import matplotlib.pyplot as plt
                    plt.imsave(filename, image)
                    print(f"Image saved to {filename} (using matplotlib)")
            else:
                print("Failed to render image")
        finally:
            # Restore original camera mode
            self.camera_mode = original_camera_mode

    def _setup_spaces(self):
        """Cria :pyattr:`action_space` e :pyattr:`observation_space`."""

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        obs_high = np.inf * np.ones(self.model.nq + self.model.nv, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Retorna o vetor de observação atual (*qpos*‖*qvel*)."""

        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    # ------------------------------------------------------------------
    # Métodos auxiliares internos
    # ------------------------------------------------------------------

    def _build_actuator_index(self):
        """Gera um dicionário *nome do atuador → id do atuador*."""

        name_to_id: dict[str, int] = {}
        for i in range(self.model.nu):
            name_val = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name_val is None:
                continue

            # MuJoCo < 3.1.6 devolvia ``bytes`` enquanto ≥ 3.1.7 devolve ``str``.
            if isinstance(name_val, bytes):
                name_val = name_val.decode()

            name_to_id[str(name_val)] = i
        return name_to_id

    def _extract_home_keyframe(self):
        """Extrai *qpos*/*ctrl* do keyframe chamado "home" dentro do XML."""

        for key_id in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)

            if key_name is None:
                continue

            if isinstance(key_name, bytes):
                key_name = key_name.decode()

            if key_name == "home":
                # O keyframe armazena qpos (nq), qvel (nv) e ctrl (nu).
                self._home_qpos = self.model.key_qpos[key_id].copy()
                if self.model.nu:
                    self._home_ctrl = self.model.key_ctrl[key_id].copy()
                return

        # Caso não exista keyframe "home": vetores já iniciados em zero.


def main():
    """Simple main function to demonstrate the lerobot in Mujoco environment."""
    import time
    
    # Try human rendering first, fall back to rgb_array if no display
    try:
        env = SoArm100Env(render_mode="human")
        print("Lerobot Mujoco Environment Demo")
        print("Close the viewer window to exit")
        
        # Use passive viewer like manual_control.py
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch(env.model, env.data)
        try:
            while v.is_running():
                # Advance simulation without control inputs (like manual_control.py)
                for _ in range(env.frame_skip):
                    mujoco.mj_step(env.model, env.data)
                # Update the window
                v.sync()
                # Small delay to control simulation speed
                time.sleep(0.01)
        finally:
            v.close()
                
    except Exception as e:
        if "GLFW" in str(e) or "DISPLAY" in str(e):
            print("No display available, running in headless mode...")
            env = SoArm100Env(render_mode="rgb_array")
            
            # Reset to initial state
            obs, info = env.reset()
            
            try:
                for _ in range(100):  # Run for 100 steps in headless mode
                    # Take a zero action
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    time.sleep(0.01)
                    if terminated or truncated:
                        break
                print("Headless simulation completed")
            finally:
                env.close()
        else:
            raise


if __name__ == "__main__":
    main()
