#!/usr/bin/env python3
"""Train ACT with the official LeRobot pipeline.

This helper mirrors :mod:`lerobot.scripts.lerobot_train` but exposes a lightweight
CLI suitable for local datasets captured with the recording tools that ship in
this repository. Under the hood we instantiate the exact same configs, datasets
and policy classes that the upstream project uses, ensuring parity with the
reference implementation while keeping ergonomics familiar (``python src/train_act.py --dataset ...``).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List

# Ensure the in-tree LeRobot package is importable without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
LERO_SRC = REPO_ROOT / "lerobot" / "src"
if str(LERO_SRC) not in sys.path:
    sys.path.insert(0, str(LERO_SRC))

from lerobot.configs.default import DatasetConfig, WandBConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.transforms import ImageTransformsConfig  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 100
DEFAULT_N_OBS_STEPS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT using LeRobot components.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a LeRobot dataset root (must contain meta/info.json)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Identifier passed to LeRobot factories. Defaults to the dataset folder name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to LeRobot's timestamped outputs/train structure.",
    )
    parser.add_argument("--overwrite-output", action="store_true", help="Delete an existing output directory.")
    parser.add_argument("--steps", type=int, default=100_000, help="Number of optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override ACT chunk size (defaults to the upstream value of 100).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="How many predicted actions to execute per environment step (<= chunk size).",
    )
    parser.add_argument("--n-obs-steps", type=int, default=DEFAULT_N_OBS_STEPS, help="Observation history length.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string, e.g. cuda or cpu.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed passed to the trainer.")
    parser.add_argument("--streaming", action="store_true", help="Use the streaming dataset loader.")
    parser.add_argument(
        "--video-backend",
        type=str,
        choices=("pyav", "video_reader", "torchcodec"),
        default=None,
        help="Force a specific video decoding backend (defaults to LeRobot's auto-detected choice).",
    )
    parser.add_argument("--log-freq", type=int, default=200, help="Logging frequency (in steps).")
    parser.add_argument("--save-freq", type=int, default=20_000, help="Checkpoint frequency (in steps).")
    parser.add_argument("--eval-freq", type=int, default=0, help="Evaluation frequency. Set 0 to disable.")
    parser.add_argument("--no-save-checkpoint", action="store_true", help="Skip saving checkpoints during training.")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional subset of episode indices. Accepts comma separated values and closed ranges (e.g. 0:10,15).",
    )
    parser.add_argument("--no-imagenet-stats", action="store_true", help="Disable ImageNet normalization presets.")
    parser.add_argument("--image-transforms", action="store_true", help="Enable default image augmentation pipeline.")
    parser.add_argument(
        "--image-transforms-max",
        type=int,
        default=None,
        help="Maximum number of augmentations to sample per frame when transforms are enabled.",
    )
    parser.add_argument(
        "--image-transforms-random-order",
        action="store_true",
        help="Shuffle image transforms order for each frame.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="lerobot", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team) name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
        help="W&B mode. Defaults to upstream behavior.",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Optional job name overriding the default policy name.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload checkpoints to the Hugging Face Hub.")
    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default=None,
        help="Repository ID used when pushing checkpoints. Required when --push-to-hub is set.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override ACT optimizer learning rate.")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Override ACT vision backbone learning rate (defaults to --lr when provided).",
    )
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay.")
    return parser.parse_args()


def parse_episode_selection(spec: str | None) -> List[int] | None:
    if spec is None:
        return None
    selections: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            start_str, end_str = token.split(":", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{token}' (end < start)")
            selections.extend(range(start, end + 1))
        else:
            selections.append(int(token))
    return selections if selections else None


def build_policy_config(meta: LeRobotDatasetMetadata, args: argparse.Namespace) -> ACTConfig:
    features = dataset_to_policy_features(meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not output_features:
        raise ValueError("Dataset does not expose any ACTION features")
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    chunk_size = args.chunk_size or DEFAULT_CHUNK_SIZE
    n_action_steps = args.n_action_steps or chunk_size
    if n_action_steps > chunk_size:
        raise ValueError("n_action_steps must be <= chunk_size")

    policy_kwargs = dict(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        n_obs_steps=args.n_obs_steps,
        device=args.device,
        use_amp=args.use_amp,
        push_to_hub=args.push_to_hub,
        repo_id=args.policy_repo_id,
    )
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--policy-repo-id must be provided when --push-to-hub is enabled")
    if args.lr is not None:
        policy_kwargs["optimizer_lr"] = args.lr
        policy_kwargs.setdefault("optimizer_lr_backbone", args.backbone_lr or args.lr)
    if args.backbone_lr is not None:
        policy_kwargs["optimizer_lr_backbone"] = args.backbone_lr
    if args.weight_decay is not None:
        policy_kwargs["optimizer_weight_decay"] = args.weight_decay

    policy_cfg = ACTConfig(**policy_kwargs)
    policy_cfg.validate_features()
    return policy_cfg


def build_dataset_config(meta_repo_id: str, dataset_root: Path, args: argparse.Namespace) -> DatasetConfig:
    img_tf_cfg = ImageTransformsConfig()
    if args.image_transforms:
        img_tf_cfg.enable = True
    if args.image_transforms_max is not None:
        img_tf_cfg.max_num_transforms = args.image_transforms_max
    if args.image_transforms_random_order:
        img_tf_cfg.random_order = True

    dataset_kwargs = dict(
        repo_id=meta_repo_id,
        root=str(dataset_root),
        episodes=parse_episode_selection(args.episodes),
        image_transforms=img_tf_cfg,
        streaming=args.streaming,
        use_imagenet_stats=not args.no_imagenet_stats,
    )

    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend

    return DatasetConfig(**dataset_kwargs)


def build_wandb_config(args: argparse.Namespace) -> WandBConfig:
    return WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )


def maybe_prepare_output(path: Path | None, allow_overwrite: bool) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.exists():
        if not allow_overwrite:
            raise FileExistsError(
                f"Output directory {resolved} already exists. Pass --overwrite-output to remove it first."
            )
        shutil.rmtree(resolved)
    return resolved


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    meta_path = dataset_root / "meta" / "info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} is missing. Please provide a valid LeRobot dataset root.")

    dataset_repo_id = args.dataset_repo_id or dataset_root.name
    LOGGER.info("Loading dataset metadata from %s (repo_id=%s)", dataset_root, dataset_repo_id)
    meta = LeRobotDatasetMetadata(dataset_repo_id, root=str(dataset_root), force_cache_sync=False)

    policy_cfg = build_policy_config(meta, args)
    dataset_cfg = build_dataset_config(dataset_repo_id, dataset_root, args)
    wandb_cfg = build_wandb_config(args)
    output_dir = maybe_prepare_output(args.output, args.overwrite_output)

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=args.job_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_checkpoint=not args.no_save_checkpoint,
        save_freq=args.save_freq,
        seed=args.seed,
        wandb=wandb_cfg,
    )

    init_logging()
    lerobot_train.train(train_cfg)


if __name__ == "__main__":
    main()
