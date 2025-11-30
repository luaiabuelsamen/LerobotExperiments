#!/usr/bin/env python3
"""
Standalone ACT Training Script

Train an Action Chunking Transformer (ACT) policy on a LeRobot-format dataset.
This script extracts the essential training logic without the full lerobot framework.

Usage:
    python train_act.py --dataset ./recordings/dataset_20251130_140934 --output ./outputs/act_policy
    
    # With custom settings
    python train_act.py --dataset ./my_dataset --epochs 100 --batch_size 8 --lr 1e-5
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from torchvision import transforms
    from torchvision.io import read_video
    import cv2
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Loading
# ============================================================================

class LeRobotDataset(Dataset):
    """Load a LeRobot v3.0 format dataset for training."""
    
    def __init__(
        self,
        root: str,
        chunk_size: int = 100,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        image_transforms: Optional[callable] = None,
    ):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.image_transforms = image_transforms
        
        # Load metadata
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        
        self.fps = self.info["fps"]
        self.features = self.info["features"]
        
        # Load stats for normalization
        stats_path = self.root / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)
        else:
            self.stats = None
        
        # Load data
        self.data = self._load_data()
        
        # Setup delta timestamps for action chunking
        if delta_timestamps is None:
            # Default: predict chunk_size future actions
            self.action_delta_indices = list(range(chunk_size))
        else:
            self.action_delta_indices = [int(dt * self.fps) for dt in delta_timestamps.get("action", [])]
        
        logger.info(f"Loaded dataset with {len(self)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """Load all parquet data files."""
        data_dir = self.root / "data"
        dfs = []
        
        for parquet_file in sorted(data_dir.glob("**/*.parquet")):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        return pd.concat(dfs, ignore_index=True)
    
    def __len__(self) -> int:
        # Subtract chunk_size to ensure we can always get full action sequences
        return max(0, len(self.data) - self.chunk_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        episode_idx = row["episode_index"]
        
        # Get state
        state = np.array(row["observation.state"], dtype=np.float32)
        
        # Get action chunk (future actions)
        actions = []
        for delta in self.action_delta_indices:
            future_idx = min(idx + delta, len(self.data) - 1)
            future_row = self.data.iloc[future_idx]
            
            # Only include actions from same episode
            if future_row["episode_index"] == episode_idx:
                action = np.array(future_row["action"], dtype=np.float32)
            else:
                # Pad with last valid action
                action = np.array(row["action"], dtype=np.float32)
            actions.append(action)
        
        actions = np.stack(actions, axis=0)
        
        # Get image if available
        image = None
        if "observation.images.front" in self.features:
            image = self._load_image(idx, episode_idx)
        
        sample = {
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(actions),
        }
        
        if image is not None:
            sample["observation.images.front"] = image
        
        return sample
    
    def _load_image(self, idx: int, episode_idx: int) -> Optional[torch.Tensor]:
        """Load image from video file."""
        # For now, return None - video loading is complex
        # In production, you'd decode the video frame here
        return None
    
    def get_normalizer_stats(self) -> Dict:
        """Get statistics for normalization."""
        return self.stats


# ============================================================================
# ACT Model Components  
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ACTEncoder(nn.Module):
    """Transformer encoder for ACT."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        
        # Project state
        state_embed = self.state_proj(state).unsqueeze(1)  # (B, 1, D)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, state_embed], dim=1)  # (B, 2, D)
        
        # Transformer
        x = self.transformer(x)
        
        return x[:, 0]  # Return CLS token output


class ACTDecoder(nn.Module):
    """Transformer decoder for ACT - predicts action chunks."""
    
    def __init__(
        self,
        action_dim: int,
        chunk_size: int = 100,
        hidden_dim: int = 512,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # Learnable action queries
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_output.shape[0]
        
        # Expand encoder output as memory
        memory = encoder_output.unsqueeze(1)  # (B, 1, D)
        
        # Action queries
        queries = self.action_queries.expand(batch_size, -1, -1)  # (B, chunk_size, D)
        
        # Decode
        x = self.transformer(queries, memory)
        
        # Predict actions
        actions = self.action_head(x)  # (B, chunk_size, action_dim)
        
        return actions


class ACTPolicy(nn.Module):
    """
    Action Chunking Transformer (ACT) Policy.
    
    Simplified implementation for training on robot manipulation data.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 100,
        hidden_dim: int = 512,
        encoder_layers: int = 4,
        decoder_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_vae: bool = True,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = ACTEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # VAE components (optional)
        if use_vae:
            # VAE encoder - encodes actions to latent
            self.vae_encoder = nn.Sequential(
                nn.Linear(action_dim * chunk_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.vae_mu = nn.Linear(hidden_dim, latent_dim)
            self.vae_logvar = nn.Linear(hidden_dim, latent_dim)
            
            # Latent to decoder input
            self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Decoder
        decoder_input_dim = hidden_dim + (hidden_dim if use_vae else 0)
        self.decoder_proj = nn.Linear(decoder_input_dim, hidden_dim) if use_vae else nn.Identity()
        
        self.decoder = ACTDecoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def encode_vae(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode actions using VAE encoder."""
        batch_size = actions.shape[0]
        actions_flat = actions.view(batch_size, -1)
        
        h = self.vae_encoder(actions_flat)
        mu = self.vae_mu(h)
        logvar = self.vae_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def forward(
        self,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            state: Robot state (B, state_dim)
            actions: Ground truth actions for training (B, chunk_size, action_dim)
        
        Returns:
            predicted_actions: (B, chunk_size, action_dim)
            info: Dictionary with loss components
        """
        # Encode state
        encoder_output = self.encoder(state)
        
        info = {}
        
        if self.use_vae and actions is not None:
            # VAE: encode actions to latent
            z, mu, logvar = self.encode_vae(actions)
            latent_embed = self.latent_proj(z)
            
            # Combine encoder output with latent
            decoder_input = self.decoder_proj(
                torch.cat([encoder_output, latent_embed], dim=-1)
            )
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            info["kl_loss"] = kl_loss.mean()
        else:
            decoder_input = encoder_output
            if self.use_vae:
                # During inference, sample from prior
                z = torch.randn(state.shape[0], self.latent_dim, device=state.device)
                latent_embed = self.latent_proj(z)
                decoder_input = self.decoder_proj(
                    torch.cat([encoder_output, latent_embed], dim=-1)
                )
        
        # Decode to actions
        predicted_actions = self.decoder(decoder_input)
        
        return predicted_actions, info
    
    def compute_loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss."""
        predicted_actions, info = self.forward(state, actions)
        
        # L1 reconstruction loss
        recon_loss = F.l1_loss(predicted_actions, actions)
        info["recon_loss"] = recon_loss
        
        # Total loss
        loss = recon_loss
        if self.use_vae and "kl_loss" in info:
            loss = loss + self.kl_weight * info["kl_loss"]
        
        info["total_loss"] = loss
        return loss, info


# ============================================================================
# Normalizer
# ============================================================================

class Normalizer:
    """Normalize/denormalize data using dataset statistics."""
    
    def __init__(self, stats: Dict):
        self.stats = stats
    
    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats:
            return x
        
        mean = torch.tensor(self.stats[key]["mean"], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.stats[key]["std"], device=x.device, dtype=x.dtype)
        std = torch.clamp(std, min=1e-6)
        
        return (x - mean) / std
    
    def denormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats:
            return x
        
        mean = torch.tensor(self.stats[key]["mean"], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.stats[key]["std"], device=x.device, dtype=x.dtype)
        
        return x * std + mean


# ============================================================================
# Training Loop
# ============================================================================

def train_act(
    dataset_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    chunk_size: int = 100,
    hidden_dim: int = 512,
    use_vae: bool = True,
    kl_weight: float = 10.0,
    save_freq: int = 10,
    log_freq: int = 10,
    device: str = "cuda",
):
    """
    Train ACT policy on a dataset.
    
    Args:
        dataset_path: Path to LeRobot format dataset
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        chunk_size: Action chunk size
        hidden_dim: Transformer hidden dimension
        use_vae: Whether to use VAE
        kl_weight: KL divergence weight
        save_freq: Save checkpoint every N epochs
        log_freq: Log every N steps
        device: Device to train on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    device = torch.device(device)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = LeRobotDataset(
        root=dataset_path,
        chunk_size=chunk_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    
    # Get dimensions from data
    sample = dataset[0]
    state_dim = sample["observation.state"].shape[0]
    action_dim = sample["action"].shape[1]
    
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Create normalizer
    normalizer = Normalizer(dataset.stats) if dataset.stats else None
    
    # Create model
    model = ACTPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_dim=hidden_dim,
        use_vae=use_vae,
        kl_weight=kl_weight,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            state = batch["observation.state"].to(device)
            actions = batch["action"].to(device)
            
            # Normalize
            if normalizer:
                state = normalizer.normalize(state, "observation.state")
                actions = normalizer.normalize(actions, "action")
            
            # Forward pass
            loss, info = model.compute_loss(state, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if global_step % log_freq == 0:
                log_msg = f"Epoch {epoch+1}/{epochs} | Step {global_step} | Loss: {loss.item():.4f}"
                if "recon_loss" in info:
                    log_msg += f" | Recon: {info['recon_loss'].item():.4f}"
                if "kl_loss" in info:
                    log_msg += f" | KL: {info['kl_loss'].item():.4f}"
                logger.info(log_msg)
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "chunk_size": chunk_size,
                    "hidden_dim": hidden_dim,
                    "use_vae": use_vae,
                    "kl_weight": kl_weight,
                },
                "stats": dataset.stats,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "hidden_dim": hidden_dim,
            "use_vae": use_vae,
            "kl_weight": kl_weight,
        },
        "stats": dataset.stats,
    }, final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy on LeRobot dataset")
    
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Path to LeRobot format dataset")
    parser.add_argument("--output", "-o", type=str, default="models/act_policy",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Action chunk size")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Transformer hidden dimension")
    parser.add_argument("--no_vae", action="store_true",
                        help="Disable VAE")
    parser.add_argument("--kl_weight", type=float, default=10.0,
                        help="KL divergence weight")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Log every N steps")
    
    args = parser.parse_args()
    
    train_act(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        use_vae=not args.no_vae,
        kl_weight=args.kl_weight,
        device=args.device,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
    )


if __name__ == "__main__":
    main()
