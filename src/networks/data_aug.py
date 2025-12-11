#!/usr/bin/env python3
"""
Data augmentation for residual RL training.

Based on the FAR paper's RandomShiftsAug implementation.
"""
from __future__ import annotations

import torch
from torch import nn


class RandomShiftsAug:
    """
    Random crop augmentation via grid sampling.
    
    Pads the image and samples a random crop, effectively shifting the image.
    This is critical for vision-based RL to prevent overfitting.
    """
    
    def __init__(self, pad: int = 4):
        """
        Args:
            pad: Number of pixels to pad on each side
        """
        self.pad = pad
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random shift augmentation.
        
        Args:
            x: Image tensor [B, C, H, W]
            
        Returns:
            Augmented image tensor [B, C, H, W]
        """
        n, c, h, w = x.size()
        assert h == w, "Expected square images"
        
        # Pad with replicate
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        
        # Create base grid
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad,
            device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        
        # Random shift
        shift = torch.randint(
            0, 2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        
        grid = base_grid + shift
        return nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )


class NoAug:
    """No augmentation (identity)."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == "__main__":
    # Test augmentation
    aug = RandomShiftsAug(pad=4)
    x = torch.rand(4, 3, 84, 84)
    y = aug(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Changed: {(x != y).any().item()}")
