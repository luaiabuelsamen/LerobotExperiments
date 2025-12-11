#!/usr/bin/env python3
"""
Vision Transformer (ViT) encoder for residual RL.

Based on the FAR paper's MinViT implementation.
Processes images into patch-based features for the critic/actor.
"""
from __future__ import annotations

import einops
import torch
from torch import nn
from torch.nn.init import trunc_normal_


class PatchEmbed(nn.Module):
    """
    Patch embedding with convolutional layers.
    
    For 84x84 or 96x96 images, produces 81 or 121 patches respectively.
    """
    
    def __init__(self, embed_dim: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(3, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)
        
        # For 84x84 input: 81 patches
        # For 96x96 input: 121 patches
        # For 480x480 input: need to calculate
        self.patch_dim = embed_dim
    
    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size."""
        # After first conv: (img_size - 8) / 4 + 1
        # After second conv: floor((first - 3) / 2 + 1)
        h1 = (img_size - 8) // 4 + 1
        h2 = (h1 - 3) // 2 + 1
        return h2 * h2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h w) c")
        return y


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(
            qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_heads
        ).unbind(1)
        
        # Scaled dot-product attention
        attn_v = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, attn_mask=attn_mask
        )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    """Single transformer layer with pre-norm."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(nn.functional.gelu(self.linear1(x)))


class MinViT(nn.Module):
    """
    Minimal Vision Transformer for RL.
    
    Takes images and produces patch-based features.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        embed_norm: bool = True,
        num_heads: int = 4,
        depth: int = 1,
        img_size: int = 84,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(embed_dim, use_norm=embed_norm)
        self.num_patches = self.patch_embed.get_num_patches(img_size)
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        self.layers = nn.Sequential(*[
            TransformerLayer(embed_dim, num_heads, dropout=0.0)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [B, C, H, W]
            
        Returns:
            Features [B, num_patches, embed_dim]
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.layers(x)
        return self.norm(x)


class VitEncoder(nn.Module):
    """
    ViT encoder wrapper for RL.
    
    Handles image normalization and produces flattened features.
    """
    
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        embed_dim: int = 128,
        embed_norm: bool = True,
        num_heads: int = 4,
        depth: int = 1,
    ):
        """
        Args:
            obs_shape: (C, H, W) shape of input images
            embed_dim: Embedding dimension for patches
            embed_norm: Whether to use GroupNorm in patch embedding
            num_heads: Number of attention heads
            depth: Number of transformer layers
        """
        super().__init__()
        
        c, h, w = obs_shape
        assert h == w, f"Expected square images, got {h}x{w}"
        
        self.vit = MinViT(
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            num_heads=num_heads,
            depth=depth,
            img_size=h,
        )
        
        self.num_patches = self.vit.num_patches
        self.patch_repr_dim = embed_dim
        self.repr_dim = embed_dim * self.num_patches
    
    def forward(self, obs: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """
        Args:
            obs: Images [B, C, H, W] - can be uint8 [0, 255] or float [0, 1]
            flatten: If True, return [B, D*N], else [B, N, D]
            
        Returns:
            Features tensor
        """
        # Normalize images
        if obs.max() > 5:
            obs = obs.float() / 255.0
        obs = obs - 0.5  # Center around 0
        
        feats = self.vit(obs)  # [B, N, D]
        
        if flatten:
            feats = feats.flatten(1, 2)  # [B, N*D]
        
        return feats


if __name__ == "__main__":
    # Test with different image sizes
    for img_size in [84, 96, 480]:
        enc = VitEncoder(obs_shape=(3, img_size, img_size))
        x = torch.rand(2, 3, img_size, img_size)
        y = enc(x)
        print(f"Input: {x.shape} -> Output: {y.shape} (num_patches={enc.num_patches})")
