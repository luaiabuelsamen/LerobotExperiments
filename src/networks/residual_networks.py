#!/usr/bin/env python3
"""
Residual actor and ensemble critic for  residual RL.

Based on the FAR paper implementation with:
- ViT encoder for images
- Spatial embedding for fusing patches with proprioception
- Ensemble critics (10 heads with random pair min)
- L2 action regularization
- Gradient clipping
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.init import trunc_normal_


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    use_layer_norm: bool = True,
    dropout: float = 0.0,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build MLP with LayerNorm and ReLU."""
    dims = [in_dim] + [hidden_dim] * num_layers
    
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(dims[-1], out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    
    return nn.Sequential(*layers)


class SpatialEmbedding(nn.Module):
    """
    Spatial embedding that fuses image patches with proprioceptive state.
    
    From FAR paper: Projects patch features concatenated with prop into
    a fixed-size representation via learned weights.
    """
    
    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        prop_dim: int,
        proj_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Project (num_patches + prop_dim) -> proj_dim for each patch
        proj_in_dim = num_patches + prop_dim
        num_proj = patch_dim
        
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        
        # Learned weight for weighted sum over patches
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.weight)
    
    def forward(self, feat: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Patch features [B, num_patches, patch_dim]
            prop: Proprioceptive state [B, prop_dim]
            
        Returns:
            Fused features [B, proj_dim]
        """
        # Transpose to [B, patch_dim, num_patches]
        feat = feat.transpose(1, 2)
        
        # Concatenate prop to each patch position
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat([feat, repeated_prop], dim=-1)
        
        # Project
        y = self.input_proj(feat)
        
        # Weighted sum over patches
        z = (self.weight * y).sum(dim=1)
        z = self.dropout(z)
        
        return z


class ResidualActor(nn.Module):
    """
    Residual actor for  residual RL.
    
    Takes:
    - Encoded image features (from ViT encoder)
    - Proprioceptive state
    - Base action from frozen BC policy
    
    Outputs residual action scaled by action_scale.
    """
    
    def __init__(
        self,
        repr_dim: int,
        patch_repr_dim: int,
        prop_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        action_scale: float = 0.2,
        dropout: float = 0.0,
        spatial_emb_dim: int = 1024,
        last_layer_init_scale: float = 0.001,
    ):
        """
        Args:
            repr_dim: Total representation dim from encoder (num_patches * patch_dim)
            patch_repr_dim: Dimension of each patch
            prop_dim: Proprioceptive state dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            action_scale: Scale for residual actions
            dropout: Dropout probability
            spatial_emb_dim: Dimension for spatial embedding output
            last_layer_init_scale: Scale for last layer initialization
        """
        super().__init__()
        
        self.action_scale = action_scale
        self.action_dim = action_dim
        
        # Total prop dim includes base action for residual actor
        total_prop_dim = prop_dim + action_dim
        
        # Spatial embedding for image patches + prop
        num_patches = repr_dim // patch_repr_dim
        self.compress = SpatialEmbedding(
            num_patches=num_patches,
            patch_dim=patch_repr_dim,
            prop_dim=total_prop_dim,
            proj_dim=spatial_emb_dim,
            dropout=dropout,
        )
        
        # Policy MLP
        policy_in_dim = spatial_emb_dim + total_prop_dim
        self.policy = build_mlp(
            in_dim=policy_in_dim,
            hidden_dim=hidden_dim,
            out_dim=action_dim,
            num_layers=num_layers,
            use_layer_norm=True,
            dropout=dropout,
            final_activation=nn.Tanh(),
        )
        
        # Initialize last layer with small weights
        self._init_last_layer(last_layer_init_scale)
    
    def _init_last_layer(self, scale: float):
        """Initialize last layer with small weights for stable residuals."""
        for module in reversed(list(self.policy.modules())):
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -scale, scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break
    
    def forward(
        self,
        feat: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: Encoded image features [B, num_patches, patch_dim] (not flattened)
            state: Proprioceptive state [B, prop_dim]
            base_action: Base action from BC policy [B, action_dim]
            
        Returns:
            Residual action [B, action_dim] in [-action_scale, action_scale]
        """
        # Combine state and base_action as prop
        prop = torch.cat([state, base_action], dim=-1)
        
        # Spatial embedding
        compressed = self.compress(feat, prop)
        
        # Concatenate for policy
        policy_input = torch.cat([compressed, prop], dim=-1)
        
        # Output [-1, 1] scaled by action_scale
        mu = self.policy(policy_input)
        return mu * self.action_scale


class EnsembleCritic(nn.Module):
    """
    Ensemble Q-critic for  residual RL.
    
    Uses 10 Q-heads (RED-Q style) with random pair min for targets.
    Takes encoded image features + state + action.
    """
    
    def __init__(
        self,
        repr_dim: int,
        patch_repr_dim: int,
        prop_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_q: int = 10,
        dropout: float = 0.0,
        spatial_emb_dim: int = 1024,
    ):
        """
        Args:
            repr_dim: Total representation dim from encoder
            patch_repr_dim: Dimension of each patch
            prop_dim: Proprioceptive state dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            num_q: Number of Q-networks in ensemble
            dropout: Dropout probability
            spatial_emb_dim: Dimension for spatial embedding output
        """
        super().__init__()
        
        self.num_q = num_q
        self.action_dim = action_dim
        
        # Spatial embedding
        num_patches = repr_dim // patch_repr_dim
        self.compress = SpatialEmbedding(
            num_patches=num_patches,
            patch_dim=patch_repr_dim,
            prop_dim=prop_dim,
            proj_dim=spatial_emb_dim,
            dropout=dropout,
        )
        
        # Q-network heads
        q_in_dim = spatial_emb_dim + prop_dim + action_dim
        
        self.q_networks = nn.ModuleList([
            build_mlp(
                in_dim=q_in_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                num_layers=num_layers,
                use_layer_norm=True,
                dropout=dropout,
            )
            for _ in range(num_q)
        ])
        
        # Initialize with orthogonal
        self._init_weights()
    
    def _init_weights(self):
        for q_net in self.q_networks:
            for m in q_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        feat: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: Encoded features [B, num_patches, patch_dim]
            state: State [B, prop_dim]
            action: Combined action [B, action_dim]
            
        Returns:
            Q-values from all heads [num_q, B, 1]
        """
        # Spatial embedding
        compressed = self.compress(feat, state)
        
        # Concatenate for Q-network
        q_input = torch.cat([compressed, state, action], dim=-1)
        
        # Get Q-values from all heads
        q_values = torch.stack([q(q_input) for q in self.q_networks], dim=0)
        return q_values
    
    def q_min_random_pair(
        self,
        feat: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get min Q-value from random pair of 2 heads (RED-Q style)."""
        q_all = self.forward(feat, state, action)  # [num_q, B, 1]
        
        # Random pair
        idx = torch.randperm(self.num_q, device=q_all.device)[:2]
        q_subset = q_all[idx]  # [2, B, 1]
        
        return q_subset.min(dim=0)[0]  # [B, 1]
    
    def q_mean(
        self,
        feat: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get mean Q-value over all heads (for policy gradient)."""
        q_all = self.forward(feat, state, action)
        return q_all.mean(dim=0)


# =============================================================================
# Simple MLP versions (without image encoding, for state-only)
# =============================================================================

class SimpleResidualActor(nn.Module):
    """
    Simple MLP residual actor for state-only observations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_scale: float = 0.2,
    ):
        super().__init__()
        
        self.action_scale = action_scale
        
        # Input: state + base_action
        input_dim = state_dim + action_dim
        
        self.network = build_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=action_dim,
            num_layers=num_layers,
            use_layer_norm=True,
            final_activation=nn.Tanh(),
        )
        
        # Small last layer init
        for module in reversed(list(self.network.modules())):
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -1e-3, 1e-3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break
    
    def forward(self, state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, base_action], dim=-1)
        return self.network(x) * self.action_scale


class SimpleEnsembleCritic(nn.Module):
    """
    Simple MLP ensemble critic for state-only observations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_q: int = 10,
    ):
        super().__init__()
        
        self.num_q = num_q
        
        # Input: state + base_action + residual_action
        input_dim = state_dim + 2 * action_dim
        
        self.q_networks = nn.ModuleList([
            build_mlp(
                in_dim=input_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                num_layers=num_layers,
                use_layer_norm=True,
            )
            for _ in range(num_q)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for q_net in self.q_networks:
            for m in q_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, base_action, residual_action], dim=-1)
        q_values = torch.stack([q(x) for q in self.q_networks], dim=0)
        return q_values
    
    def q_min_random_pair(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> torch.Tensor:
        q_all = self.forward(state, base_action, residual_action)
        idx = torch.randperm(self.num_q, device=q_all.device)[:2]
        return q_all[idx].min(dim=0)[0]
    
    def q_mean(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state, base_action, residual_action).mean(dim=0)


if __name__ == "__main__":
    # Test vision-based networks
    repr_dim = 81 * 128  # 81 patches, 128 dim each
    patch_repr_dim = 128
    prop_dim = 6
    action_dim = 6
    
    actor = ResidualActor(repr_dim, patch_repr_dim, prop_dim, action_dim)
    critic = EnsembleCritic(repr_dim, patch_repr_dim, prop_dim, action_dim)
    
    feat = torch.rand(4, 81, 128)  # Not flattened
    state = torch.rand(4, 6)
    base_action = torch.rand(4, 6)
    
    residual = actor(feat, state, base_action)
    print(f"Actor output: {residual.shape}, range: [{residual.min():.3f}, {residual.max():.3f}]")
    
    combined = base_action + residual
    q_vals = critic(feat, state, combined)
    print(f"Critic output: {q_vals.shape}")
    
    # Test simple networks
    simple_actor = SimpleResidualActor(prop_dim, action_dim)
    simple_critic = SimpleEnsembleCritic(prop_dim, action_dim)
    
    residual_simple = simple_actor(state, base_action)
    q_simple = simple_critic(state, base_action, residual_simple)
    print(f"Simple actor: {residual_simple.shape}, Simple critic: {q_simple.shape}")
