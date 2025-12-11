#!/usr/bin/env python3
"""
Q-Agent for  residual RL.

Combines:
- ViT encoder for images
- Ensemble critic
- Residual actor
- Data augmentation
- Gradient clipping
- LR warmup
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from src.networks.data_aug import RandomShiftsAug
from src.networks.residual_networks import (
    EnsembleCritic,
    ResidualActor,
    SimpleEnsembleCritic,
    SimpleResidualActor,
)
from src.networks.vit_encoder import VitEncoder


class QAgent(nn.Module):
    """
    Q-Agent combining encoder, actor, and critic.
    
    Based on FAR paper implementation with all the bells and whistles.
    """
    
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_dim: int,
        action_dim: int,
        # Encoder config
        embed_dim: int = 128,
        encoder_depth: int = 1,
        encoder_heads: int = 4,
        # Network config
        hidden_dim: int = 1024,
        num_layers: int = 2,
        spatial_emb_dim: int = 1024,
        # Critic config
        num_q: int = 10,
        # Actor config
        action_scale: float = 0.2,
        action_l2_reg: float = 0.001,
        actor_last_layer_init: float = 0.001,
        # Training config
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        tau: float = 0.01,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
        lr_warmup_steps: int = 1000,
        lr_warmup_start: float = 1e-8,
        # Data augmentation
        aug_pad: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.tau = tau
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.action_scale = action_scale
        self.action_l2_reg = action_l2_reg
        self.lr_warmup_steps = lr_warmup_steps
        
        # Build encoder
        self.encoder = VitEncoder(
            obs_shape=obs_shape,
            embed_dim=embed_dim,
            embed_norm=True,
            num_heads=encoder_heads,
            depth=encoder_depth,
        ).to(self.device)
        
        repr_dim = self.encoder.repr_dim
        patch_repr_dim = self.encoder.patch_repr_dim
        
        # Build actor
        self.actor = ResidualActor(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=prop_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            action_scale=action_scale,
            spatial_emb_dim=spatial_emb_dim,
            last_layer_init_scale=actor_last_layer_init,
        ).to(self.device)
        
        # Build critic
        self.critic = EnsembleCritic(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=prop_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_q=num_q,
            spatial_emb_dim=spatial_emb_dim,
        ).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Freeze targets
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=critic_lr
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr
        )
        
        # LR warmup schedulers
        if lr_warmup_steps > 0:
            start_factor = lr_warmup_start / critic_lr
            start_factor = max(start_factor, 1e-8)
            
            self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.encoder_opt, start_factor=start_factor, total_iters=lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=start_factor, total_iters=lr_warmup_steps
            )
            
            actor_start_factor = lr_warmup_start / actor_lr
            actor_start_factor = max(actor_start_factor, 1e-8)
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=lr_warmup_steps
            )
        else:
            self.encoder_scheduler = None
            self.critic_scheduler = None
            self.actor_scheduler = None
        
        # Data augmentation
        self.aug = RandomShiftsAug(pad=aug_pad)
        
        self.total_it = 0
    
    def _encode(self, obs: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Encode images with optional augmentation."""
        # Normalize if needed
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        
        if augment:
            obs = self.aug(obs)
        
        # Encode (not flattened, [B, N, D])
        return self.encoder(obs, flatten=False)
    
    def select_action(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
        stddev: float = 0.0,
    ) -> torch.Tensor:
        """
        Select action for environment interaction.
        
        Args:
            image: Image observation [B, C, H, W]
            state: Proprioceptive state [B, prop_dim]
            base_action: Base action from BC policy [B, action_dim]
            stddev: Exploration noise std
            
        Returns:
            Residual action [B, action_dim]
        """
        with torch.no_grad():
            feat = self._encode(image, augment=False)
            residual = self.actor(feat, state, base_action)
            
            if stddev > 0:
                noise = torch.randn_like(residual) * stddev
                residual = (residual + noise).clamp(
                    -self.action_scale, self.action_scale
                )
            
            return residual
    
    def update_critic(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_image: torch.Tensor,
        next_state: torch.Tensor,
        next_base_action: torch.Tensor,
        done: torch.Tensor,
        stddev: float = 0.1,
        n_step: int = 1,
    ) -> dict:
        """Update critic with TD3-style target."""
        metrics = {}
        
        with torch.no_grad():
            # Encode next observation
            next_feat = self._encode(next_image, augment=False)
            
            # Get target residual action with noise
            next_residual = self.actor_target(next_feat, next_state, next_base_action)
            noise = (torch.randn_like(next_residual) * stddev).clamp(-0.3, 0.3)
            next_residual = (next_residual + noise).clamp(
                -self.action_scale, self.action_scale
            )
            
            # Combined action (clamped to [-1, 1])
            next_combined = torch.clamp(next_base_action + next_residual, -1.0, 1.0)
            
            # Target Q (min over random pair)
            target_q = self.critic_target.q_min_random_pair(
                next_feat, next_state, next_combined
            )
            target_q = reward + (1 - done) * (self.gamma ** n_step) * target_q
        
        # Encode current observation with augmentation
        feat = self._encode(image, augment=True)
        
        # Current Q values
        current_q = self.critic(feat, state, action)  # [num_q, B, 1]
        
        # MSE loss for all heads
        critic_loss = ((current_q - target_q.unsqueeze(0)) ** 2).mean()
        
        # Optimize
        self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping
        encoder_grad_norm = nn.utils.clip_grad_norm_(
            self.encoder.parameters(), self.grad_clip
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.grad_clip
        )
        
        self.encoder_opt.step()
        self.critic_opt.step()
        
        if self.encoder_scheduler:
            self.encoder_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()
        
        metrics["critic_loss"] = critic_loss.item()
        metrics["target_q"] = target_q.mean().item()
        metrics["encoder_grad_norm"] = encoder_grad_norm.item()
        metrics["critic_grad_norm"] = critic_grad_norm.item()
        
        return metrics
    
    def update_actor(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        base_action: torch.Tensor,
    ) -> dict:
        """Update actor with policy gradient + L2 regularization."""
        metrics = {}
        
        # Encode with augmentation (but detach for actor update)
        feat = self._encode(image, augment=True).detach()
        
        # Get residual action
        residual = self.actor(feat, state, base_action)
        
        # L2 regularization on residual magnitude
        l2_penalty = self.action_l2_reg * (residual ** 2).sum(dim=-1).mean()
        
        # Combined action
        combined = torch.clamp(base_action + residual, -1.0, 1.0)
        
        # Q-value (mean over all heads)
        q_value = self.critic.q_mean(feat, state, combined)
        
        actor_loss = -q_value.mean() + l2_penalty
        
        # Optimize
        self.actor_opt.zero_grad()
        actor_loss.backward()
        
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.grad_clip
        )
        
        self.actor_opt.step()
        
        if self.actor_scheduler:
            self.actor_scheduler.step()
        
        # Soft update targets
        self._soft_update()
        
        metrics["actor_loss"] = actor_loss.item()
        metrics["actor_l2"] = l2_penalty.item()
        metrics["actor_grad_norm"] = actor_grad_norm.item()
        metrics["residual_magnitude"] = residual.abs().mean().item()
        
        return metrics
    
    def _soft_update(self):
        """Soft update target networks."""
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
    
    def save(self, path: str, step: int):
        """Save checkpoint."""
        torch.save({
            "step": step,
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "action_scale": self.action_scale,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.encoder_opt.load_state_dict(checkpoint["encoder_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
        return checkpoint.get("step", 0)


class SimpleQAgent:
    """
    Simple Q-Agent for state-only observations (no images).
    
    Uses simpler MLP networks without ViT encoder.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_q: int = 10,
        action_scale: float = 0.2,
        action_l2_reg: float = 0.001,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        tau: float = 0.01,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
        lr_warmup_steps: int = 1000,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.tau = tau
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.action_scale = action_scale
        self.action_l2_reg = action_l2_reg
        
        # Networks
        self.actor = SimpleResidualActor(
            state_dim, action_dim, hidden_dim, num_layers, action_scale
        ).to(self.device)
        
        self.critic = SimpleEnsembleCritic(
            state_dim, action_dim, hidden_dim, num_layers, num_q
        ).to(self.device)
        
        # Targets
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)
        
        # LR warmup
        if lr_warmup_steps > 0:
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=1e-8, total_iters=lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=1e-8, total_iters=lr_warmup_steps
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
        
        self.total_it = 0
    
    def select_action(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        std: float = 0.0,
    ) -> torch.Tensor:
        """Select residual action."""
        with torch.no_grad():
            residual = self.actor(state, base_action)
            if std > 0:
                noise = torch.randn_like(residual) * std
                residual = (residual + noise).clamp(-self.action_scale, self.action_scale)
            return residual
    
    def train(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_base_action: torch.Tensor,
        done: torch.Tensor,
        update_actor: bool = True,
        stddev: float = 0.1,
        n_step: int = 1,
    ) -> dict:
        """Train one step."""
        self.total_it += 1
        metrics = {}
        
        # Scale reward
        reward = reward / 100.0
        
        # =====================================================================
        # CRITIC UPDATE
        # =====================================================================
        with torch.no_grad():
            # Target residual with noise
            next_residual = self.actor_target(next_state, next_base_action)
            noise = (torch.randn_like(next_residual) * stddev).clamp(-0.3, 0.3)
            next_residual = (next_residual + noise).clamp(-self.action_scale, self.action_scale)
            
            # Target Q
            target_q = self.critic_target.q_min_random_pair(
                next_state, next_base_action, next_residual
            )
            target_q = reward + (1 - done) * (self.gamma ** n_step) * target_q
        
        # Current Q
        current_q = self.critic(state, base_action, residual_action)
        
        # Critic loss
        critic_loss = ((current_q - target_q.unsqueeze(0)) ** 2).mean()
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.grad_clip
        )
        self.critic_opt.step()
        
        if self.critic_scheduler:
            self.critic_scheduler.step()
        
        metrics["critic_loss"] = critic_loss.item()
        
        # =====================================================================
        # ACTOR UPDATE
        # =====================================================================
        if update_actor:
            residual_pred = self.actor(state, base_action)
            
            # L2 regularization
            l2_penalty = self.action_l2_reg * (residual_pred ** 2).sum(dim=-1).mean()
            
            # Q-value
            q_value = self.critic.q_mean(state, base_action, residual_pred)
            actor_loss = -q_value.mean() + l2_penalty
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip
            )
            self.actor_opt.step()
            
            if self.actor_scheduler:
                self.actor_scheduler.step()
            
            # Soft update
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_l2"] = l2_penalty.item()
            metrics["residual_magnitude"] = residual_pred.abs().mean().item()
        
        return metrics
    
    def save(self, path: str, step: int):
        """Save checkpoint."""
        torch.save({
            "step": step,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "action_scale": self.action_scale,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        return checkpoint.get("step", 0)
