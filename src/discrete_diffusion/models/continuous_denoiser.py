"""Continuous Denoiser model for embedding diffusion.

This module provides a transformer-based denoiser that operates on continuous
embeddings, predicting clean embeddings from noisy inputs conditioned on timestep.
"""

from __future__ import annotations

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .common import (
    DDiTBlock,
    DDiTFinalLayer,
    LayerNorm,
    Rotary,
    TimestepEmbedder,
)


def _forward_block_with_checkpoint(
    block,
    x,
    rotary_cos_sin,
    c,
    attn_mask,
    use_checkpoint: bool,
    training: bool
):
    """Forward through a single block with optional gradient checkpointing."""
    # torch.func.jvp (forward-mode AD) is incompatible with checkpoint wrappers.
    # For iMF/MeanFlow we bypass checkpointing during forward-mode passes.
    if use_checkpoint and training and (not torch._C._is_fwd_grad_enabled()):
        return torch_checkpoint(
            block, x, rotary_cos_sin, c, attn_mask, use_reentrant=False
        )
    return block(x, rotary_cos_sin, c, attn_mask=attn_mask)


class ContinuousDenoiser(nn.Module):
    """Transformer denoiser for continuous embedding diffusion.
    
    Takes noisy embeddings z_t and timestep t, predicts clean embeddings z_0
    (or noise epsilon, depending on parameterization).
    
    Architecture:
        1. Input projection: embed_dim -> hidden_size
        2. Time embedding via sinusoidal + MLP
        3. Stack of DDiT blocks with adaptive LayerNorm conditioning
        4. Output projection: hidden_size -> embed_dim
    
    Attributes:
        embed_dim: Input/output embedding dimension (e.g., 1024 for Qwen)
        hidden_size: Internal transformer dimension
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks
        parameterization: 'x0' (predict clean), 'epsilon' (predict noise), or 'v' (velocity)
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_size: int = 1024,
        n_heads: int = 16,
        n_blocks: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        parameterization: Literal["x0", "epsilon", "v"] = "x0",
        gradient_checkpointing: bool = False,
        use_input_norm: bool = True,
        use_output_norm: bool = True,
        residual_scale: float = 1.0,
        objective: str = "ddpm",
        attn_backend: str = "auto",
        use_velocity_head: bool = True,
        use_u_head: bool = True,
    ):
        """Initialize the continuous denoiser.
        
        Args:
            embed_dim: Input/output embedding dimension (match encoder output)
            hidden_size: Internal transformer hidden dimension
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks (depth)
            mlp_ratio: MLP hidden dim multiplier
            dropout: Dropout probability
            time_embed_dim: Dimension of time embedding
            parameterization: What the model predicts ('x0', 'epsilon', 'v')
            gradient_checkpointing: Enable gradient checkpointing for memory
            use_input_norm: Apply LayerNorm to input embeddings
            use_output_norm: Apply LayerNorm before output projection
            residual_scale: Scale factor for residual connections
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.parameterization = parameterization
        self.gradient_checkpointing = gradient_checkpointing
        self.residual_scale = residual_scale
        self.objective = objective
        self.attn_backend = attn_backend
        
        # Input processing
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.input_norm = LayerNorm(embed_dim)
        
        # Project from embed_dim to hidden_size
        if embed_dim != hidden_size:
            self.input_proj = nn.Linear(embed_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()
        
        # Time embedding
        self.time_embedder = TimestepEmbedder(time_embed_dim)
        self.r_embedder = TimestepEmbedder(time_embed_dim)
        
        # Position embeddings (rotary)
        self.rotary_emb = Rotary(hidden_size // n_heads, base=10000)
        
        # Transformer blocks with adaptive LN conditioning
        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=hidden_size,
                n_heads=n_heads,
                adaLN=True,
                cond_dim=time_embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_backend=attn_backend,
            )
            for _ in range(n_blocks)
        ])
        
        # Initialize modulation weights (small random for stability)
        for block in self.blocks:
            nn.init.normal_(block.adaLN_modulation.weight, mean=0.0, std=0.02)
            nn.init.zeros_(block.adaLN_modulation.bias)
        
        # Output processing
        self.use_output_norm = use_output_norm
        if use_output_norm:
            self.output_norm = LayerNorm(hidden_size)
        
        # Project back to embed_dim
        if embed_dim != hidden_size:
            self.output_proj = nn.Linear(hidden_size, embed_dim)
        else:
            self.output_proj = nn.Identity()
        self.velocity_head = nn.Linear(hidden_size, embed_dim) if use_velocity_head else None
        self.u_head = nn.Linear(hidden_size, embed_dim) if use_u_head else None
        
        # Initialize output projection to zero for residual learning
        if isinstance(self.output_proj, nn.Linear):
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        if self.velocity_head is not None:
            nn.init.zeros_(self.velocity_head.weight)
            nn.init.zeros_(self.velocity_head.bias)
        if self.u_head is not None:
            nn.init.zeros_(self.u_head.weight)
            nn.init.zeros_(self.u_head.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        head: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass: predict target from noisy embeddings.
        
        Args:
            z_t: Noisy embeddings [B, L, embed_dim]
            t: Timesteps [B] with values in [0, 1]
            attention_mask: Optional attention mask [B, L]
            
        Returns:
            Predicted target [B, L, embed_dim]:
                - If parameterization='x0': predicted clean embeddings
                - If parameterization='epsilon': predicted noise
                - If parameterization='v': predicted velocity
        """
        B, L, D = z_t.shape
        device = z_t.device
        del cond, span_mask
        
        # Prepare attention mask for transformer blocks
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (batch, seq_len)")
            # SDPA boolean masks use True=keep and False=masked-out.
            attn_mask = attention_mask.to(device=device, dtype=torch.bool)
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L]
        
        # Input processing
        x = z_t
        if self_cond is not None:
            x = x + self_cond
        if self.use_input_norm:
            x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Time conditioning
        c = F.silu(self.time_embedder(t))  # [B, time_embed_dim]
        if r is not None:
            c = c + F.silu(self.r_embedder(r))
        
        # Rotary position embeddings
        rotary_cos_sin = self.rotary_emb(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = _forward_block_with_checkpoint(
                block, x, rotary_cos_sin, c, attn_mask,
                self.gradient_checkpointing, self.training
            )
        
        # Output processing
        if self.use_output_norm:
            x = self.output_norm(x)
        if head == "velocity":
            if self.velocity_head is None:
                output = self.output_proj(x)
            else:
                output = self.velocity_head(x)
        elif head == "u":
            if self.u_head is None:
                output = self.output_proj(x)
            else:
                output = self.u_head(x)
        else:
            output = self.output_proj(x)
        
        # Optional residual connection for x0 prediction
        if head is None and self.parameterization == "x0" and self.residual_scale > 0:
            output = output + self.residual_scale * z_t
        
        return output

    def set_attention_backend(self, backend: str) -> None:
        self.attn_backend = backend
        for block in self.blocks:
            block.attn_backend = backend

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            z_t=z_t,
            t=t,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
            cond=cond,
            head="velocity",
        )

    def predict_u(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            z_t=z_t,
            t=t,
            r=r,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
            cond=cond,
            head="u",
        )
    
    def predict_x0(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict clean embeddings z_0 from noisy z_t.
        
        Handles different parameterizations by converting to x0 prediction.
        
        Args:
            z_t: Noisy embeddings [B, L, D]
            t: Timesteps [B]
            alpha_t: Schedule alpha values [B, 1, 1] or broadcastable
            attention_mask: Optional attention mask
            
        Returns:
            Predicted clean embeddings z_0 [B, L, D]
        """
        model_output = self.forward(z_t, t, attention_mask)
        
        if self.parameterization == "x0":
            return model_output
        
        elif self.parameterization == "epsilon":
            # z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * epsilon
            # z_0 = (z_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
            z_0 = (z_t - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t.clamp(min=1e-20)
            return z_0
        
        elif self.parameterization == "v":
            # v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * z_0
            # z_0 = (sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v) / 1
            #     = sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v
            # Actually: z_0 = sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v (when alpha_t + (1-alpha_t) = 1)
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
            z_0 = sqrt_alpha_t * z_t - sqrt_one_minus_alpha_t * model_output
            return z_0
        
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def predict_epsilon(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise epsilon from noisy z_t.
        
        Args:
            z_t: Noisy embeddings [B, L, D]
            t: Timesteps [B]
            alpha_t: Schedule alpha values [B, 1, 1] or broadcastable
            attention_mask: Optional attention mask
            
        Returns:
            Predicted noise epsilon [B, L, D]
        """
        model_output = self.forward(z_t, t, attention_mask)
        
        if self.parameterization == "epsilon":
            return model_output
        
        elif self.parameterization == "x0":
            # epsilon = (z_t - sqrt(alpha_t) * z_0) / sqrt(1 - alpha_t)
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
            epsilon = (z_t - sqrt_alpha_t * model_output) / sqrt_one_minus_alpha_t.clamp(min=1e-20)
            return epsilon
        
        elif self.parameterization == "v":
            # v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * z_0
            # Need z_0 first, then compute epsilon
            z_0 = self.predict_x0(z_t, t, alpha_t, attention_mask)
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
            epsilon = (z_t - sqrt_alpha_t * z_0) / sqrt_one_minus_alpha_t.clamp(min=1e-20)
            return epsilon
        
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")


class ContinuousEmbeddingModel(nn.Module):
    """Full continuous embedding diffusion model.
    
    Combines:
        1. Frozen embedding encoder (e.g., Qwen3-Embedding)
        2. Continuous denoiser (trainable transformer)
        3. Learned decoder (maps embeddings to token logits)
    
    This is the main model class used by the training algorithm.
    """
    
    def __init__(
        self,
        config,
        vocab_size: int,
    ):
        """Initialize the continuous embedding model from config.
        
        Args:
            config: Hydra config with model parameters
            vocab_size: Vocabulary size from tokenizer
        """
        super().__init__()
        
        model_cfg = config.model
        
        # Core dimensions
        self.vocab_size = vocab_size
        self.embed_dim = getattr(model_cfg, 'embed_dim', 1024)
        self.hidden_size = getattr(model_cfg, 'hidden_size', 1024)
        self.max_seq_len = getattr(model_cfg, 'length', 256)
        
        # Denoiser config
        n_heads = getattr(model_cfg, 'n_heads', 16)
        n_blocks = getattr(model_cfg, 'n_blocks', 12)
        mlp_ratio = getattr(model_cfg, 'mlp_ratio', 4)
        dropout = getattr(model_cfg, 'dropout', 0.1)
        time_embed_dim = getattr(model_cfg, 'time_embed_dim', 256)
        parameterization = getattr(model_cfg, 'parameterization', 'x0')
        objective = getattr(model_cfg, 'objective', 'ddpm')
        attn_backend = getattr(model_cfg, 'attn_backend', 'auto')
        gradient_checkpointing = getattr(model_cfg, 'gradient_checkpointing', False)
        
        # Decoder config
        decoder_hidden_size = getattr(model_cfg, 'decoder_hidden_size', None) or self.embed_dim
        decoder_n_heads = getattr(model_cfg, 'decoder_n_heads', None) or n_heads
        decoder_n_blocks = getattr(model_cfg, 'decoder_n_blocks', 2)
        
        # Build denoiser
        self.denoiser = ContinuousDenoiser(
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            n_heads=n_heads,
            n_blocks=n_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
            parameterization=parameterization,
            gradient_checkpointing=gradient_checkpointing,
            objective=objective,
            attn_backend=attn_backend,
        )
        
        # Build decoder (maps embeddings to token logits)
        self.decoder = TransformerDecoder(
            embed_dim=self.embed_dim,
            hidden_size=decoder_hidden_size,
            vocab_size=vocab_size,
            n_heads=decoder_n_heads,
            n_blocks=decoder_n_blocks,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """Forward pass through denoiser and optionally decoder.
        
        Args:
            z_t: Noisy embeddings [B, L, embed_dim]
            t: Timesteps [B]
            attention_mask: Optional attention mask [B, L]
            return_logits: If True, also decode to token logits
            
        Returns:
            If return_logits: token logits [B, L, vocab_size]
            Else: predicted clean embeddings [B, L, embed_dim]
        """
        z_hat_0 = self.denoiser(z_t, t, attention_mask)
        
        if return_logits:
            return self.decoder(z_hat_0)
        return z_hat_0
    
    def denoise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise embeddings (predict z_0 from z_t)."""
        return self.denoiser(z_t, t, attention_mask)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to token logits."""
        return self.decoder(z)
    
    def optimizable_parameters(self):
        """Return all trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def denoiser_parameters(self):
        """Return denoiser parameters."""
        return self.denoiser.parameters()
    
    def decoder_parameters(self):
        """Return decoder parameters."""
        return self.decoder.parameters()


class TransformerDecoder(nn.Module):
    """Transformer decoder: embeddings -> token logits.
    
    Maps continuous embeddings back to discrete token probabilities.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        vocab_size: int,
        n_heads: int = 8,
        n_blocks: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """Initialize transformer decoder.
        
        Args:
            embed_dim: Input embedding dimension
            hidden_size: Internal hidden dimension
            vocab_size: Output vocabulary size
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks
            dropout: Dropout probability
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.gradient_checkpointing = gradient_checkpointing
        
        # Input projection
        if embed_dim != hidden_size:
            self.input_proj = nn.Linear(embed_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()
        
        # Position embeddings
        self.rotary_emb = Rotary(hidden_size // n_heads, base=10000)
        
        # Transformer blocks (no time conditioning)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=hidden_size,
                n_heads=n_heads,
                adaLN=False,  # No adaptive LN for decoder
                cond_dim=1,   # Dummy
                mlp_ratio=4,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])
        
        # Output projection to vocabulary
        self.output_norm = LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to token logits.
        
        Args:
            z: Input embeddings [B, L, embed_dim]
            
        Returns:
            Token logits [B, L, vocab_size]
        """
        B = z.shape[0]
        
        x = self.input_proj(z)
        rotary_cos_sin = self.rotary_emb(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = _forward_block_with_checkpoint(
                block, x, rotary_cos_sin, None, None,
                self.gradient_checkpointing, self.training
            )
        
        # Output
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits


__all__ = [
    "ContinuousDenoiser",
    "ContinuousEmbeddingModel", 
    "TransformerDecoder",
]
