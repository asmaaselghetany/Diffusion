"""Latent JEPA-style discrete diffusion model.

Student encoder E_θ: Tokens → Latents (trainable)
Teacher encoder Ē_θ: EMA copy (frozen, no gradients)
Predictor P_θ: Noisy latents → Clean latents
Readout decoder D_ϕ: Latents → Token logits
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable
from torch.nn.parameter import Parameter

from .common import EmbeddingLayer, LayerNorm, TimestepEmbedder, DDiTBlock, Rotary


class LatentJEPA(nn.Module):
  """Latent JEPA discrete diffusion model."""

  def __init__(self, config, vocab_size: int):
    """Initialize LatentJEPA from config.
    
    Args:
      config: Hydra config with model parameters
      vocab_size: Vocabulary size from tokenizer
    """
    super().__init__()
    
    # Extract model config
    model_cfg = config.model
    
    # Core params
    self.vocab_size = vocab_size
    self.max_seq_len = getattr(model_cfg, 'length', 256)
    self.latent_dim = getattr(model_cfg, 'latent_dim', 256)
    self.hidden_size = getattr(model_cfg, 'hidden_size', 512)
    self.time_conditioning = getattr(model_cfg, 'time_conditioning', False)
    self.ema_decay = getattr(model_cfg, 'ema_decay', 0.996)
    self.ema_warmup_steps = getattr(model_cfg, 'ema_warmup_steps', 50000)
    self.ema_final_decay = getattr(model_cfg, 'ema_final_decay', 0.9999)
    
    # Local vars for init
    latent_dim = self.latent_dim
    hidden_size = self.hidden_size
    n_heads = getattr(model_cfg, 'n_heads', 8)
    n_blocks = getattr(model_cfg, 'n_blocks', 6)
    dropout = getattr(model_cfg, 'dropout', 0.1)
    time_conditioning = self.time_conditioning
    time_embed_dim = getattr(model_cfg, 'time_embed_dim', 128)
    predictor_type = getattr(model_cfg, 'predictor_type', 'mlp')
    predictor_depth = getattr(model_cfg, 'predictor_depth', 3)
    predictor_hidden_size = getattr(model_cfg, 'predictor_hidden_size', None)
    predictor_n_heads = getattr(model_cfg, 'predictor_n_heads', None)
    predictor_use_projections = getattr(model_cfg, 'predictor_use_projections', True)
    readout_type = getattr(model_cfg, 'readout_type', 'tied_linear')
    readout_hidden_size = getattr(model_cfg, 'readout_hidden_size', None)
    readout_depth = getattr(model_cfg, 'readout_depth', 2)
    readout_n_heads = getattr(model_cfg, 'readout_n_heads', None)
    readout_bidirectional = getattr(model_cfg, 'readout_bidirectional', True)
    tie_readout_to_embedding = getattr(model_cfg, 'tie_readout_to_embedding', True)
    copy_on_init = getattr(model_cfg, 'copy_on_init', True)

    # Student encoder (trainable)
    self.student_encoder = LatentEncoder(
      vocab_size=vocab_size, hidden_size=hidden_size, latent_dim=latent_dim,
      n_heads=n_heads, n_blocks=n_blocks, dropout=dropout,
      time_conditioning=time_conditioning, time_embed_dim=time_embed_dim)

    # Teacher encoder (EMA, no gradients)
    self.teacher_encoder = LatentEncoder(
      vocab_size=vocab_size, hidden_size=hidden_size, latent_dim=latent_dim,
      n_heads=n_heads, n_blocks=n_blocks, dropout=dropout,
      time_conditioning=time_conditioning, time_embed_dim=time_embed_dim)

    if copy_on_init:
      self._copy_student_to_teacher()
    for param in self.teacher_encoder.parameters():
      param.requires_grad = False

    # Predictor
    predictor_hidden = predictor_hidden_size or hidden_size
    if predictor_type == "mlp":
      self.predictor = MLPPredictor(
        latent_dim=latent_dim, hidden_size=predictor_hidden, depth=predictor_depth,
        dropout=dropout, time_conditioning=time_conditioning, time_embed_dim=time_embed_dim)
    elif predictor_type == "transformer":
      self.predictor = TransformerPredictor(
        latent_dim=latent_dim, hidden_size=predictor_hidden,
        n_heads=(predictor_n_heads or n_heads), n_blocks=predictor_depth,
        dropout=dropout, time_conditioning=time_conditioning,
        time_embed_dim=time_embed_dim, use_projections=predictor_use_projections)
    else:
      raise ValueError(f"Unknown predictor_type: {predictor_type}")

    # Readout
    readout_hidden = readout_hidden_size or latent_dim
    if readout_type == "tied_linear":
      self.readout = TiedLinearReadout(
        latent_dim=latent_dim, vocab_size=vocab_size,
        tie_to_embedding=tie_readout_to_embedding,
        embedding_layer=self.student_encoder.vocab_embed if tie_readout_to_embedding else None)
    elif readout_type == "mlp":
      self.readout = MLPReadout(
        latent_dim=latent_dim, hidden_size=readout_hidden,
        vocab_size=vocab_size, depth=readout_depth, dropout=dropout)
    elif readout_type == "tiny_transformer":
      self.readout = TinyTransformerReadout(
        latent_dim=latent_dim, hidden_size=readout_hidden, vocab_size=vocab_size,
        n_heads=n_heads, n_blocks=readout_depth, dropout=dropout)
    elif readout_type == "transformer":
      self.readout = TransformerReadout(
        latent_dim=latent_dim, hidden_size=readout_hidden, vocab_size=vocab_size,
        n_heads=(readout_n_heads or n_heads), n_blocks=readout_depth,
        dropout=dropout, bidirectional=readout_bidirectional)
    else:
      raise ValueError(f"Unknown readout_type: {readout_type}")

  def _copy_student_to_teacher(self):
    with torch.no_grad():
      for s_p, t_p in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
        t_p.data.copy_(s_p.data)

  def encode_student(self, input_ids: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    return self.student_encoder(input_ids, t)

  def encode_teacher(self, input_ids: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    with torch.no_grad():
      return self.teacher_encoder(input_ids, t)

  def predict_latent(self, z_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    return self.predictor(z_t, t)

  def readout_tokens(self, z: torch.Tensor) -> torch.Tensor:
    return self.readout(z)

  def forward(self, input_ids: torch.Tensor, t: Optional[torch.Tensor] = None,
              return_latents: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    z_t = self.encode_student(input_ids, t)
    z_hat_0 = self.predict_latent(z_t, t)
    logits = self.readout_tokens(z_hat_0)
    return (logits, z_hat_0) if return_latents else logits

  def optimizable_parameters(self) -> Iterable[Parameter]:
    return filter(lambda p: p.requires_grad, self.parameters())

  def decoder_parameters(self) -> Iterable[Parameter]:
    return self.readout.parameters()

  def update_ema(self, step: int):
    if self.ema_warmup_steps > 0 and step < self.ema_warmup_steps:
      ratio = step / self.ema_warmup_steps
      decay = self.ema_final_decay - (self.ema_final_decay - self.ema_decay) * 0.5 * (1.0 + math.cos(math.pi * ratio))
    else:
      decay = self.ema_final_decay if self.ema_warmup_steps > 0 else self.ema_decay

    with torch.no_grad():
      for s_p, t_p in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
        t_p.data.mul_(decay).add_(s_p.data, alpha=1 - decay)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
      with torch.no_grad():
        for t_p in self.teacher_encoder.parameters():
          torch.distributed.broadcast(t_p.data, src=0)


class LatentEncoder(nn.Module):
  """Transformer encoder: Tokens → Latents."""

  def __init__(self, vocab_size: int, hidden_size: int, latent_dim: int, n_heads: int,
               n_blocks: int, dropout: float, time_conditioning: bool, time_embed_dim: int):
    super().__init__()
    self.time_conditioning = time_conditioning
    self.n_heads = n_heads

    self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
    if time_conditioning:
      self.time_embedder = TimestepEmbedder(time_embed_dim)

    self.rotary_emb = Rotary(hidden_size // n_heads, base=10000)
    cond_dim = time_embed_dim if time_conditioning else 1
    self.blocks = nn.ModuleList([
      DDiTBlock(hidden_size, n_heads, adaLN=True, cond_dim=cond_dim, dropout=dropout)
      for _ in range(n_blocks)])

    # JEPA encoder: small random init instead of zero init
    for block in self.blocks:
      nn.init.normal_(block.adaLN_modulation.weight, mean=0.0, std=0.02)
      nn.init.zeros_(block.adaLN_modulation.bias)

    self.to_latent = nn.Linear(hidden_size, latent_dim)

  def forward(self, input_ids: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, L = input_ids.shape
    device = input_ids.device
    x = self.vocab_embed(input_ids)

    if self.time_conditioning:
      if t is None:
        t = torch.zeros(B, device=device, dtype=torch.float32)
      c = F.silu(self.time_embedder(t))
    else:
      c = torch.zeros(B, 1, device=device, dtype=x.dtype)

    rotary_cos_sin = self.rotary_emb(x)
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c)
    return self.to_latent(x)


class MLPPredictor(nn.Module):
  """MLP predictor: z_t (+ optional t) → z_hat_0."""

  def __init__(self, latent_dim: int, hidden_size: int, depth: int, dropout: float,
               time_conditioning: bool, time_embed_dim: int):
    super().__init__()
    self.time_conditioning = time_conditioning
    if time_conditioning:
      self.time_embedder = TimestepEmbedder(time_embed_dim)
      input_dim = latent_dim + time_embed_dim
    else:
      input_dim = latent_dim

    layers = []
    for i in range(depth):
      in_d = input_dim if i == 0 else hidden_size
      out_d = latent_dim if i == depth - 1 else hidden_size
      layers.append(nn.Linear(in_d, out_d))
      if i < depth - 1:
        layers.extend([nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout)])
    self.mlp = nn.Sequential(*layers)

  def forward(self, z_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, L, D = z_t.shape
    if self.time_conditioning:
      if t is None:
        t = torch.zeros(B, device=z_t.device, dtype=torch.float32)
      t_emb = F.silu(self.time_embedder(t)).unsqueeze(1).expand(B, L, -1)
      x = torch.cat([z_t, t_emb], dim=-1)
    else:
      x = z_t
    return self.mlp(x)


class TransformerPredictor(nn.Module):
  """Transformer predictor: z_t (+ optional t) → z_hat_0."""

  def __init__(self, latent_dim: int, hidden_size: int, n_heads: int, n_blocks: int,
               dropout: float, time_conditioning: bool, time_embed_dim: int,
               use_projections: bool = True):
    super().__init__()
    self.time_conditioning = time_conditioning
    self.n_heads = n_heads
    self.use_projections = use_projections

    if time_conditioning:
      self.time_embedder = TimestepEmbedder(time_embed_dim)

    self.input_norm = nn.LayerNorm(latent_dim)
    if use_projections or hidden_size != latent_dim:
      self.input_proj = nn.Linear(latent_dim, hidden_size)
      transformer_dim = hidden_size
    else:
      self.input_proj = nn.Identity()
      transformer_dim = latent_dim

    self.rotary_emb = Rotary(transformer_dim // n_heads, base=10000)
    cond_dim = time_embed_dim if time_conditioning else 1
    self.blocks = nn.ModuleList([
      DDiTBlock(transformer_dim, n_heads, adaLN=True, cond_dim=cond_dim, dropout=dropout)
      for _ in range(n_blocks)])

    for block in self.blocks:
      nn.init.normal_(block.adaLN_modulation.weight, mean=0.0, std=0.02)
      nn.init.zeros_(block.adaLN_modulation.bias)

    if use_projections or transformer_dim != latent_dim:
      self.output_proj = nn.Linear(transformer_dim, latent_dim)
    else:
      self.output_proj = nn.Identity()

  def forward(self, z_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, L, D = z_t.shape
    device = z_t.device
    x = self.input_proj(self.input_norm(z_t))

    if self.time_conditioning:
      if t is None:
        t = torch.zeros(B, device=device, dtype=torch.float32)
      c = F.silu(self.time_embedder(t))
    else:
      c = torch.zeros(B, 1, device=device, dtype=x.dtype)

    rotary_cos_sin = self.rotary_emb(x)
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c)
    return self.output_proj(x)


class TiedLinearReadout(nn.Module):
  """Tied linear readout: z → logits, optionally tied to embedding."""

  def __init__(self, latent_dim: int, vocab_size: int, tie_to_embedding: bool,
               embedding_layer: Optional[EmbeddingLayer] = None):
    super().__init__()
    self.tie_to_embedding = tie_to_embedding
    self.latent_dim = latent_dim

    if tie_to_embedding:
      assert embedding_layer is not None
      embedding_dim = embedding_layer.embedding.shape[1]
      self.projection = nn.Linear(latent_dim, embedding_dim) if latent_dim != embedding_dim else None
      self.embedding_layer = embedding_layer
      self.bias = nn.Parameter(torch.zeros(vocab_size))
    else:
      self.projection = None
      self.linear = nn.Linear(latent_dim, vocab_size)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    if self.tie_to_embedding:
      if self.projection is not None:
        z = self.projection(z)
      return F.linear(z, self.embedding_layer.embedding, self.bias)
    return self.linear(z)


class MLPReadout(nn.Module):
  """MLP readout: z → logits."""

  def __init__(self, latent_dim: int, hidden_size: int, vocab_size: int, depth: int, dropout: float):
    super().__init__()
    layers = []
    for i in range(depth):
      in_d = latent_dim if i == 0 else hidden_size
      out_d = vocab_size if i == depth - 1 else hidden_size
      layers.append(nn.Linear(in_d, out_d))
      if i < depth - 1:
        layers.extend([nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout)])
    self.mlp = nn.Sequential(*layers)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.mlp(z)


class TinyTransformerReadout(nn.Module):
  """Tiny transformer readout: z → logits (adds local context)."""

  def __init__(self, latent_dim: int, hidden_size: int, vocab_size: int,
               n_heads: int, n_blocks: int, dropout: float):
    super().__init__()
    self.n_heads = n_heads
    self.input_proj = nn.Linear(latent_dim, hidden_size)
    self.rotary_emb = Rotary(hidden_size // n_heads, base=10000)
    self.blocks = nn.ModuleList([
      DDiTBlock(hidden_size, n_heads, adaLN=True, cond_dim=1, dropout=dropout)
      for _ in range(n_blocks)])
    self.output_proj = nn.Linear(hidden_size, vocab_size)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    B = z.shape[0]
    x = self.input_proj(z)
    c = torch.zeros(B, 1, device=z.device, dtype=x.dtype)
    rotary_cos_sin = self.rotary_emb(x)
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c)
    return self.output_proj(x)


class TransformerReadout(nn.Module):
  """Flexible transformer readout: z → logits."""

  def __init__(self, latent_dim: int, hidden_size: int, vocab_size: int,
               n_heads: int, n_blocks: int, dropout: float, bidirectional: bool = True):
    super().__init__()
    self.n_heads = n_heads
    self.bidirectional = bidirectional

    if hidden_size == latent_dim:
      self.input_proj = nn.Identity()
      transformer_dim = latent_dim
    else:
      self.input_proj = nn.Linear(latent_dim, hidden_size)
      transformer_dim = hidden_size

    self.rotary_emb = Rotary(transformer_dim // n_heads, base=10000)
    self.blocks = nn.ModuleList([
      DDiTBlock(transformer_dim, n_heads, adaLN=True, cond_dim=1, dropout=dropout)
      for _ in range(n_blocks)])
    self.output_proj = nn.Linear(transformer_dim, vocab_size)

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    B = z.shape[0]
    x = self.input_proj(z)
    c = torch.zeros(B, 1, device=z.device, dtype=x.dtype)
    rotary_cos_sin = self.rotary_emb(x)
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c)
    return self.output_proj(x)

