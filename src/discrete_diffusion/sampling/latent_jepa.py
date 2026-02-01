"""Latent sampler for JEPA-style discrete diffusion.

This sampler iteratively denoises in latent space:
1. Encode current masked sequence: z_t = E_θ(x_t)
2. Predict clean latents: ẑ_0 = P_θ(z_t, t)
3. Decode to token logits: logits = D_ϕ(ẑ_0)
4. Select positions and tokens using existing criteria
5. Commit subset of masked tokens
6. Repeat until no masks remain
"""

from __future__ import annotations

import torch

from .base import Sampler
from .position_scorer import ConfidencePositionScorer
from .token_selection import (
  GreedySelection,
  NucleusSelection,
  TemperatureSelection,
  TopKSelection,
)


class LatentJEPASampler(Sampler):
  """Latent-space discrete diffusion sampler.

  Generates sequences by iteratively denoising masked tokens through latent space.

  Args:
      config: Sampling configuration with optional attributes:
          - sampling.steps: Number of denoising steps
          - sampling.inject_bos: Whether to inject BOS token
      position_scorer: Criteria for selecting which positions to denoise
      token_selector: Criteria for sampling tokens from logits
      commit_fraction: Fraction of remaining masks to commit per step (if None, computed from timesteps)
  """

  def __init__(
    self,
    config,
    position_scorer=None,
    token_selector=None,
    commit_fraction=None,
    commit_schedule=None,
    top_p=None,
    top_k=None,
    temperature=None,
    min_tokens_to_keep=None,
    ban_special_tokens=None,
    forward_process=None,
    latent_norm=None,
    **kwargs,
  ):
    self.config = config
    # Latent normalization layer (e.g., LayerNorm) to apply before readout
    # This ensures decoder sees normalized latents consistent with training
    self.latent_norm = latent_norm
    self.position_scorer = position_scorer or ConfidencePositionScorer()
    self.commit_fraction = commit_fraction
    self.commit_schedule = commit_schedule or getattr(
      self.config.sampling, "commit_schedule", "hazard"
    )
    self.top_p = top_p if top_p is not None else getattr(
      self.config.sampling, "top_p", getattr(self.config.sampling, "p_nucleus", 1.0)
    )
    self.top_k = top_k if top_k is not None else getattr(self.config.sampling, "top_k", 0)
    self.temperature = temperature if temperature is not None else getattr(
      self.config.sampling, "temperature", 1.0
    )
    self.min_tokens_to_keep = min_tokens_to_keep if min_tokens_to_keep is not None else getattr(
      self.config.sampling, "min_tokens_to_keep", 1
    )
    self.ban_special_tokens = ban_special_tokens if ban_special_tokens is not None else getattr(
      self.config.sampling, "ban_special_tokens", True
    )
    self.position_score_noise = getattr(self.config.sampling, "position_score_noise", 0.0)
    self.forward_process = forward_process
    self.token_selector = token_selector or self._build_token_selector()

  def _build_token_selector(self):
    if self.top_p is not None and self.top_p < 1.0:
      return NucleusSelection(
        p=self.top_p,
        temperature=self.temperature,
        min_tokens_to_keep=self.min_tokens_to_keep,
      )
    if self.top_k is not None and self.top_k > 0:
      return TopKSelection(k=self.top_k, temperature=self.temperature)
    if self.temperature is not None and self.temperature != 1.0:
      return TemperatureSelection(temperature=self.temperature)
    return GreedySelection()

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Generate samples using latent diffusion denoising.

    Args:
        model: LatentJEPA model with encode_student, predict_latent, readout_tokens
        num_samples: Number of sequences to generate
        num_steps: Number of sampling steps (overrides config if provided)
        eps: Small epsilon for timestep bounds
        inject_bos: Whether to inject BOS token at position 0

    Returns:
        torch.Tensor: Generated sequences of shape [num_samples, seq_length]
    """
    model.eval()
    device = next(model.parameters()).device
    mask_id = model.mask_id if hasattr(model, 'mask_id') else model.tokenizer.mask_token_id
    seq_length = model.num_tokens if hasattr(model, 'num_tokens') else model.max_seq_len

    steps = num_steps if num_steps is not None else getattr(self.config.sampling, 'steps', 64)
    inject_bos = self.config.sampling.inject_bos if inject_bos is None else inject_bos

    # Initialize with all masks
    x = torch.full((num_samples, seq_length), mask_id, dtype=torch.long, device=device)

    if inject_bos and hasattr(model, 'tokenizer'):
      x[:, 0] = model.tokenizer.bos_token_id

    # Fixed positions mask
    fix_mask = x != mask_id

    banned_token_ids = None
    if self.ban_special_tokens:
      banned_token_ids = {int(mask_id)}
      tokenizer = getattr(model, "tokenizer", None)
      if tokenizer is not None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
          banned_token_ids.add(int(pad_id))
    if banned_token_ids:
      banned_token_ids = sorted(banned_token_ids)

    # Timestep schedule
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    for i in range(steps):
      mask_index = x == mask_id
      if not mask_index.any():
        break

      t, s = timesteps[i], timesteps[i + 1]
      t_batch = torch.full((num_samples,), t, device=device, dtype=torch.float32)

      # Latent forward pass
      z_t = model.encode_student(x, t_batch)
      z_hat_0 = model.predict_latent(z_t, t_batch)
      
      # Apply latent normalization before readout (consistent with training)
      if self.latent_norm is not None:
        z_hat_0 = self.latent_norm(z_hat_0)
      
      logits = model.readout_tokens(z_hat_0)

      # Score positions
      if banned_token_ids:
        logits[..., banned_token_ids] = -float("inf")
      scores = self.position_scorer(logits, device)
      if self.position_score_noise > 0:
        scores = scores + torch.randn_like(scores) * self.position_score_noise
      scores = scores.masked_fill(~mask_index, -float("inf"))

      # Determine how many positions to transfer per sequence
      masked_counts = mask_index.sum(dim=-1)
      if self.commit_fraction is not None:
        p_transfer = self.commit_fraction
        num_to_transfer = torch.ceil(masked_counts.float() * p_transfer).to(torch.long)
      elif self.commit_schedule == "uniform":
        remaining_steps = max(1, steps - i)
        num_to_transfer = torch.ceil(masked_counts.float() / remaining_steps).to(torch.long)
      elif self.commit_schedule == "hazard":
        p_transfer = 1 - s / t if i < steps - 1 else 1.0
        num_to_transfer = torch.ceil(masked_counts.float() * p_transfer).to(torch.long)
      else:
        raise ValueError(f"Unknown commit_schedule: {self.commit_schedule}")
      num_to_transfer = torch.minimum(num_to_transfer, masked_counts)

      batch_size, seq_len = x.shape

      # Rank positions by score
      order = torch.argsort(scores, dim=-1, descending=True)

      # Build boolean mask selecting top-k per row
      rank_threshold = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) < num_to_transfer.unsqueeze(1)

      select_mask = torch.zeros_like(mask_index, dtype=torch.bool)
      select_mask.scatter_(1, order, rank_threshold)
      select_mask = select_mask & mask_index

      # Select and commit tokens
      if select_mask.any():
        selected_logits = logits[select_mask]
        selected_tokens = self.token_selector(selected_logits)
        x[select_mask] = selected_tokens

    model.train()
    return x


__all__ = ['LatentJEPASampler']
