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
from .token_selection import GreedySelection


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

  def __init__(self, config, position_scorer=None, token_selector=None, commit_fraction=None):
    self.config = config
    self.position_scorer = position_scorer or ConfidencePositionScorer()
    self.token_selector = token_selector or GreedySelection()
    self.commit_fraction = commit_fraction

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
      logits = model.readout_tokens(z_hat_0)

      # Determine transfer probability
      p_transfer = self.commit_fraction if self.commit_fraction is not None else (1 - s / t if i < steps - 1 else 1.0)

      # Score positions
      scores = self.position_scorer(logits, device)
      scores = scores.masked_fill(~mask_index, -float("inf"))

      # Determine how many positions to transfer per sequence
      masked_counts = mask_index.sum(dim=-1)
      num_to_transfer = torch.ceil(masked_counts.float() * p_transfer).to(torch.long)
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

