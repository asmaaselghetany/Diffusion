"""BlockGen-style ARPC scoring helpers (default-off decode levers)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def divergence_scores(
    p_ar: torch.Tensor,
    p_diff: torch.Tensor,
    measure: str = 'kld',
) -> torch.Tensor:
  """Per-token divergence between AR and diffusion predictive dists.

  Args:
    p_ar / p_diff: ``[B, L, V]`` probabilities
    measure: ``kld`` | ``reverse_kld`` | ``tvd``
  Returns:
    ``[B, L]`` higher = more disagreement (prefer to re-noise).
  """
  eps = 1e-12
  if measure == 'kld':
    return (p_ar * ((p_ar + eps).log() - (p_diff + eps).log())).sum(-1)
  if measure == 'reverse_kld':
    return (p_diff * ((p_diff + eps).log() - (p_ar + eps).log())).sum(-1)
  if measure == 'tvd':
    return 0.5 * (p_ar - p_diff).abs().sum(-1)
  raise ValueError(f'Unknown divergence_measure={measure!r}')


def diffusion_scores(log_p_x0: torch.Tensor, metric: str = 'confidence') -> torch.Tensor:
  """Uncertainty under the diffusion predictive distribution.

  Higher = more uncertain (prefer to re-noise).
  """
  p_x0 = log_p_x0.exp()
  if metric == 'entropy':
    return torch.special.entr(p_x0).sum(-1)
  if metric == 'confidence':
    return -p_x0.max(dim=-1).values
  if metric == 'margin':
    top2 = torch.topk(p_x0, 2, dim=-1).values
    return -(top2[..., 0] - top2[..., 1])
  raise ValueError(f'Unknown diffusion_metric={metric!r}')


def ar_scores(
    log_p_ar: torch.Tensor,
    x_current: torch.Tensor,
    metric: str = 'nll',
) -> torch.Tensor:
  """How bad the current tokens look under the AR predictive dist."""
  if metric == 'nll':
    return -log_p_ar.gather(-1, x_current.unsqueeze(-1)).squeeze(-1)
  if metric == 'gap_to_top1':
    log_top1 = log_p_ar.max(-1).values
    log_xhat = log_p_ar.gather(-1, x_current.unsqueeze(-1)).squeeze(-1)
    return log_top1 - log_xhat
  if metric == 'entropy':
    return torch.special.entr(log_p_ar.exp()).sum(-1)
  raise ValueError(f'Unknown ar_metric={metric!r}')


def corruption_indices(
    *,
    log_p_x0: torch.Tensor,
    log_p_ar: torch.Tensor | None,
    x_current: torch.Tensor,
    num_to_corrupt: int,
    block_prefix_len: int,
    corruption_mode: str,
    divergence_measure: str = 'kld',
    diffusion_metric: str = 'confidence',
    ar_metric: str = 'nll',
) -> torch.Tensor:
  """Select non-prefix positions to re-noise. Returns ``[B, K]`` indices."""
  bsz, block_len, _ = log_p_x0.shape
  k = max(1, min(int(num_to_corrupt), block_len))
  if corruption_mode == 'random':
    scores = torch.rand(bsz, block_len, device=log_p_x0.device)
  elif corruption_mode == 'divergence':
    if log_p_ar is None:
      raise ValueError('divergence mode requires log_p_ar')
    scores = divergence_scores(
        log_p_ar.exp(), log_p_x0.exp(), measure=divergence_measure)
  elif corruption_mode == 'diffusion_metric':
    scores = diffusion_scores(log_p_x0, metric=diffusion_metric)
  elif corruption_mode == 'ar_metric':
    if log_p_ar is None:
      raise ValueError('ar_metric mode requires log_p_ar')
    scores = ar_scores(log_p_ar, x_current, metric=ar_metric)
  else:
    raise ValueError(f'Unknown corruption_mode={corruption_mode!r}')

  if block_prefix_len > 0:
    scores = scores.clone()
    scores[:, :block_prefix_len] = float('-inf')
  return torch.topk(scores, k, dim=-1).indices


def causal_log_probs_for_span(
    logits: torch.Tensor,
    start: int,
    end: int,
    vocab_size: int,
) -> torch.Tensor:
  """Build ``[B, end-start, V]`` log-probs for tokens ``[start:end)``.

  HF causal logits at position ``j`` predict token ``j+1``. Token 0 has no
  AR context → uniform log-prob.
  """
  del vocab_size  # kept for call-site clarity
  log_p = F.log_softmax(logits, dim=-1)
  bsz = logits.shape[0]
  v = logits.shape[-1]
  device = logits.device
  dtype = log_p.dtype
  out = []
  for j in range(start, end):
    if j == 0:
      out.append(
          torch.full((bsz, v), -torch.log(torch.tensor(float(v), device=device)),
                     device=device, dtype=dtype))
    else:
      # logits length may be `end` (scored on x0[:, :end])
      out.append(log_p[:, j - 1, :])
  return torch.stack(out, dim=1)


__all__ = [
    'ar_scores',
    'causal_log_probs_for_span',
    'corruption_indices',
    'diffusion_scores',
    'divergence_scores',
]
