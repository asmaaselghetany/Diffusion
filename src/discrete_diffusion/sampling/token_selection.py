"""Token selection criteria for iterative sampling."""

import torch


def _safe_sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
  """Numerically-stable categorical sampling from unnormalized logits.

  Handles NaN/Inf and degenerate all-masked rows by falling back to argmax.
  Expects 2D logits shaped [N, V].
  """
  if logits.ndim != 2:
    raise ValueError(f"_safe_sample_from_logits expects 2D logits, got shape {tuple(logits.shape)}")

  # Work in fp32 for stable softmax/multinomial.
  work = logits.float()
  work = torch.nan_to_num(work, nan=-1e4, posinf=1e4, neginf=-1e4)

  has_finite = torch.isfinite(work).any(dim=-1)
  if not torch.all(has_finite):
    work = work.clone()
    work[~has_finite, 0] = 0.0

  probs = torch.softmax(work, dim=-1)
  probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

  denom = probs.sum(dim=-1, keepdim=True)
  bad_rows = denom.squeeze(-1) <= 0
  if torch.any(bad_rows):
    probs = probs.clone()
    probs[bad_rows] = 0.0
    fallback = torch.argmax(work[bad_rows], dim=-1)
    probs[bad_rows, fallback] = 1.0
    denom = probs.sum(dim=-1, keepdim=True)

  probs = probs / denom.clamp(min=1e-12)
  return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TokenSelectionCriteria:
  """Base class for token selection criteria during sampling."""

  def __call__(self, logits):
    """Select tokens from logits.

    Args:
        logits: Tensor of shape [N, vocab_size] or [batch_size, seq_len, vocab_size]

    Returns:
        selected_tokens: Tensor of token indices
    """
    raise NotImplementedError


class GreedySelection(TokenSelectionCriteria):
  """Greedy token selection - always choose the token with highest probability."""

  def __call__(self, logits):
    return torch.argmax(logits, dim=-1)


class TemperatureSelection(TokenSelectionCriteria):
  """Temperature-based sampling from the full vocabulary distribution."""

  def __init__(self, temperature=1.0):
    assert temperature > 0, "Temperature must be positive"
    self.temperature = temperature

  def __call__(self, logits):
    logits = logits / self.temperature
    if logits.dim() == 2:
      return _safe_sample_from_logits(logits)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    sampled_flat = _safe_sample_from_logits(logits_flat)
    return sampled_flat.view(batch_size, seq_len)


class TopKSelection(TokenSelectionCriteria):
  """Top-k sampling - sample from the k most likely tokens."""

  def __init__(self, k=50, temperature=1.0):
    assert k > 0, "k must be positive"
    assert temperature > 0, "Temperature must be positive"
    self.k = k
    self.temperature = temperature

  def __call__(self, logits):
    logits = logits / self.temperature
    k = min(self.k, logits.size(-1))
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    if logits.dim() == 2:
      sampled_indices = _safe_sample_from_logits(top_k_values)
      return torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    batch_size, seq_len = logits.shape[:2]
    values_flat = top_k_values.view(batch_size * seq_len, k)
    sampled_indices = _safe_sample_from_logits(values_flat)
    sampled_indices = sampled_indices.view(batch_size, seq_len)
    return torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


class NucleusSelection(TokenSelectionCriteria):
  """Nucleus (top-p) sampling - sample from tokens comprising the top-p probability mass."""

  def __init__(self, p=0.9, temperature=1.0, min_tokens_to_keep=1):
    assert 0 < p <= 1, "p must be in (0, 1]"
    assert temperature > 0, "Temperature must be positive"
    assert min_tokens_to_keep >= 1, "min_tokens_to_keep must be >= 1"
    self.p = p
    self.temperature = temperature
    self.min_tokens_to_keep = min_tokens_to_keep

  def __call__(self, logits):
    logits = logits / self.temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = cumulative_probs > self.p
    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
    cutoff_mask[..., 0] = False
    if self.min_tokens_to_keep > 1:
      cutoff_mask[..., :self.min_tokens_to_keep] = False
    sorted_logits = sorted_logits.masked_fill(cutoff_mask, float("-inf"))
    if logits.dim() == 2:
      sampled_flat = _safe_sample_from_logits(sorted_logits)
      return torch.gather(sorted_indices, -1, sampled_flat.unsqueeze(-1)).squeeze(-1)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = sorted_logits.view(batch_size * seq_len, vocab_size)
    sampled_flat = _safe_sample_from_logits(logits_flat)
    sampled_indices = sampled_flat.view(batch_size, seq_len)
    return torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


__all__ = [
  'TokenSelectionCriteria',
  'GreedySelection',
  'TemperatureSelection',
  'TopKSelection',
  'NucleusSelection',
]
