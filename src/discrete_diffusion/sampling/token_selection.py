"""Token selection criteria for iterative sampling."""

import torch


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
    probs = torch.softmax(logits, dim=-1)
    if logits.dim() == 2:
      return torch.multinomial(probs, num_samples=1).squeeze(-1)
    batch_size, seq_len, vocab_size = probs.shape
    probs_flat = probs.view(batch_size * seq_len, vocab_size)
    sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
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
    top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
    probs = torch.softmax(top_k_values, dim=-1)
    if logits.dim() == 2:
      sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
      return torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    batch_size, seq_len = logits.shape[:2]
    probs_flat = probs.view(batch_size * seq_len, self.k)
    sampled_indices = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    sampled_indices = sampled_indices.view(batch_size, seq_len)
    return torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


class NucleusSelection(TokenSelectionCriteria):
  """Nucleus (top-p) sampling - sample from tokens comprising the top-p probability mass."""

  def __init__(self, p=0.9, temperature=1.0):
    assert 0 < p <= 1, "p must be in (0, 1]"
    assert temperature > 0, "Temperature must be positive"
    self.p = p
    self.temperature = temperature

  def __call__(self, logits):
    logits = logits / self.temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = cumulative_probs > self.p
    cutoff_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff_mask, float("-inf"))
    nucleus_probs = torch.softmax(sorted_logits, dim=-1)
    if logits.dim() == 2:
      sampled_flat = torch.multinomial(nucleus_probs, num_samples=1).squeeze(-1)
      return torch.gather(sorted_indices, -1, sampled_flat.unsqueeze(-1)).squeeze(-1)
    batch_size, seq_len, vocab_size = logits.shape
    nucleus_probs_flat = nucleus_probs.view(batch_size * seq_len, vocab_size)
    sampled_flat = torch.multinomial(nucleus_probs_flat, num_samples=1).squeeze(-1)
    sampled_indices = sampled_flat.view(batch_size, seq_len)
    return torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)


__all__ = [
  'TokenSelectionCriteria',
  'GreedySelection',
  'TemperatureSelection',
  'TopKSelection',
  'NucleusSelection',
]

