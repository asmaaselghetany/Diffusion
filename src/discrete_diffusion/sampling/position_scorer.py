"""Position scoring criteria for iterative sampling."""

import torch


class PositionScoringCriteria:
  """Base class for position scoring criteria during sampling."""

  def __call__(self, logits, device):
    """Score positions for selection during sampling.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        device: torch.device for tensor operations

    Returns:
        scores: Tensor of shape [batch_size, seq_len] with scores for each position.
                Higher scores indicate higher priority for selection.
    """
    raise NotImplementedError


class RandomPositionScorer(PositionScoringCriteria):
  """Random position scoring - assign random scores to all positions."""

  def __call__(self, logits, device):
    batch_size, seq_len = logits.shape[:2]
    return torch.rand((batch_size, seq_len), device=device, dtype=torch.float)


class ConfidencePositionScorer(PositionScoringCriteria):
  """Confidence-based position scoring - score positions based on model confidence."""

  def __call__(self, logits, device):
    probs = torch.softmax(logits, dim=-1)
    return torch.max(probs, dim=-1)[0]


class EntropyPositionScorer(PositionScoringCriteria):
  """Entropy-based position scoring - score positions based on prediction uncertainty."""

  def __call__(self, logits, device):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return -entropy  # Negative entropy so higher = more confident


class MarginPositionScorer(PositionScoringCriteria):
  """Margin-based position scoring - score positions based on the confidence margin."""

  def __call__(self, logits, device):
    probs = torch.softmax(logits, dim=-1)
    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
    return top2_probs[..., 0] - top2_probs[..., 1]


__all__ = [
  'PositionScoringCriteria',
  'RandomPositionScorer',
  'ConfidencePositionScorer',
  'EntropyPositionScorer',
  'MarginPositionScorer',
]

