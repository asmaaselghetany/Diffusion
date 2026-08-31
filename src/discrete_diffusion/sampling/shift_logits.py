"""Fast-dLLM logit shift alignment (train / eval / sample)."""

from __future__ import annotations

import torch


def align_shift_logits(logits: torch.Tensor, *, enabled: bool) -> torch.Tensor:
  """Align backbone logits with ``shift_loss_targets`` training grid.

  Matches ``block_qwen_lm_eval.get_loglikelihood`` and ``BlockTrainer`` when
  ``algo.shift_loss_targets=true``: position ``i`` predicts token ``i`` (not ``i+1``).
  """
  if not enabled:
    return logits
  return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


__all__ = ['align_shift_logits']
