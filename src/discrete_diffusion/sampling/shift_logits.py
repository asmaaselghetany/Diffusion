"""Fast-dLLM logit shift alignment (train / eval / sample)."""

from __future__ import annotations

from typing import Literal

import torch

ShiftMode = Literal['full', 'window']


def align_shift_logits(
    logits: torch.Tensor,
    *,
    enabled: bool,
    window: tuple[int, int] | None = None,
    mode: ShiftMode | None = None,
) -> torch.Tensor:
  """Align backbone logits with ``shift_loss_targets`` training grid.

  Matches ``block_qwen_lm_eval.get_loglikelihood`` and ``BlockTrainer`` when
  ``algo.shift_loss_targets=true``: position ``i`` predicts token ``i``
  (not ``i+1``).

  Modes
  -----
  ``mode='full'`` (default when ``mode is None``)
      Shift the **whole** logit tensor::

          cat([logits[:, :1], logits[:, :-1]], dim=1)

      Use for dense / prefill / hierarchical full-block forwards (Hub
      ``batch_sample``: shift block logits, *then* slice the denoise window).
      ``window`` is ignored for the shift itself.

  ``mode='window'``
      Shift **inside** ``window=(w0, w1)`` only so ``aligned[w0]`` keeps
      ``logits[w0]`` (first column of the window). **Only** correct when
      DualCache ``replace`` zero-pads outside the window — a full-seq shift
      would pull a zero vector into ``aligned[w0]`` → greedy token-id 0.

  Never pass ``mode='window'`` merely because a denoise sub-window is set;
  dense/prefill paths must stay ``mode='full'``.
  """
  if not enabled:
    return logits
  shift_mode: ShiftMode = 'full' if mode is None else mode
  if shift_mode == 'full':
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
  if shift_mode != 'window':
    raise ValueError(f'unknown shift mode {mode!r}')
  if window is None:
    raise ValueError("mode='window' requires window=(w0, w1)")
  w0, w1 = window
  if not (0 <= w0 < w1 <= logits.size(1)):
    raise ValueError(f'bad shift window {window} for len={logits.size(1)}')
  out = logits.clone()
  win = logits[:, w0:w1, :]
  out[:, w0:w1, :] = torch.cat([win[:, :1], win[:, :-1]], dim=1)
  return out


__all__ = ['align_shift_logits', 'ShiftMode']
