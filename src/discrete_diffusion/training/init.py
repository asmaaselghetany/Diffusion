"""AR → block-diffusion init metrics (G5).

At step 0, compare causal (AR) logits with block-diff logits on
``concat(x0, x0)`` and log agreement statistics before any weight updates.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def logit_agreement_rho(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
  """Mean cosine similarity of per-position softmax distributions."""
  pa = F.softmax(logits_a.float(), dim=-1)
  pb = F.softmax(logits_b.float(), dim=-1)
  num = (pa * pb).sum(dim=-1)
  den = pa.norm(dim=-1) * pb.norm(dim=-1)
  rho = (num / den.clamp(min=1e-12)).mean()
  return float(rho.detach().cpu())


def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
  """Fraction of positions where argmax tokens match."""
  agree = (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean()
  return float(agree.detach().cpu())


@torch.no_grad()
def compute_ar_block_init_metrics(
    backbone,
    input_ids: torch.Tensor | None = None,
    *,
    seed: int = 0,
) -> dict[str, float]:
  """Compare AR causal vs block-diff logits at initialization.

  Args:
    backbone: ``QwenBlockForCausalLM`` (or compatible module).
    input_ids: Optional ``[B, n]`` token ids; random if omitted.
    seed: RNG seed when ``input_ids`` is omitted.

  Returns:
    Dict with ``logit_rho``, ``top1_agreement``, ``max_logit_diff``.
  """
  device = next(backbone.parameters()).device
  n = int(getattr(backbone, 'n_tokens', input_ids.shape[1] if input_ids is not None else 0))
  if input_ids is None:
    vocab = int(backbone.model.config.vocab_size)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    input_ids = torch.randint(0, vocab, (1, n), generator=gen, device=device)

  causal = backbone.causal_logits(input_ids)
  block = backbone.block_diff_logits(torch.cat([input_ids, input_ids], dim=-1))
  max_diff = (causal - block).abs().max()
  return {
      'logit_rho': logit_agreement_rho(causal, block),
      'top1_agreement': top1_agreement(causal, block),
      'max_logit_diff': float(max_diff.detach().cpu()),
  }


def log_ar_block_init_metrics(module: Any, metrics: dict[str, float]) -> None:
  """Log G5 init metrics via Lightning ``log`` when available."""
  for key, value in metrics.items():
    name = f'init/{key}'
    if hasattr(module, 'log'):
      module.log(name, value, on_step=False, on_epoch=True, sync_dist=False)
    else:
      print(f'{name}={value:.6f}')
