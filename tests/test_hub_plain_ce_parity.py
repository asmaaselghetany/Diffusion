"""Hub-faithful plain_ce / complementary reduction — numerical parity lock.

Why this exists: reading ``hub_ref/modeling.py`` and claiming "matches" has
missed critical bugs twice (plain_ce sign; denom). This test encodes Hub's
*observable* train loss reduction without importing Hub (transformers version
skew). If this fails, C2 is not Hub-CE.

Hub modeling.py (training):
  noisy[labels!=-100] = corrupt(...)
  labels[noisy != mask_id] = -100          # CE only on MASK sites
  complementary view; cat on batch
  ForCausalLMLoss: shift logits/labels, mean over labels != -100
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.forward_process.block_masked import (
    BlockMaskedForwardProcess,
    complementary_pair_from_mask,
)


def _hub_shifted_mask_ce_mean(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    *,
    mask_id: int,
    supervised: torch.Tensor,
) -> torch.Tensor:
  """Hub ForCausalLMLoss on one view after their label rewrite + shift."""
  # labels start as x0 on supervised, -100 elsewhere
  labels = x0.clone()
  labels = torch.where(supervised.bool(), labels, torch.full_like(labels, -100))
  # Hub: only supervised positions may be noisy; then clean → -100
  labels = torch.where(xt == mask_id, labels, torch.full_like(labels, -100))
  # CausalLM shift
  shift_logits = logits[:, :-1].contiguous()
  shift_labels = labels[:, 1:].contiguous()
  vocab = shift_logits.size(-1)
  loss = F.cross_entropy(
      shift_logits.reshape(-1, vocab),
      shift_labels.reshape(-1),
      ignore_index=-100,
      reduction='mean',
  )
  return loss


def test_our_plain_ce_matches_hub_ce_reduction_single_view():
  torch.manual_seed(0)
  b, t, v, mid = 2, 8, 32, 0
  logits = torch.randn(b, t, v)
  x0 = torch.randint(1, v, (b, t))
  xt = x0.clone()
  xt[:, 1::2] = mid
  supervised = torch.ones(b, t)
  supervised[:, :2] = 0  # prompt

  hub = _hub_shifted_mask_ce_mean(
      logits, x0, xt, mask_id=mid, supervised=supervised)

  m = SimpleNamespace(
      shift_loss_targets=True,
      loss_weighting='plain_ce',
      mask_id=mid,
      _plain_ce_mask_buf=[],
  )
  m._record_plain_ce_mask = (
      lambda mp: BlockTrainer._record_plain_ce_mask(m, mp))
  alpha = torch.zeros(b, t)
  dalpha = torch.zeros(b, t)
  per_tok = BlockTrainer._masked_loss(m, logits, xt, x0, alpha, dalpha)
  # Align supervised to T-1 grid (Hub shift drops labels[0])
  vt = supervised[:, 1:]
  pcm = m._plain_ce_mask_buf[0] * vt
  ours = (per_tok * vt).sum() / pcm.sum().clamp(min=1)
  assert torch.allclose(ours, hub, rtol=1e-5, atol=1e-5), (ours.item(), hub.item())


def test_complementary_hub_mean_is_mask_union_not_doubled_valid():
  """Hub cats both views then one CE mean over all non-100 labels."""
  torch.manual_seed(1)
  b, t, v, mid = 2, 8, 32, 0
  x0 = torch.randint(1, v, (b, t))
  move = torch.zeros(b, t, dtype=torch.bool)
  move[:, 1::2] = True
  xt_a, xt_b = complementary_pair_from_mask(x0, move, mid)
  supervised = torch.ones(b, t)

  logits_a = torch.randn(b, t, v)
  logits_b = torch.randn(b, t, v)
  # Hub: one mean over concatenated batch
  hub = _hub_shifted_mask_ce_mean(
      torch.cat([logits_a, logits_b], 0),
      torch.cat([x0, x0], 0),
      torch.cat([xt_a, xt_b], 0),
      mask_id=mid,
      supervised=torch.cat([supervised, supervised], 0),
  )

  m = SimpleNamespace(
      shift_loss_targets=True,
      loss_weighting='plain_ce',
      mask_id=mid,
      complementary_masks=True,
      complementary_batching='fused',
      ignore_bos=False,
      forward_process_name='masked',
      neg_infinity=-1e6,
      vocab_size=v,
      num_tokens=t,
      block_size=4,
      block_size_mixture=[],
      block_weights=None,
      pure_noise_block_sizes=set(),
      loss_per_block_size={},
      loss_type='elbo',
      device=torch.device('cpu'),
      sampling_eps=1e-3,
      antithetic_sampling=False,
      stratified_gamma=None,
      mask_schedule='fast_dllm',
      joint_ar_alpha=0.0,
      causal_clean_stream=False,
      _pending_clean_logits=None,
      _plain_ce_mask_buf=None,
      _plain_ce_token_count=None,
      _forward_process=BlockMaskedForwardProcess.__new__(
          BlockMaskedForwardProcess),
  )

  def _corrupt(x0_, t_, *, block_size, return_move_mask=False,
               corruption_mask=None, **_k):
    del t_, block_size, corruption_mask
    xt = x0_.clone()
    xt[move] = mid
    if return_move_mask:
      return xt, move
    return xt

  def _backbone(xt, x0_, **kwargs):
    del x0_, kwargs
    # Map view by whether odd positions are masked
    if bool((xt[0, 1] == mid).item()):
      return logits_a if xt.size(0) == b else torch.cat([logits_a, logits_b], 0)
    return logits_b

  m._corrupt = _corrupt
  m._backbone_logits = _backbone
  m._sample_training_block_size = lambda *a, **k: 4
  m._process_model_input = lambda x, vt: (x, vt)
  m._masked_loss = lambda *a, **k: BlockTrainer._masked_loss(m, *a, **k)
  m._loss_for_block = lambda *a, **k: BlockTrainer._loss_for_block(m, *a, **k)
  m._apply_ignore_bos_mask = (
      lambda *a, **k: BlockTrainer._apply_ignore_bos_mask(m, *a, **k))
  m._elbo_schedule_weights = lambda t_: (
      torch.zeros_like(t_), torch.zeros_like(t_))
  m._record_plain_ce_mask = (
      lambda mp: BlockTrainer._record_plain_ce_mask(m, mp))
  m.nll = lambda *a, **k: BlockTrainer.nll(m, *a, **k)
  m._loss = lambda *a, **k: BlockTrainer._loss(m, *a, **k)

  # Force complementary path to use our fixed move mask via corrupt
  out = BlockTrainer._loss(m, x0, supervised)
  assert torch.allclose(out.loss, hub, rtol=1e-4, atol=1e-4), (
      out.loss.item(), hub.item(), out.num_tokens.item())
