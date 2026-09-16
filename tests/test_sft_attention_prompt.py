"""Regression: SFT attention must see the prompt; loss uses assistant labels."""

from types import SimpleNamespace

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer


def test_batch_valid_tokens_is_assistant_not_padding():
  batch = {
      'attention_mask': torch.tensor([[1, 1, 1, 1, 0]]),
      'labels': torch.tensor([[-100, -100, 5, 6, -100]]),
  }
  valid = BlockTrainer._batch_valid_tokens(batch)
  assert valid.tolist() == [[0, 0, 1, 1, 0]]


def test_nll_uses_padding_attention_not_assistant_mask():
  """Default path: real padding mask (prompt visible; pads blocked)."""
  captured = {}

  def _backbone(xt, x0, *, block_size=None, attention_mask=None, **kwargs):
    del x0, block_size, kwargs
    captured['attention_mask'] = (
        None if attention_mask is None else attention_mask.clone())
    b, t = xt.shape
    return torch.zeros(b, t, 32)

  m = SimpleNamespace(
      shift_loss_targets=False,
      ignore_bos=False,
      forward_process_name='masked',
      mask_id=0,
      neg_infinity=-1e6,
      vocab_size=32,
      num_tokens=8,
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
      complementary_masks=False,
      mask_schedule='alpha',
      joint_ar_alpha=0.0,
      causal_clean_stream=False,
      _pending_clean_logits=None,
      loss_weighting='elbo',
      hub_struct_attn_only=False,
      _plain_ce_mask_buf=None,
      _plain_ce_token_count=None,
      _forward_process=None,
  )

  def _corrupt(x0, t, *, block_size, corruption_mask=None, **_kwargs):
    del t, block_size
    xt = x0.clone()
    if corruption_mask is not None:
      msk = corruption_mask.bool()
      xt = torch.where(msk & (torch.arange(x0.size(-1)) % 2 == 1),
                       torch.zeros_like(x0), x0)
    return xt

  m._corrupt = _corrupt
  m._backbone_logits = _backbone
  m._sample_training_block_size = lambda *a, **k: m.block_size
  m._process_model_input = lambda x0, vt: (x0, vt)
  m._masked_loss = lambda *a, **k: torch.zeros(a[1].shape[0], a[1].shape[1])
  m._loss_for_block = lambda logits, xt, x0, *a, **k: torch.zeros_like(
      xt, dtype=torch.float32)
  m._apply_ignore_bos_mask = lambda loss, vt: (loss, vt)
  m._elbo_schedule_weights = lambda t: (
      torch.full_like(t, 0.5), torch.full_like(t, -0.5))

  b, t = 2, 8
  x0 = torch.randint(1, 32, (b, t))
  supervised = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0],
                             [0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.float32)
  padding = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)

  BlockTrainer.nll(m, x0, supervised, attention_mask=padding)
  assert torch.equal(captured['attention_mask'], padding)
  assert not torch.equal(captured['attention_mask'], supervised)


def test_hub_struct_attn_only_passes_none():
  """Hub train: structural mask only — no pad tensor into SDPA."""
  captured = {}

  def _backbone(xt, x0, *, attention_mask=None, **kwargs):
    del x0, kwargs
    captured['attention_mask'] = attention_mask
    return torch.zeros(*xt.shape, 32)

  m = SimpleNamespace(
      shift_loss_targets=False,
      ignore_bos=False,
      forward_process_name='masked',
      mask_id=0,
      vocab_size=32,
      num_tokens=8,
      block_size=4,
      block_size_mixture=[],
      block_weights=None,
      pure_noise_block_sizes=set(),
      loss_per_block_size={},
      loss_type='elbo',
      device=torch.device('cpu'),
      sampling_eps=1e-3,
      antithetic_sampling=True,  # must be ignored under fast_dllm
      stratified_gamma=None,
      complementary_masks=False,
      mask_schedule='fast_dllm',
      joint_ar_alpha=0.0,
      causal_clean_stream=False,
      _pending_clean_logits=None,
      loss_weighting='elbo',
      hub_struct_attn_only=True,
      _plain_ce_mask_buf=None,
      _plain_ce_token_count=None,
      _forward_process=None,
  )
  m._corrupt = lambda x0, t, **k: x0.clone()
  m._backbone_logits = _backbone
  m._sample_training_block_size = lambda *a, **k: 4
  m._loss_for_block = lambda logits, xt, x0, *a, **k: torch.zeros_like(
      xt, dtype=torch.float32)
  m._apply_ignore_bos_mask = lambda loss, vt: (loss, vt)
  m._elbo_schedule_weights = lambda t: (
      torch.zeros_like(t), torch.zeros_like(t))

  x0 = torch.randint(1, 32, (2, 8))
  vt = torch.ones(2, 8)
  pad = torch.ones(2, 8)
  pad[:, -1] = 0
  BlockTrainer.nll(m, x0, vt, attention_mask=pad)
  assert captured['attention_mask'] is None
