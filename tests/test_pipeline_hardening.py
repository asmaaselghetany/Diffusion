"""Regression tests for conversion fingerprint / export / seq-extend hardening."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from discrete_diffusion.data.conversion_baseline import (
    NEMOTRON_PREPROCESSING_VERSION,
    conversion_chat_template_hash,
    nemotron_cache_fingerprint,
)
from discrete_diffusion.data import loaders as loaders_mod
from tools.export_block_ckpt_to_fastdllm_hf import _backbone_export_tensors


class _TinyTok:
  __class__ = type('Tok', (), {})
  name_or_path = 'tiny'
  _commit_hash = None
  bos_token_id = 1
  eos_token_id = 2
  pad_token_id = 3
  mask_token_id = 4

  def get_vocab(self):
    return {'a': 0, 'b': 1}

  def __len__(self):
    return 2


def test_fingerprint_lives_in_conversion_baseline_only(monkeypatch):
  monkeypatch.delenv('NEMOTRON_SFT_SPLITS', raising=False)
  monkeypatch.delenv('NEMOTRON_SFT_MAX_PER_SPLIT', raising=False)
  tok = _TinyTok()
  fp = nemotron_cache_fingerprint(tok, None, 32)
  assert len(fp) == 12
  # loaders must re-export the same function (no duplicate body).
  assert loaders_mod._nemotron_cache_fingerprint is nemotron_cache_fingerprint
  assert fp == loaders_mod._nemotron_cache_fingerprint(tok, None, 32)
  assert 'tplhash' in NEMOTRON_PREPROCESSING_VERSION
  assert len(conversion_chat_template_hash()) == 16


def test_fingerprint_changes_when_template_hash_input_changes(monkeypatch):
  monkeypatch.delenv('NEMOTRON_SFT_SPLITS', raising=False)
  monkeypatch.delenv('NEMOTRON_SFT_MAX_PER_SPLIT', raising=False)
  tok = _TinyTok()
  fp1 = nemotron_cache_fingerprint(tok, None, 32)
  import discrete_diffusion.data.conversion_baseline as cb
  monkeypatch.setattr(
      cb, 'FAST_DLLM_SFT_CHAT_TEMPLATE', cb.FAST_DLLM_SFT_CHAT_TEMPLATE + ' ')
  # Clear cached hash path by calling fingerprint after template change —
  # conversion_chat_template_hash reads the module attribute.
  fp2 = nemotron_cache_fingerprint(tok, None, 32)
  assert fp1 != fp2


def test_export_ema_shape_assert_and_noise_refuse(monkeypatch):
  import tools.export_block_ckpt_to_fastdllm_hf as exp
  monkeypatch.setattr(exp, 'HUB_VOCAB', 8)
  # Two tied-style tensors: embed + norm (no lm_head in export keys).
  embed = torch.randn(8, 4)
  norm = torch.randn(4)
  sd = {
      'backbone.model.model.embed_tokens.weight': embed,
      'backbone.model.model.norm.weight': norm,
      'backbone.model.lm_head.weight': embed,  # tied; excluded from export
  }
  shadows = [embed + 0.1, norm + 0.1]
  ckpt = {'state_dict': sd, 'ema': {'shadow_params': shadows}}
  out = _backbone_export_tensors(ckpt, use_ema=True, allow_embed_pad=False)
  assert set(out) == {
      'model.embed_tokens.weight', 'model.norm.weight'}
  assert out['model.embed_tokens.weight'].shape == (8, 4)

  bad = {
      'state_dict': sd,
      'ema': {'shadow_params': [torch.randn(3, 4), norm]},
  }
  with pytest.raises(ValueError, match='EMA shadow'):
    _backbone_export_tensors(bad, use_ema=True, allow_embed_pad=False)

  noisy = {
      'state_dict': {**sd, 'noise.foo': torch.randn(2)},
      'ema': {'shadow_params': shadows},
  }
  with pytest.raises(ValueError, match='noise'):
    _backbone_export_tensors(noisy, use_ema=True, allow_embed_pad=False)

  small = {
      'state_dict': {
          'backbone.model.model.embed_tokens.weight': torch.randn(4, 4),
          'backbone.model.model.norm.weight': norm,
      },
      'ema': {'shadow_params': [torch.randn(4, 4), norm]},
  }
  with pytest.raises(ValueError, match='refuse silent'):
    _backbone_export_tensors(small, use_ema=True, allow_embed_pad=False)


def test_seq_extend_restores_num_tokens_and_backbone_n_tokens():
  """Exercise the real ``temporary_model_seq_len`` try/finally contract."""
  from discrete_diffusion.evaluations.block_qwen_eval_utils import (
      temporary_model_seq_len,
  )

  backbone = SimpleNamespace(n_tokens=2048)
  model = SimpleNamespace(num_tokens=2048, backbone=backbone)
  holder = SimpleNamespace(seq_len=2048)

  with temporary_model_seq_len(model, 4096, seq_len_holder=holder) as ctx:
    assert ctx.extended
    assert model.num_tokens == 4096
    assert backbone.n_tokens == 4096
    assert holder.seq_len == 4096
  assert model.num_tokens == 2048
  assert backbone.n_tokens == 2048
  assert holder.seq_len == 2048

  # Restore must run even when the body raises.
  try:
    with temporary_model_seq_len(model, 8192, seq_len_holder=holder):
      assert model.num_tokens == 8192
      raise RuntimeError('boom')
  except RuntimeError:
    pass
  assert model.num_tokens == 2048
  assert backbone.n_tokens == 2048
  assert holder.seq_len == 2048
