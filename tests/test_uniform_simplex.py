"""Uniform simplex helpers — MASK/PAD/im_start-safe V_eff vs BlockGen Unif(V)."""

from __future__ import annotations

import torch

from discrete_diffusion.forward_process.utils import (
    sample_uniform_excluding_mask,
    uniform_noise_exclude_ids,
    uniform_simplex_size,
)


def test_uniform_simplex_size_drops_mask():
  assert uniform_simplex_size(100, 50) == 99
  assert uniform_simplex_size(100, None) == 100
  assert uniform_simplex_size(100, 100) == 100  # out of range → full V
  assert uniform_simplex_size(1, 0) == 1


def test_uniform_simplex_size_multi_exclude():
  # MASK + PAD + im_start
  assert uniform_simplex_size(100, 50, exclude_ids=(50, 1, 2)) == 97


def test_sample_uniform_excluding_mask_never_hits_mask():
  mid = 7
  v = 32
  out = sample_uniform_excluding_mask(
      (4096,), vocab_size=v, mask_id=mid,
      device=torch.device('cpu'), dtype=torch.int64)
  assert out.min() >= 0 and out.max() < v
  assert not bool((out == mid).any())
  # Index-shift covers both sides of MASK.
  assert bool((out < mid).any())
  assert bool((out > mid).any())


def test_sample_uniform_multi_exclude_never_hits_holes():
  holes = (3, 10, 20)
  v = 32
  out = sample_uniform_excluding_mask(
      (8192,), vocab_size=v, mask_id=holes[0],
      device=torch.device('cpu'), dtype=torch.int64,
      exclude_ids=holes)
  assert out.min() >= 0 and out.max() < v
  for h in holes:
    assert not bool((out == h).any()), h
  # All remaining ids appear with enough draws.
  assert len(set(out.tolist())) == v - len(holes)


def test_sample_uniform_excluding_mask_no_mask_is_full_v():
  out = sample_uniform_excluding_mask(
      (2048,), vocab_size=16, mask_id=None,
      device=torch.device('cpu'), dtype=torch.int64)
  assert out.min() >= 0 and out.max() < 16


def test_uniform_noise_exclude_ids_dedup():
  ids = uniform_noise_exclude_ids(
      100, mask_id=5, pad_id=5, extra_ids=(5, 7))
  assert ids == (5, 7)


def test_unused_embed_slot_ids():
  from discrete_diffusion.forward_process.utils import unused_embed_slot_ids

  class _Tok:
    def __len__(self):
      return 10

  assert unused_embed_slot_ids(_Tok(), 10) == ()
  assert unused_embed_slot_ids(_Tok(), 13) == (10, 11, 12)


def test_sample_uniform_trailing_holes_fast_path():
  """Qwen-style: content prefix + trailing unused embed slots."""
  v, n_tok = 64, 50
  holes = tuple(range(n_tok, v))
  out = sample_uniform_excluding_mask(
      (8192,), vocab_size=v, mask_id=None,
      device=torch.device('cpu'), dtype=torch.int64,
      exclude_ids=holes)
  assert int(out.min()) >= 0 and int(out.max()) < n_tok
  for h in holes:
    assert not bool((out == h).any())


def test_resolve_excludes_unused_embed_slots():
  from discrete_diffusion.forward_process.utils import resolve_uniform_exclude_ids

  class _Tok:
    pad_token_id = 1
    unk_token_id = 0

    def __len__(self):
      return 20

    def convert_tokens_to_ids(self, s):
      return {'<|im_start|>': 3}.get(s, 0)

  ids = resolve_uniform_exclude_ids(_Tok(), mask_id=5, vocab_size=24)
  assert 5 in ids and 1 in ids and 3 in ids
  assert set(range(20, 24)).issubset(ids)


def test_noise_redraw_excludes_eos_but_simplex_keeps_it():
  from discrete_diffusion.forward_process.utils import (
      resolve_uniform_exclude_ids,
      resolve_uniform_noise_redraw_exclude_ids,
  )

  class _Tok:
    pad_token_id = 1
    eos_token_id = 9
    unk_token_id = 0

    def __len__(self):
      return 20

    def convert_tokens_to_ids(self, s):
      return {'<|im_start|>': 3, '<|im_end|>': 9}.get(s, 0)

  simplex = resolve_uniform_exclude_ids(_Tok(), mask_id=5, vocab_size=20)
  redraw = resolve_uniform_noise_redraw_exclude_ids(
      _Tok(), mask_id=5, vocab_size=20)
  assert 9 not in simplex
  assert 9 in redraw
  assert set(simplex).issubset(redraw)


def test_blockgen_mode_empty_exclude_and_full_v_sample():
  from discrete_diffusion.forward_process.utils import (
      normalize_uniform_simplex_mode,
      resolve_uniform_exclude_ids,
      resolve_uniform_noise_redraw_exclude_ids,
  )

  assert normalize_uniform_simplex_mode('blockgen') == 'blockgen'
  assert normalize_uniform_simplex_mode('unif_v') == 'blockgen'
  assert normalize_uniform_simplex_mode('conversion') == 'conversion'

  class _Tok:
    pad_token_id = 1
    eos_token_id = 9
    unk_token_id = 0

    def __len__(self):
      return 20

    def convert_tokens_to_ids(self, s):
      return {'<|im_start|>': 3, '<|im_end|>': 9}.get(s, 0)

  assert resolve_uniform_exclude_ids(
      _Tok(), mask_id=5, vocab_size=20, mode='blockgen') == ()
  assert resolve_uniform_noise_redraw_exclude_ids(
      _Tok(), mask_id=5, vocab_size=20, mode='blockgen') == ()
  assert uniform_simplex_size(20, 5, exclude_ids=()) == 20

  v, mid = 32, 7
  out = sample_uniform_excluding_mask(
      (8192,), vocab_size=v, mask_id=None,
      device=torch.device('cpu'), dtype=torch.int64,
      exclude_ids=())
  assert int(out.min()) >= 0 and int(out.max()) < v
  # Full Unif(V) must be allowed to hit MASK (ordinary id in ablation).
  assert bool((out == mid).any())
  assert len(set(out.tolist())) == v
