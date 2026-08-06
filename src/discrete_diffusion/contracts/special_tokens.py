"""Layer 0 — single source of truth for mask / pad / eos ids.

Every training and eval path that touches special tokens should resolve
them through ``ensure_special_tokens`` (or ``ensure_mask_token``, which
delegates here) and then ``assert_same_mask_id`` when a second object
also holds a mask id (forward process, sampler, harness encode).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..forward_process.utils import _effective_vocab_size


@dataclass(frozen=True)
class SpecialTokenIds:
  """Resolved once at load; compare by value across code paths."""

  mask_id: int
  pad_id: int | None
  eos_id: int | None
  vocab_size: int


def ensure_special_tokens(tokenizer) -> SpecialTokenIds:
  """Resolve mask/pad/eos and ensure ``tokenizer.mask_token_id`` is set.

  If the tokenizer has no mask token, allocate ``mask_id = effective_vocab``
  and grow vocab by one (same behaviour as historical ``ensure_mask_token``).
  """
  vocab_size = _effective_vocab_size(tokenizer)
  if getattr(tokenizer, 'mask_token', None) is None:
    mask_id = vocab_size
    vocab_size += 1
  else:
    mid = getattr(tokenizer, 'mask_token_id', None)
    if mid is None:
      raise ValueError(
          'tokenizer.mask_token is set but mask_token_id is None — '
          'refuse silent divergence (L0 contract)')
    mask_id = int(mid)
  if getattr(tokenizer, 'mask_token_id', None) is None:
    setattr(tokenizer, 'mask_token_id', int(mask_id))

  pad_id = getattr(tokenizer, 'pad_token_id', None)
  eos_id = getattr(tokenizer, 'eos_token_id', None)
  return SpecialTokenIds(
      mask_id=int(mask_id),
      pad_id=int(pad_id) if pad_id is not None else None,
      eos_id=int(eos_id) if eos_id is not None else None,
      vocab_size=int(vocab_size),
  )


def assert_same_mask_id(expected: int, actual: int, *, where: str) -> None:
  """Hard-fail if two paths disagree on mask id."""
  if int(expected) != int(actual):
    raise AssertionError(
        f'L0 mask_id mismatch at {where}: '
        f'expected={int(expected)} actual={int(actual)}')
