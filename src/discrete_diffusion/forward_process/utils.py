"""Utilities for forward-process implementations.

Includes tokenizer helpers and a numerically stable categorical sampler.
"""

from __future__ import annotations

import torch


def _effective_vocab_size(tokenizer) -> int:
  """Return the effective vocabulary size for a tokenizer.
  
  Prefer an explicitly annotated `_effective_vocab_size`, otherwise fall back to
  the actual length (which reflects special tokens added via `add_special_tokens`).
  """
  eff = getattr(tokenizer, "_effective_vocab_size", None)
  if eff is not None:
    return int(eff)
  return int(len(tokenizer))


def _mask_token_id(tokenizer) -> int:
  """Return the tokenizer's mask token id, raising if not defined."""
  mask_id = getattr(tokenizer, "mask_token_id", None)
  if mask_id is None:
    raise ValueError("Mask token id is not defined for tokenizer")
  return int(mask_id)


def uniform_noise_exclude_ids(
    vocab_size: int,
    *,
    mask_id: int | None = None,
    pad_id: int | None = None,
    extra_ids: tuple[int, ...] | list[int] | None = None,
) -> tuple[int, ...]:
  """Ids that must never appear as Unif *noise* on the conversion stack.

  BlockGen/Duo have no MASK and typically no ChatML control tokens in the
  same role. Our Qwen Instruct conversion keeps MASK for the masked arm and
  ChatML specials for templating. Drawing them as uniform redraw noise is
  out of the ordinary:
    - MASK: reserved absorbing atom on the other arm
    - PAD / ``<|endoftext|>``: padding, not content
    - ``<|im_start|>``: structural ChatML opener (not mid-CoT content)

  ``<|im_end|>`` / EOS stays *in* the simplex so the model can still emit
  stop tokens via ``p(x0)`` (noise may rarely redraw EOS — accepted).
  """
  v = int(vocab_size)
  out: list[int] = []
  for raw in (mask_id, pad_id, *(extra_ids or ())):
    if raw is None:
      continue
    mid = int(raw)
    if 0 <= mid < v and mid not in out:
      out.append(mid)
  return tuple(out)


def chatml_struct_extra_ids(tokenizer) -> tuple[int, ...]:
  """Qwen ChatML structural ids to keep out of Unif noise (not EOS/im_end)."""
  extras: list[int] = []
  for tok in ('<|im_start|>',):
    tid = None
    convert = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if convert is not None:
      tid = convert(tok)
    if tid is None or int(tid) < 0:
      continue
    # HuggingFace returns unk id when missing — skip if it equals unk.
    unk = getattr(tokenizer, 'unk_token_id', None)
    if unk is not None and int(tid) == int(unk):
      continue
    extras.append(int(tid))
  return tuple(extras)


def unused_embed_slot_ids(
    tokenizer,
    vocab_size: int,
) -> tuple[int, ...]:
  """HF padded embed rows with no tokenizer string (Qwen V=151936, len≈151666).

  Drawing these as Unif noise injects empty ``decode([id])==''`` holes into
  generations (seen on U0 hier_ss samples). Carve them out of the simplex.
  """
  v = int(vocab_size)
  try:
    n_tok = int(len(tokenizer)) if tokenizer is not None else v
  except TypeError:
    # Unit-test mocks (SimpleNamespace) have no ``__len__``.
    return ()
  if n_tok >= v:
    return ()
  return tuple(range(n_tok, v))


def normalize_uniform_simplex_mode(mode: str | None) -> str:
  """``conversion`` = Unif(V\\E); ``blockgen`` = literal Unif(V) ablation."""
  raw = str(mode or 'conversion').strip().lower()
  if raw in ('blockgen', 'literal', 'unif_v', 'full_v', 'v'):
    return 'blockgen'
  if raw in ('conversion', 'v_eff', 'default', ''):
    return 'conversion'
  raise ValueError(
      f'uniform_simplex_mode={mode!r} not in '
      f'{{conversion, blockgen}} (aliases: literal|unif_v|full_v)')


def resolve_uniform_exclude_ids(
    tokenizer,
    *,
    mask_id: int | None,
    vocab_size: int | None = None,
    mode: str | None = 'conversion',
) -> tuple[int, ...]:
  """Full Unif-simplex exclusion (MASK+PAD+im_start+unused HF embed slots).

  EOS / ``<|im_end|>`` stay *in* the simplex so ELBO / ``p(x0)`` can emit
  stops. Use ``resolve_uniform_noise_redraw_exclude_ids`` for prior/redraw
  draws so Unif noise cannot plant early EOS that ``stop_on_eos`` treats as
  a real end-of-turn.

  ``mode=blockgen``: empty E → literal ``Unif(V)`` (BlockGen/Duo ablation on
  a vocab that may still *contain* MASK as an ordinary id).
  """
  if normalize_uniform_simplex_mode(mode) == 'blockgen':
    return ()
  v = int(vocab_size if vocab_size is not None else _effective_vocab_size(tokenizer))
  pad_id = getattr(tokenizer, 'pad_token_id', None)
  extras = list(chatml_struct_extra_ids(tokenizer))
  extras.extend(unused_embed_slot_ids(tokenizer, v))
  return uniform_noise_exclude_ids(
      v, mask_id=mask_id, pad_id=pad_id, extra_ids=extras)


def resolve_uniform_noise_redraw_exclude_ids(
    tokenizer,
    *,
    mask_id: int | None,
    vocab_size: int | None = None,
    mode: str | None = 'conversion',
) -> tuple[int, ...]:
  """Ids banned from Unif *noise redraws* / prior (simplex E plus EOS/im_end)."""
  if normalize_uniform_simplex_mode(mode) == 'blockgen':
    return ()
  v = int(vocab_size if vocab_size is not None else _effective_vocab_size(tokenizer))
  base = list(resolve_uniform_exclude_ids(
      tokenizer, mask_id=mask_id, vocab_size=v, mode='conversion'))
  extras: list[int] = []
  for tok in ('<|im_end|>',):
    convert = getattr(tokenizer, 'convert_tokens_to_ids', None)
    tid = convert(tok) if convert is not None else None
    if tid is None or int(tid) < 0:
      continue
    unk = getattr(tokenizer, 'unk_token_id', None)
    if unk is not None and int(tid) == int(unk):
      continue
    extras.append(int(tid))
  eos = getattr(tokenizer, 'eos_token_id', None)
  if eos is not None:
    extras.append(int(eos))
  return uniform_noise_exclude_ids(
      v, mask_id=mask_id, extra_ids=tuple(base) + tuple(extras))


def uniform_simplex_size(
    vocab_size: int,
    mask_id: int | None,
    exclude_ids: tuple[int, ...] | list[int] | None = None,
) -> int:
  """Effective Unif simplex size after carving reserved specials out of ``V``.

  BlockGen / Duo uniform models never add MASK, so ``V = vocab_size``.
  Our Qwen conversion always keeps MASK for the shared masked arm; the
  uniform / hybrid-uniform contract is therefore ``Unif(V \\ E)`` with
  ``V_eff = vocab_size - |E|`` where ``E`` includes MASK (+ PAD / im_start
  when passed via ``exclude_ids``).

  Explicit ``exclude_ids=()`` means full ``V`` (BlockGen ablation) — do **not**
  auto-merge ``mask_id``.
  """
  v = int(vocab_size)
  if exclude_ids is not None:
    ids = uniform_noise_exclude_ids(
        v, mask_id=None, extra_ids=exclude_ids)
  else:
    ids = uniform_noise_exclude_ids(v, mask_id=mask_id)
  n_ex = len(ids)
  if n_ex == 0:
    return max(v, 1)
  return max(v - n_ex, 1)


def sample_uniform_excluding_mask(
    shape,
    *,
    vocab_size: int,
    mask_id: int | None,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
    exclude_ids: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
  """Sample ``Unif({0..V-1} \\ E)`` for reserved specials ``E``.

  Default ``E={mask_id}`` (legacy). Pass ``exclude_ids`` / pad via
  ``uniform_noise_exclude_ids`` for Instruct-safe noise.

  Explicit ``exclude_ids=()`` with ``mask_id=None`` → literal ``Unif(V)``
  (BlockGen ablation). Explicit non-empty ``exclude_ids`` is the set E and
  does **not** auto-merge ``mask_id`` (callers that want MASK out must
  include it, as ``resolve_uniform_exclude_ids`` does).
  """
  v = int(vocab_size)
  if exclude_ids is not None:
    ids = uniform_noise_exclude_ids(
        v, mask_id=None, extra_ids=exclude_ids)
    if not ids:
      return torch.randint(0, max(v, 1), shape, device=device, dtype=dtype)
  elif mask_id is None or not (0 <= int(mask_id) < v) or v <= 1:
    return torch.randint(0, max(v, 1), shape, device=device, dtype=dtype)
  else:
    ids = uniform_noise_exclude_ids(v, mask_id=mask_id)
  holes = sorted(ids)
  n_eff = v - len(holes)
  if n_eff <= 0:
    raise ValueError(
        f'Unif exclude set empties the simplex: V={v} E={holes[:8]}…')
  if len(holes) == 1:
    mid = holes[0]
    u = torch.randint(0, v - 1, shape, device=device, dtype=dtype)
    return torch.where(u >= mid, u + 1, u)
  # Peel trailing contiguous unused-embed pad ``[trail, V)`` (Qwen Hub).
  # Only bump past the few low specials (MASK/PAD/im_start).
  trail = v
  i = len(holes) - 1
  while i >= 0 and holes[i] == trail - 1:
    trail = holes[i]
    i -= 1
  low = holes[: i + 1]
  if trail < v:
    n_content = trail - len(low)
    if n_content <= 0:
      raise ValueError(
          f'Unif exclude empties content prefix: trail={trail} low={low}')
    if not low:
      return torch.randint(0, trail, shape, device=device, dtype=dtype)
    u = torch.randint(0, n_content, shape, device=device, dtype=dtype)
    out = u.clone()
    for h in low:
      out = torch.where(out >= h, out + 1, out)
    return out
  # General: draw from V-|E| then map through sorted holes.
  u = torch.randint(0, n_eff, shape, device=device, dtype=dtype)
  out = u.clone()
  for h in holes:
    out = torch.where(out >= h, out + 1, out)
  return out


def _unsqueeze(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
  """Match `x` rank to `reference` by appending singleton dims."""
  return x.view(*x.shape, * ((1,) * (len(reference.shape) - len(x.shape))))


def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
  """Sample categories via a Gumbel-max formulation for stability.

  Expects `categorical_probs` to be non-negative and to sum to one along the
  last dimension. This implementation mirrors the stable sampler used in the
  existing absorbing helpers for consistency.
  """
  gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)
