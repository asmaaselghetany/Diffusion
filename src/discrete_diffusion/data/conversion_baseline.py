"""Shared conversion-baseline hygiene (masked + uniform arms).

Not Fast-dLLM *levers* — train↔eval chat contract, Nemotron mix defaults,
eval context policy, packing fingerprint, and preprocessing version.
Both ``ar2block`` masked (C0) and uniform (B1) should import from here so
arms stay aligned on one on-disk cache.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence

# Bump when template / default splits / packing / caps / fingerprint payload
# change so the Nemotron on-disk cache rebuilds for *all* arms together.
# v3.1: fingerprint now includes chat-template body hash (M3 audit).
NEMOTRON_PREPROCESSING_VERSION = 'qwen-chat-block-aligned-v3.1-tplhash'

# Enhanced conversion baseline: chat+safety+science full; math/code raised
# well above the old 100k caps. Full Nemotron math/code are ~22M/~10M rows —
# uncapped would drown the mix and explode cache build time.
# Override with NEMOTRON_SFT_SPLITS / NEMOTRON_SFT_MAX_PER_SPLIT.
NEMOTRON_DEFAULT_SPLITS: tuple[str, ...] = (
    'chat', 'safety', 'science', 'math', 'code',
)
NEMOTRON_DEFAULT_MAX_PER_SPLIT: dict[str, int | None] = {
    'chat': None,
    'safety': None,
    'science': None,
    'math': 1_000_000,
    'code': 500_000,
}

# Short system prompt used in Fast-dLLM-style SFT (not Alibaba stock Qwen).
CONVERSION_SYSTEM_PROMPT = 'You are a helpful assistant.'

# Hub eval.py uses max_new=2048 with max_position_embeddings≈32k. Our train
# length is typically 2048; lm-eval may extend the generation buffer up to
# this cap (block-aligned) so prompts are not wiped and max_new is not
# silently clamped to ~seq_len-prefix. Override with EVAL_MAX_SEQ_LEN.
EVAL_MAX_SEQ_LEN_DEFAULT = 8192

# Qwen ChatML with assistant generation tags + add_generation_prompt support.
# Train and lm-eval / free-gen must share this (template mismatch was a
# confirmed Hub-gap wiring bug).
FAST_DLLM_SFT_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + "
    "'<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant."
    "<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}"
    "{{- message['content'] + '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + "
    "'<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'system' and not loop.first %}"
    "{{- '<|im_start|>system\\n' + message['content'] + "
    "'<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def nemotron_split_list() -> list[str]:
  """Resolved Nemotron SFT split names (env override or defaults)."""
  raw = os.environ.get('NEMOTRON_SFT_SPLITS', '').strip()
  if not raw:
    return list(NEMOTRON_DEFAULT_SPLITS)
  splits = [s.strip() for s in raw.split(',') if s.strip()]
  if not splits:
    raise ValueError('NEMOTRON_SFT_SPLITS is set but empty')
  return splits


def nemotron_max_for_split(split: str) -> int | None:
  """Resolved per-split cap (env override or defaults). ``None`` = full split."""
  raw = os.environ.get('NEMOTRON_SFT_MAX_PER_SPLIT', '').strip()
  overrides: dict[str, int | None] = {}
  if raw:
    for part in raw.split(','):
      if not part.strip():
        continue
      if '=' not in part:
        raise ValueError(
            f'NEMOTRON_SFT_MAX_PER_SPLIT entries must be split=N, got {part!r}')
      key, val = part.split('=', 1)
      key = key.strip()
      val = val.strip().lower()
      overrides[key] = None if val in {'none', 'all', ''} else int(val)
  if split in overrides:
    return overrides[split]
  return NEMOTRON_DEFAULT_MAX_PER_SPLIT.get(split)


def nemotron_resolved_caps() -> tuple[tuple[str, int | None], ...]:
  """Stable (split, cap) tuples for cache fingerprinting (not raw env text)."""
  return tuple((s, nemotron_max_for_split(s)) for s in nemotron_split_list())


def eval_max_seq_len(default: int | None = None) -> int:
  """Upper bound for lm-eval generation buffer (block-aligned by caller)."""
  raw = os.environ.get('EVAL_MAX_SEQ_LEN', '').strip()
  if raw:
    return max(1, int(raw))
  return int(default if default is not None else EVAL_MAX_SEQ_LEN_DEFAULT)


def conversion_chat_template_hash() -> str:
  """Stable short hash of the train↔eval ChatML body (fingerprint input)."""
  return hashlib.sha256(FAST_DLLM_SFT_CHAT_TEMPLATE.encode('utf-8')).hexdigest()[:16]


def tokenizer_cache_identity(tokenizer) -> tuple:
  """Tokenizer identity tuple cached on the object for fingerprint reuse."""
  cached = getattr(tokenizer, '_discrete_diffusion_cache_identity', None)
  if cached is not None:
    return cached
  vocab_hash = hashlib.sha256()
  for token, token_id in sorted(
      tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])):
    vocab_hash.update(str(token_id).encode('ascii'))
    vocab_hash.update(b'\0')
    vocab_hash.update(token.encode('utf-8'))
    vocab_hash.update(b'\0')
  cached = (
      tokenizer.__class__.__name__,
      getattr(tokenizer, 'name_or_path', None),
      getattr(tokenizer, '_commit_hash', None),
      len(tokenizer),
      tokenizer.bos_token_id,
      tokenizer.eos_token_id,
      tokenizer.pad_token_id,
      tokenizer.mask_token_id,
      vocab_hash.hexdigest(),
  )
  setattr(tokenizer, '_discrete_diffusion_cache_identity', cached)
  return cached


def nemotron_cache_fingerprint(
    tokenizer, revision, diffusion_block_size,
) -> str:
  """Single shared cache fingerprint for all conversion arms.

  Includes preprocessing version, ChatML body hash, tokenizer identity,
  dataset revision, diffusion block size, and resolved (split, cap) tuples
  — not raw env strings — so empty env and explicit matching defaults share
  one on-disk cache.
  """
  payload = repr((
      NEMOTRON_PREPROCESSING_VERSION,
      conversion_chat_template_hash(),
      tokenizer_cache_identity(tokenizer),
      revision,
      int(diffusion_block_size),
      nemotron_resolved_caps(),
  ))
  return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def conversion_baseline_manifest() -> dict[str, Any]:
  """Single source of truth for audit files / shared-cache locks."""
  caps = {s: c for s, c in nemotron_resolved_caps()}
  return {
      'preprocessing_version': NEMOTRON_PREPROCESSING_VERSION,
      'chat_template_hash': conversion_chat_template_hash(),
      'splits': list(nemotron_split_list()),
      'max_per_split': caps,
      'system_prompt': CONVERSION_SYSTEM_PROMPT,
      'eval_max_seq_len_default': EVAL_MAX_SEQ_LEN_DEFAULT,
      'packing': 'block_aligned_sft_pad_to_diffusion_block_then_pack',
  }


def install_conversion_chat_template(tokenizer) -> None:
  """Install the shared conversion ChatML on a tokenizer (train + eval)."""
  tokenizer.chat_template = FAST_DLLM_SFT_CHAT_TEMPLATE


def apply_conversion_chat_template(
    tokenizer,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    **kwargs,
):
  """``apply_chat_template`` always using the conversion baseline template."""
  return tokenizer.apply_chat_template(
      list(messages),
      chat_template=FAST_DLLM_SFT_CHAT_TEMPLATE,
      add_generation_prompt=add_generation_prompt,
      tokenize=tokenize,
      **kwargs,
  )


def conversion_prefix_text(user_prompt: str | None = None) -> str:
  """Bare assistant-turn prefix for free-gen (matches train system prompt)."""
  if user_prompt:
    return ''  # caller should use apply_conversion_chat_template
  return (
      '<|im_start|>system\n'
      f'{CONVERSION_SYSTEM_PROMPT}'
      '<|im_end|>\n'
      '<|im_start|>assistant\n'
  )


def maybe_keep_hub_vocab_size(tokenizer) -> None:
  """Hub Fast-dLLM policy: add MASK without shrinking padded HF embeddings.

  Qwen2.5 config ``vocab_size=151936`` while ``len(tokenizer)`` after MASK is
  ~151666. Hub resizes only when ``len > embed``; we annotate
  ``_effective_vocab_size`` so *new trains* keep V=151936 and never shrink.
  Checkpoint load overrides this via ``apply_checkpoint_embed_vocab_size``.
  """
  name = getattr(tokenizer, 'name_or_path', None) or ''
  if not name or name in ('text8', 'synthetic', 'bert-base-uncased'):
    return
  try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    pad_v = int(getattr(cfg, 'vocab_size', 0) or 0)
  except Exception as exc:
    # Surface the cause — silent return previously left a shrunk vocab with
    # no hint that Hub config lookup failed (offline, bad path, etc.).
    import logging
    logging.getLogger(__name__).warning(
        'Hub vocab_size lookup failed for %r (%s: %s); '
        'keeping tokenizer len=%s (may mismatch padded HF embed)',
        name, type(exc).__name__, exc, len(tokenizer))
    return
  if pad_v <= 0:
    return
  if len(tokenizer) <= pad_v:
    tokenizer._effective_vocab_size = pad_v
    tokenizer._hub_keep_vocab_size = pad_v


def checkpoint_embed_vocab_size(ckpt: dict) -> int | None:
  """Read embed row count from a Lightning ``state_dict`` (old vs Hub-pad)."""
  sd = ckpt.get('state_dict') or {}
  for key in (
      'backbone.model.model.embed_tokens.weight',
      'backbone.model.embed_tokens.weight',
      'model.model.model.embed_tokens.weight',
      'model.model.embed_tokens.weight',
  ):
    w = sd.get(key)
    if w is not None and getattr(w, 'ndim', 0) >= 1:
      return int(w.shape[0])
  return None


def apply_checkpoint_embed_vocab_size(tokenizer, embed_vocab: int) -> None:
  """Force trainer/backbone V to match a checkpoint (eval load path)."""
  v = int(embed_vocab)
  if v <= 0:
    raise ValueError(f'embed_vocab must be >0, got {embed_vocab!r}')
  tokenizer._effective_vocab_size = v
  tokenizer._hub_keep_vocab_size = v


__all__ = [
    'CONVERSION_SYSTEM_PROMPT',
    'EVAL_MAX_SEQ_LEN_DEFAULT',
    'FAST_DLLM_SFT_CHAT_TEMPLATE',
    'NEMOTRON_DEFAULT_MAX_PER_SPLIT',
    'NEMOTRON_DEFAULT_SPLITS',
    'NEMOTRON_PREPROCESSING_VERSION',
    'apply_checkpoint_embed_vocab_size',
    'apply_conversion_chat_template',
    'checkpoint_embed_vocab_size',
    'conversion_baseline_manifest',
    'conversion_chat_template_hash',
    'conversion_prefix_text',
    'eval_max_seq_len',
    'install_conversion_chat_template',
    'maybe_keep_hub_vocab_size',
    'nemotron_cache_fingerprint',
    'nemotron_max_for_split',
    'nemotron_resolved_caps',
    'nemotron_split_list',
    'tokenizer_cache_identity',
]
