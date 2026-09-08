"""Dependency-light helpers for the block-Qwen lm-eval adapter."""

from __future__ import annotations


def generation_request_args(request, default_max_tokens: int):
  """Resolve lm-eval ``generate_until`` kwargs without losing task stops."""
  kwargs = request.args[1] if len(request.args) > 1 else {}
  if kwargs is None:
    kwargs = {}
  if not isinstance(kwargs, dict):
    raise TypeError(f'generation kwargs must be a dict, got {type(kwargs)!r}')
  max_tokens = default_max_tokens
  for name in ('max_gen_toks', 'max_new_tokens', 'max_tokens',
               'max_completion_tokens'):
    if name in kwargs:
      max_tokens = kwargs[name]
      break
  max_tokens = int(max_tokens)
  if max_tokens < 1:
    raise ValueError(f'generation token limit must be positive, got {max_tokens}')
  until = kwargs.get('until') or []
  if isinstance(until, str):
    until = [until]
  return max_tokens, [str(stop) for stop in until if str(stop)]


def encode_context_continuation(tokenizer, context: str, continuation: str):
  """Tokenize an lm-eval pair without losing a whitespace-prefixed target.

  Also handles BPE merges across the context/continuation boundary (e.g.
  ``'The'+'re'`` → one token). When the joint encoding is not a strict
  prefix of ``context`` tokens, fall back to encoding the continuation
  alone so scoring still has a non-empty target while keeping the
  standalone context ids.
  """
  trailing_spaces = len(context) - len(context.rstrip())
  if trailing_spaces:
    continuation = context[-trailing_spaces:] + continuation
    context = context[:-trailing_spaces]
  whole = tokenizer(context + continuation, add_special_tokens=False)[
      'input_ids']
  context_ids = tokenizer(context, add_special_tokens=False)['input_ids']
  target_ids = whole[len(context_ids):]
  # BPE merge: joint encoding may not extend context_ids as a prefix.
  if (not target_ids or whole[:len(context_ids)] != context_ids) and continuation:
    target_ids = tokenizer(continuation, add_special_tokens=False)['input_ids']
  if not context_ids:
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_id is None:
      bos_id = getattr(tokenizer, 'eos_token_id', None)
    if bos_id is None:
      raise ValueError('tokenizer needs BOS or EOS for an empty context')
    context_ids = [int(bos_id)]
  return context_ids, target_ids


def truncate_at_stops(text: str, stops: list[str]) -> str:
  hits = [text.find(stop) for stop in stops if stop and stop in text]
  return text[:min(hits)] if hits else text


def require_masked_likelihood(forward_process_name: str) -> None:
  if forward_process_name != 'masked':
    raise NotImplementedError(
        'block_qwen likelihood scoring is a masked-corruption heuristic and '
        'is invalid for the uniform arm; use generative tasks or a '
        'principled conditional uniform bound')
