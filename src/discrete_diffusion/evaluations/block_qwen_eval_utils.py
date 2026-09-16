"""Dependency-light helpers for the block-Qwen lm-eval adapter."""

from __future__ import annotations

import re

# lm-eval HumanEval defaults (completion / mid-function stops). With
# ``--apply_chat_template``, instruct models rewrite ``def …`` inside a
# markdown fence; the first ``\ndef`` then truncates the sample to
# `````python`` and pass@1 collapses to ~0.
_HUMANEVAL_COMPLETION_STOPS = frozenset({
    '\nclass',
    '\ndef',
    '\n#',
    '\nif',
    '\nprint',
})


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

  Only ASCII spaces are moved onto the continuation — not newlines.
  ChatML generation prompts end with ``\\n``; ``str.rstrip()`` would steal
  that newline into every MC choice so the first target token is shared
  (``\\n``) and one-token MMLU LL collapses to chance (~25%).
  """
  trailing_spaces = len(context) - len(context.rstrip(' '))
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


def humaneval_until_for_chat(until: list[str]) -> list[str]:
  """Drop completion-style stops that kill chat/markdown HumanEval rewrites."""
  kept = [s for s in until if s not in _HUMANEVAL_COMPLETION_STOPS]
  for stop in ('<|im_end|>', '<|endoftext|>'):
    if stop not in kept:
      kept.append(stop)
  return kept or ['<|im_end|>']


def strip_think_blocks(text: str) -> str:
  """Remove ``<think>…</think>`` (and unclosed) reasoning wrappers."""
  text = re.sub(
      r'<think>\s*.*?</think\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
  text = re.sub(r'<think>\s*.*\Z', '', text, flags=re.IGNORECASE | re.DOTALL)
  # Do not ``.strip()`` — HumanEval body completions are indent-sensitive.
  return text


def extract_fenced_code(text: str) -> str | None:
  """Return the last markdown fenced code block, preferring ``python``."""
  if not text:
    return None
  blocks = re.findall(
      r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.IGNORECASE | re.DOTALL)
  if blocks:
    return blocks[-1].strip('\n')
  # Opening fence without close (truncated gen): take remainder.
  m = re.search(r'```(?:python|py)?\s*\n(.*)\Z', text, flags=re.IGNORECASE | re.DOTALL)
  if m and m.group(1).strip():
    return m.group(1).strip('\n')
  return None


def prepare_code_completion(text: str, task_name: str) -> str:
  """Normalize chat-style code completions before lm-eval filters/metrics."""
  raw = strip_think_blocks(text or '')
  name = (task_name or '').lower()
  fenced = extract_fenced_code(raw)
  if name.startswith('humaneval'):
    # Keep full rewrite (incl. ``def``) for chat-aware ``build_predictions``.
    return fenced if fenced is not None else raw
  if name.startswith('mbpp'):
    code = fenced if fenced is not None else raw
    code = code.replace('[BEGIN]', '').replace('[DONE]', '').strip()
    return code
  return raw


def assemble_humaneval_prediction(prompt: str, completion: str, entry_point: str) -> str:
  """Build an executable HumanEval program from prompt + chat/completion text.

  Completion models return a body that must be concatenated with ``prompt``.
  Chat models often regenerate ``def {entry_point}…``; concatenating then
  yields a duplicate ``def`` / syntax error — use the rewritten function and
  keep any import preamble from ``prompt``.
  """
  code = prepare_code_completion(completion, 'humaneval')
  entry = (entry_point or '').strip()
  if entry:
    match = re.search(
        rf'(?m)^[ \t]*def[ \t]+{re.escape(entry)}\s*\(', code)
    if match is not None:
      fn = code[match.start():]
      preamble = ''
      if 'def ' in prompt:
        preamble = prompt[:prompt.index('def ')]
      return preamble + fn
  return prompt + code


def require_masked_likelihood(forward_process_name: str) -> None:
  if forward_process_name != 'masked':
    raise NotImplementedError(
        'block_qwen likelihood scoring is a masked-corruption heuristic and '
        'is invalid for the uniform arm; use generative tasks or a '
        'principled conditional uniform bound')
