"""Shared helpers for block-diffusion decode traces → highlight videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# Gold intensity ladder (Set Diffusion–style legend).
GOLD_RGB = [
    (255, 236, 150),  # brightest — newly committed / active
    (255, 220, 100),
    (245, 195, 60),
    (230, 170, 40),   # dimmest pending
]
MASK_PLACEHOLDER = '░'
SPECIAL_SKIP = {
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '[PAD]', '[MASK]', '<pad>', '<mask>',
}


def load_trace(path: Path) -> dict[str, Any]:
  path = Path(path)
  if path.is_dir():
    meta = path / 'meta.json'
    events = path / 'events.jsonl'
  else:
    meta = path
    events = path.with_name('events.jsonl')
    if not events.is_file() and path.suffix == '.json':
      # allow meta.json path
      events = path.parent / 'events.jsonl'
  data = json.loads(Path(meta).read_text())
  steps = []
  with Path(events).open() as f:
    for line in f:
      line = line.strip()
      if line:
        steps.append(json.loads(line))
  data['steps'] = steps
  data['_dir'] = str(Path(meta).parent)
  return data


def ids_to_pieces(
    tokenizer,
    token_ids: Sequence[int],
    *,
    mask_id: int | None,
    prefix_len: int,
    token_strs: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
  """Per-token display pieces with roles for coloring."""
  pieces = []
  for i, tid in enumerate(token_ids):
    tid = int(tid)
    if token_strs is not None and i < len(token_strs):
      text = token_strs[i]
      if text == MASK_PLACEHOLDER or (mask_id is not None and tid == mask_id):
        role = 'mask'
      elif text in SPECIAL_SKIP:
        pieces.append({'i': i, 'text': '', 'role': 'special', 'tid': tid})
        continue
      else:
        role = 'prefix' if i < prefix_len else 'token'
      pieces.append({'i': i, 'text': text, 'role': role, 'tid': tid})
      continue
    if mask_id is not None and tid == mask_id:
      text = MASK_PLACEHOLDER
      role = 'mask'
    else:
      text = tokenizer.decode([tid], skip_special_tokens=False)
      if text in SPECIAL_SKIP:
        pieces.append({'i': i, 'text': '', 'role': 'special', 'tid': tid})
        continue
      role = 'prefix' if i < prefix_len else 'token'
    pieces.append({'i': i, 'text': text, 'role': role, 'tid': tid})
  return pieces


def annotate_intensities(
    pieces: list[dict[str, Any]],
    step: dict[str, Any],
) -> list[dict[str, Any]]:
  """Attach gold intensity 0..3 (0=brightest) or None."""
  newly = set(step.get('newly_committed_positions') or [])
  changed = set(step.get('changed_positions') or [])
  still = set(step.get('still_masked_positions') or [])
  active = set(step.get('active_positions') or [])
  w0 = int(step.get('window_start', 0))
  w1 = int(step.get('window_end', 0))
  out = []
  for p in pieces:
    i = p['i']
    inten = None
    if i in newly or i in changed:
      inten = 0
    elif i in still:
      inten = 2
    elif w0 <= i < w1 and i in active and p.get('role') != 'mask':
      inten = 1
    elif w0 <= i < w1 and p.get('role') == 'mask':
      inten = 3
    q = dict(p)
    q['intensity'] = inten
    q['in_window'] = w0 <= i < w1
    out.append(q)
  return out


def visible_span(
    pieces: Sequence[dict[str, Any]],
    *,
    prefix_len: int,
    max_gen_chars: int = 900,
) -> list[dict[str, Any]]:
  """Keep prompt tail + generated text; drop leading specials-only noise."""
  # find last non-empty prefix piece
  start = 0
  for i, p in enumerate(pieces):
    if i >= prefix_len:
      start = max(0, i - 40)  # small prompt context
      break
  # trim trailing pure masks beyond last real token
  end = len(pieces)
  for i in range(len(pieces) - 1, -1, -1):
    if pieces[i].get('role') != 'mask' and pieces[i].get('text'):
      end = min(len(pieces), i + 8)
      break
  selected = list(pieces[start:end])
  # char budget on generation side
  chars = 0
  kept = []
  for p in selected:
    t = p.get('text') or ''
    if p['i'] >= prefix_len:
      if chars > max_gen_chars and p.get('role') == 'mask':
        continue
      chars += len(t)
    kept.append(p)
  return kept


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w') as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')
