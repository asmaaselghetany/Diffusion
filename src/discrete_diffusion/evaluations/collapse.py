"""Detect repetitive / mode-collapsed generated text."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def collapse_stats(text: str) -> dict[str, Any]:
  """Heuristic stats for loop / mode-collapse in a decoded sample."""
  text = text or ''
  toks = text.split()
  n = len(toks)
  if n == 0:
    return {
        'n_toks': 0,
        'uniq_ratio': 0.0,
        'top_frac': 1.0,
        'top_tok': '',
        'char_loop': False,
        'collapsed': True,
        'reason': 'empty',
    }

  counts = Counter(toks)
  top_tok, top_n = counts.most_common(1)[0]
  uniq_ratio = len(counts) / n
  top_frac = top_n / n

  chars = re.sub(r'\s+', '', text)
  char_loop = False
  if len(chars) >= 40:
    gram = chars[:4]
    char_loop = chars.count(gram) >= max(8, len(chars) // (len(gram) * 3))

  reason = None
  if uniq_ratio < 0.05:
    reason = 'low_uniq_ratio'
  elif top_frac > 0.5:
    reason = 'high_top_frac'
  elif char_loop:
    reason = 'char_loop'
  elif len(text.strip()) < 20:
    reason = 'too_short'

  return {
      'n_toks': n,
      'uniq_ratio': round(uniq_ratio, 4),
      'top_frac': round(top_frac, 4),
      'top_tok': top_tok[:40],
      'char_loop': char_loop,
      'collapsed': reason is not None,
      'reason': reason,
      'preview': text[:160].replace('\n', ' '),
  }


def samples_collapsed(
    texts: list[str],
    *,
    uniq_ratio_max: float = 0.05,
    top_frac_min: float = 0.5,
    min_fraction: float = 0.5,
) -> tuple[bool, list[dict[str, Any]]]:
  """Return whether a batch of texts looks mode-collapsed.

  A sample is collapsed if uniq_ratio < uniq_ratio_max OR top_frac > top_frac_min
  OR the char-loop / empty heuristics fire. The batch is collapsed if at least
  ``min_fraction`` of samples are collapsed.
  """
  stats = []
  for t in texts:
    s = collapse_stats(t)
    # Allow callers to tighten/loosen via thresholds (override defaults in stats).
    collapsed = (
        s['n_toks'] == 0
        or s['uniq_ratio'] < uniq_ratio_max
        or s['top_frac'] > top_frac_min
        or s['char_loop']
        or len((t or '').strip()) < 20
    )
    s['collapsed'] = collapsed
    if collapsed and not s.get('reason'):
      s['reason'] = 'threshold'
    stats.append(s)

  n = len(stats)
  if n == 0:
    return True, stats
  n_bad = sum(1 for s in stats if s['collapsed'])
  return (n_bad / n) >= min_fraction, stats
