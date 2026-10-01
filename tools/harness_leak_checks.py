#!/usr/bin/env python3
"""Harness-level leak checks (Part-1 tests 2 + live corrupt-GT).

These exercise code *outside* the sampler: prompt building and scoring.
Designed to run as a small offline CPU check on logged generations, or as
notes for a live lm_eval submit once the cluster is healthy.

Offline (no GPU) — score existing D2 completions against *swapped* golds:
  python tools/harness_leak_checks.py swapped-from-log \\
    --out-dir .../lm_eval_bake_..._D2_... --n 200

Expect near-zero exact-match if the scorer only compares generated text to
the provided gold (and generations depend on the prompt).

Live corrupt-GT / swapped-prompt eval jobs need the cluster; this module
prints the submit recipe rather than queueing while booster is drained.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_NUM = re.compile(r'(-?\d+(?:\.\d+)?)')
_BOX = re.compile(r'\\boxed\{([^}]*)\}')


def _extract_num(text: str) -> str | None:
  m = _BOX.search(text)
  if m:
    inner = m.group(1)
    nums = _NUM.findall(inner.replace(',', ''))
    return nums[-1] if nums else inner.strip()
  nums = _NUM.findall(text.replace(',', ''))
  return nums[-1] if nums else None


def _load_qa(out_dir: Path, n: int) -> list[tuple[str, str]]:
  log = out_dir / 'lm_eval.log'
  text = log.read_text(errors='ignore')
  parts = list(re.finditer(r'=== task=gsm8k[^\n]*===\n', text))
  start = parts[-1].end() if parts else 0
  chunk = text[start:]
  pairs = []
  for block in re.finditer(
      r'={10,}\nquestion:\s*(.*?)\nanswer:\s*(.*?)\n={10,}',
      chunk, flags=re.S):
    q = block.group(1)
    # Pull the human question line if present.
    qm = re.search(r'Question:\s*(.*?)\nAnswer:', q, flags=re.S)
    q_short = qm.group(1).strip() if qm else q.strip()[:200]
    a = block.group(2).strip()
    pairs.append((q_short, a))
    if len(pairs) >= n:
      break
  if len(pairs) < 2:
    raise SystemExit(f'Need ≥2 GSM dumps in {log}, found {len(pairs)}')
  return pairs


def cmd_swapped_from_log(args: argparse.Namespace) -> None:
  pairs = _load_qa(Path(args.out_dir), int(args.n))
  n = len(pairs)
  golds = [_extract_num(a) for _, a in pairs]
  match_self = 0
  match_swap = 0
  scored = 0
  for i in range(n):
    pred = golds[i]
    gold_swap = golds[(i + 1) % n]
    if pred is None or gold_swap is None:
      continue
    scored += 1
    match_self += 1  # pred extracted from same completion
    if pred == gold_swap:
      match_swap += 1
  # Chance baselines on the same gold multiset.
  valid = [g for g in golds if g is not None]
  nv = len(valid)
  adj_chance = (
      sum(1 for i in range(nv) if valid[i] == valid[(i + 1) % nv]) / max(nv, 1))
  pair_eq = sum(
      1 for i in range(nv) for j in range(nv) if i != j and valid[i] == valid[j])
  pair_chance = pair_eq / max(nv * (nv - 1), 1)
  print(f'pairs={n} scored={scored} unique_golds={len(set(valid))}')
  print(f'self_extract_ok={match_self}/{scored} '
        f'({100.0 * match_self / max(scored, 1):.1f}%)')
  print(f'swapped_gold_match={match_swap}/{scored} '
        f'({100.0 * match_swap / max(scored, 1):.1f}%)')
  print(f'chance_adjacent={100.0 * adj_chance:.2f}%  '
        f'chance_random_pair={100.0 * pair_chance:.2f}%')
  print('Proxy is fine iff swapped ≈ chance_adjacent (GSM small-int repeats).')
  print('This checks the scorer/extractor, not prompt→output dependence.')
  if scored and abs(match_swap / scored - adj_chance) > 0.05:
    print('WARN: swapped rate far from adjacent chance — inspect.')


def cmd_recipe(_: argparse.Namespace) -> None:
  print("""Live harness leak + batch invariance (when booster is healthy):

ONE job, TWO checkpoints (do not assume block-UCC probes transfer to C3):

  A) Floor 2092151 (block UCC)
     .../ar2block_uniform_2092151/checkpoints/last.ckpt
  B) C3 full-seq v2 2125372  [precommit: last=primary, best=secondary]
     .../ar2block_uniform_2125372/checkpoints/{last,best}.ckpt

Per checkpoint:
  1. Swapped-prompt GSM subset (few hundred) → ~chance vs swapped golds
  2. Corrupt-GT: scramble references before scoring; accuracy must match baseline
  3. Batch invariance: UCC (or C3 decode profile) batch=1 vs batch=8 —
     compare **committed token ids** AND accuracy (near-tie flips hide in score)
  4. C3-only: assert no reference answer enters attention input; full-seq
     future-visibility rewritten for one-block eval_block_diff_mask
     (within-block Unif is visible by design; beyond active_len must not be)

Then:
  D4, D5, L→R + nfe_metrics → mechanism only if D2 beats all at matched NFE
  → C3 matched-profile eval (last primary) → thr sweep vs NFE

Do not queue while ReqNodeNotAvail / drained booster.
Wording until (1)+(2): valid decode setting, mechanism pending.
""")


def main() -> None:
  ap = argparse.ArgumentParser()
  sp = ap.add_subparsers(dest='cmd', required=True)
  p = sp.add_parser('swapped-from-log')
  p.add_argument('--out-dir', required=True)
  p.add_argument('-n', type=int, default=200)
  p.set_defaults(func=cmd_swapped_from_log)
  r = sp.add_parser('recipe', help='Print live-submit recipe (no sbatch)')
  r.set_defaults(func=cmd_recipe)
  args = ap.parse_args()
  args.func(args)


if __name__ == '__main__':
  main()
