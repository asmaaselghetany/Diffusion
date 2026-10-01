#!/usr/bin/env python3
"""One-shot: prune periodic step ckpts under outputs/block_qwen/*/checkpoints.

Keeps last.ckpt, best.ckpt, and the newest N ``{epoch}-{step}.ckpt`` files.
Default N=2. Use --dry-run to print plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from discrete_diffusion.callbacks.prune_periodic_checkpoints import (  # noqa: E402
    prune_periodic_checkpoints,
)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument(
      '--root',
      default=str(ROOT / 'outputs' / 'block_qwen'),
      help='block_qwen outputs root (symlink OK)',
  )
  ap.add_argument('--keep-newest', type=int, default=2)
  ap.add_argument('--dry-run', action='store_true')
  ap.add_argument(
      '--only',
      action='append',
      default=[],
      help='Basename filter (repeatable), e.g. ar2block_uniform_1955203',
  )
  args = ap.parse_args()
  root = Path(args.root).expanduser().resolve()
  if not root.is_dir():
    print(f'missing {root}', file=sys.stderr)
    return 2
  freed = 0
  n_files = 0
  for ckpt_dir in sorted(root.glob('*/checkpoints')):
    run = ckpt_dir.parent.name
    if args.only and run not in args.only:
      continue
    # dry-run: compute victims without delete
    if args.dry_run:
      from discrete_diffusion.callbacks.prune_periodic_checkpoints import (
          _PERIODIC_RE,
      )
      periodics = []
      for path in ckpt_dir.iterdir():
        m = _PERIODIC_RE.match(path.name)
        if m and path.is_file():
          periodics.append((int(m.group(2)), int(m.group(1)), path))
      periodics.sort()
      victims = periodics[:-args.keep_newest] if args.keep_newest else periodics
      if not victims:
        continue
      size = sum(p.stat().st_size for _, _, p in victims)
      print(f'{run}: would remove {len(victims)} files ({size/1e9:.1f} GB)')
      for _, _, p in victims:
        print(f'  - {p.name}')
      freed += size
      n_files += len(victims)
      continue
    removed = []
    # Size before delete for accounting.
    from discrete_diffusion.callbacks.prune_periodic_checkpoints import (
        _PERIODIC_RE,
    )
    periodics = []
    for path in ckpt_dir.iterdir():
      m = _PERIODIC_RE.match(path.name) if path.is_file() else None
      if m:
        periodics.append((int(m.group(2)), int(m.group(1)), path))
    periodics.sort()
    victims = [p for _, _, p in (
        periodics[:-args.keep_newest] if args.keep_newest else periodics)]
    size = sum(p.stat().st_size for p in victims)
    removed = prune_periodic_checkpoints(
        ckpt_dir, keep_newest=args.keep_newest)
    if not removed:
      continue
    print(f'{run}: removed {len(removed)} periodic(s) (~{size/1e9:.1f} GB)')
    freed += size
    n_files += len(removed)
  print(f'TOTAL removed {n_files} files (~{freed/1e9:.1f} GB)')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
