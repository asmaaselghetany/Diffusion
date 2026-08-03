#!/usr/bin/env python3
"""Republish offline ar2block_masked metrics into a dedicated WandB history run.

The live run cannot reliably accept older steps via `wandb sync --append` once it
has already advanced past ~3k. This script reconstructs trainer/loss and val/bpd
curves from offline .wandb transaction logs and uploads them to a companion run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import wandb


def parse_metrics(root: Path) -> tuple[dict[int, float], dict[int, float]]:
  loss_by_step: dict[int, float] = {}
  bpd_by_step: dict[int, float] = {}

  pat_loss = re.compile(
      r'trainer/global_step.{0,24}?(\d{1,5}).{0,120}?trainer/loss.{0,24}?([0-9]+\.[0-9]+)',
      re.DOTALL,
  )
  pat_loss2 = re.compile(
      r'trainer/loss.{0,24}?([0-9]+\.[0-9]+).{0,120}?trainer/global_step.{0,24}?(\d{1,5})',
      re.DOTALL,
  )
  pat_bpd = re.compile(
      r'trainer/global_step.{0,24}?(\d{1,5}).{0,160}?val/bpd.{0,24}?([0-9]+\.[0-9]+)',
      re.DOTALL,
  )
  pat_bpd2 = re.compile(
      r'val/bpd.{0,24}?([0-9]+\.[0-9]+).{0,160}?trainer/global_step.{0,24}?(\d{1,5})',
      re.DOTALL,
  )

  files = sorted(root.glob('offline-run-*/*.wandb'))
  print(f'offline files: {len(files)}')
  for wb in files:
    text = wb.read_bytes().decode('latin1', errors='ignore')
    before = len(loss_by_step)
    for m in pat_loss.finditer(text):
      step, loss = int(m.group(1)), float(m.group(2))
      if 0 <= step <= 7500 and 0 < loss < 50:
        loss_by_step[step] = loss
    for m in pat_loss2.finditer(text):
      loss, step = float(m.group(1)), int(m.group(2))
      if 0 <= step <= 7500 and 0 < loss < 50:
        loss_by_step[step] = loss
    for m in pat_bpd.finditer(text):
      step, bpd = int(m.group(1)), float(m.group(2))
      if 0 <= step <= 7500 and 0 < bpd < 30:
        bpd_by_step[step] = bpd
    for m in pat_bpd2.finditer(text):
      bpd, step = float(m.group(1)), int(m.group(2))
      if 0 <= step <= 7500 and 0 < bpd < 30:
        bpd_by_step[step] = bpd
    print(
        f'  {wb.parent.name}: +{len(loss_by_step) - before} loss steps '
        f'(total loss={len(loss_by_step)}, bpd={len(bpd_by_step)})'
    )
  return loss_by_step, bpd_by_step


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--root',
      default='outputs/block_qwen/ar2block_masked_131655/hydra/wandb',
  )
  parser.add_argument('--entity', default='asmaaselghetany-elghitany')
  parser.add_argument('--project', default='block_qwen')
  parser.add_argument('--name', default='ar2block_masked_full_history')
  parser.add_argument('--id', default='bqwen_ar2block_masked_131655_full_history')
  args = parser.parse_args()

  loss_by_step, bpd_by_step = parse_metrics(Path(args.root))
  steps = sorted(set(loss_by_step) | set(bpd_by_step))
  if not steps:
    raise SystemExit('No metrics parsed from offline runs')

  print(
      f'merged steps={len(steps)} min={steps[0]} max={steps[-1]} '
      f'below_1950={sum(1 for s in steps if s < 1950)}'
  )

  wandb.init(
      project=args.project,
      entity=args.entity,
      name=args.name,
      id=args.id,
      resume='allow',
      notes='Reconstructed offline metric history for ar2block_masked_131655.',
      tags=['history-replay', 'ar2block', 'masked'],
  )
  for step in steps:
    payload = {'trainer/global_step': step}
    if step in loss_by_step:
      payload['trainer/loss'] = loss_by_step[step]
    if step in bpd_by_step:
      payload['val/bpd'] = bpd_by_step[step]
    wandb.log(payload, step=step)
  wandb.finish()
  print(
      f'published {len(steps)} points -> '
      f'https://wandb.ai/{args.entity}/{args.project}/runs/{args.id}'
  )


if __name__ == '__main__':
  main()
