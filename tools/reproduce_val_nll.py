#!/usr/bin/env python3
"""Reproduce logged val/nll for a BlockTrainer ckpt with today's forward process.

Compares against wandb-summary ``val/nll`` (or ``--expected``). Stochastic
corruption → expect small noise; report absolute Δ and a pass/fail threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import lightning as pl
import torch
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
  sys.path.insert(0, str(_REPO / 'src'))

from discrete_diffusion.data import get_dataloaders, get_tokenizer  # noqa: E402
from discrete_diffusion.evaluations.checkpoint_utils import (  # noqa: E402
    load_block_trainer_checkpoint,
)
from discrete_diffusion.evaluations.code_fingerprint import (  # noqa: E402
    assert_forward_process_utils_ok,
    code_fingerprint_header,
)
from discrete_diffusion.train import register_config_resolvers  # noqa: E402


def _logged_val_nll(ckpt: Path) -> float | None:
  """Pull final val/nll from the run's wandb-summary if present."""
  # …/checkpoints/last.ckpt → run root is parents[1]
  root = ckpt.resolve().parents[1]
  hits = sorted((root / 'hydra' / 'wandb').glob('**/wandb-summary.json'))
  if not hits:
    return None
  doc = json.loads(hits[-1].read_text())
  v = doc.get('val/nll')
  return float(v) if v is not None else None


def _prep_config_single_gpu(config) -> None:
  OmegaConf.set_struct(config, False)
  OmegaConf.resolve(config)
  # Match train-time per-GPU eval size when possible: eval_global / (devices*nodes).
  eg = int(OmegaConf.select(config, 'loader.eval_global_batch_size') or 32)
  # Single-GPU reproduce: use eval_batch_size = min(eg, 8) to fit login memory.
  ebs = int(OmegaConf.select(config, 'loader.eval_batch_size') or 8)
  ebs = max(1, min(ebs, eg, 8))
  config.trainer.devices = 1
  config.trainer.num_nodes = 1
  config.trainer.accumulate_grad_batches = 1
  config.loader.eval_batch_size = ebs
  config.loader.eval_global_batch_size = ebs
  config.loader.batch_size = ebs
  config.loader.global_batch_size = ebs


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--expected', type=float, default=None,
                  help='Override expected val/nll (default: wandb-summary)')
  ap.add_argument('--limit-val-batches', type=float, default=1.0,
                  help='Lightning limit_val_batches (1.0 = full val)')
  ap.add_argument('--tol', type=float, default=0.05,
                  help='|repro − expected| ≤ tol → match')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', required=True)
  ap.add_argument('--device', default='cuda')
  args = ap.parse_args()

  pl.seed_everything(int(args.seed), workers=True)
  code_fp = assert_forward_process_utils_ok(require_expected_sha=False)
  print(
      f"code_fingerprint utils.py sha256={code_fp['sha256']} "
      f"n_lines={code_fp['n_lines']}",
      flush=True)

  device = torch.device(
      args.device if args.device != 'cuda' or torch.cuda.is_available()
      else 'cpu')
  register_config_resolvers()
  ckpt = Path(args.checkpoint).expanduser().resolve()
  model, config, tokenizer = load_block_trainer_checkpoint(ckpt, device)
  _prep_config_single_gpu(config)
  # Reload tokenizer/dataloaders with single-GPU loader sizes.
  tokenizer = get_tokenizer(config)
  _, val_loader = get_dataloaders(config, tokenizer, skip_train=True)

  expected = args.expected
  if expected is None:
    expected = _logged_val_nll(ckpt)

  trainer = pl.Trainer(
      accelerator='gpu' if device.type == 'cuda' else 'cpu',
      devices=1,
      limit_val_batches=args.limit_val_batches,
      logger=False,
      enable_checkpointing=False,
      enable_progress_bar=True,
  )
  results = trainer.validate(model, val_loader, verbose=True)
  val_nll = None
  for row in results or []:
    if isinstance(row, dict) and 'val/nll' in row:
      val_nll = float(row['val/nll'])
  if val_nll is None:
    raise SystemExit(f'No val/nll in trainer results: {results}')

  delta = None if expected is None else float(val_nll) - float(expected)
  match = (
      None if expected is None
      else abs(delta) <= float(args.tol))
  payload = {
      'protocol': 'val_nll_reproduce_vs_train_log',
      'code_fingerprint': code_fingerprint_header(),
      'checkpoint': str(ckpt),
      'expected_val_nll': expected,
      'reproduced_val_nll': val_nll,
      'delta': delta,
      'abs_delta': None if delta is None else abs(delta),
      'tol': float(args.tol),
      'match': match,
      'val_bpd': float(val_nll) / math.log(2),
      'val_ppl': math.exp(val_nll),
      'limit_val_batches': args.limit_val_batches,
      'eval_batch_size': int(config.loader.eval_batch_size),
      'seed': int(args.seed),
      'wording': (
          'Forward-process code at train time is not recoverable byte-for-byte; '
          'the full API was present (checkpoint loads the dependent class), and '
          + (
              f'validation NLL is reproduced to within {abs(delta):.4f} '
              f'(tol={args.tol}) with the committed version.'
              if match
              else (
                  f'validation NLL is NOT reproduced '
                  f'(Δ={delta}, tol={args.tol}) — do not imply equivalence.'
                  if expected is not None
                  else 'validation NLL not compared (no expected value).'
              )
          )
      ),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2) + '\n')
  print(json.dumps({
      'expected': expected,
      'reproduced': val_nll,
      'delta': delta,
      'match': match,
      'out': str(out),
  }, indent=2))
  return 0 if match or expected is None else 2


if __name__ == '__main__':
  sys.exit(main())
