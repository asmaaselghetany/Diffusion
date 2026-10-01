#!/usr/bin/env python3
"""Record a per-step block-diffusion decode trace for video rendering.

Works for any BlockTrainer checkpoint (masked / uniform / hybrid, any
decode profile). Writes::

  OUT/meta.json
  OUT/events.jsonl

Example::

  cd Diffusion-new
  export PYTHONPATH=src
  python scripts/viz/record_decode_trace.py \\
    --ckpt outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt \\
    --label baseline_masked \\
    --out docs/research/decode_viz/baseline_masked \\
    --prompt "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" \\
    --decode-profile hubmatch --unmask-threshold 1.0 \\
    --max-new-tokens 128 --num-steps 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# scripts/viz → repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / 'src') not in sys.path:
  sys.path.insert(0, str(_ROOT / 'src'))

from discrete_diffusion.evaluations.checkpoint_utils import (  # noqa: E402
    load_block_trainer_checkpoint,
)
from discrete_diffusion.evaluations.decode_profiles import (  # noqa: E402
    DECODE_PROFILES,
    conversion_prefix_ids,
    profile_overrides,
)

DEFAULT_PROMPT = (
    'Natalia sold clips to 48 of her friends in April, and then she sold '
    'half as many clips in May. How many clips did Natalia sell altogether '
    'in April and May?'
)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--ckpt', required=True, type=Path)
  p.add_argument('--out', required=True, type=Path)
  p.add_argument('--label', default='', help='Short name for the video legend')
  p.add_argument('--prompt', default=DEFAULT_PROMPT)
  p.add_argument(
      '--free-gen', action='store_true',
      help='Use conversion_free assistant prefix only (no user prompt)')
  p.add_argument('--decode-profile', default='baseline',
                 choices=sorted(DECODE_PROFILES) + ['keep'])
  p.add_argument('--unmask-threshold', default=None,
                 help='Override thr (e.g. 1.0 for hubmatch confidence)')
  p.add_argument('--max-new-tokens', type=int, default=128)
  p.add_argument('--num-steps', type=int, default=32)
  # Default False: uniform argmax(q) locks prior noise. Masked hubmatch
  # callers should pass --greedy explicitly (confidence commit).
  p.add_argument('--greedy', action='store_true', default=False)
  p.add_argument('--no-greedy', action='store_false', dest='greedy')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  device = torch.device(args.device)
  if args.seed is not None:
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
      torch.cuda.manual_seed_all(args.seed)

  overrides = profile_overrides(args.decode_profile)
  if args.unmask_threshold is not None:
    thr = args.unmask_threshold.strip()
    if thr in ('null', 'None', ''):
      overrides = [o for o in overrides if not o.startswith('sampling.unmask_threshold=')]
      overrides.append('sampling.unmask_threshold=null')
    else:
      overrides = [o for o in overrides if not o.startswith('sampling.unmask_threshold=')]
      overrides.append(f'sampling.unmask_threshold={float(thr)}')

  model, cfg, tok = load_block_trainer_checkpoint(
      args.ckpt, device, hydra_overrides=overrides)
  sampler = model._create_sampler()

  user = None if args.free_gen else args.prompt
  prefix, prefix_text = conversion_prefix_ids(tok, user, device)
  prefix_len = int(prefix.shape[1])

  events: list[dict] = []

  mask_id = getattr(model, 'mask_id', None)

  def hook(event: dict) -> None:
    # keep JSONL light: drop huge future MASK canvas beyond active_end + margin
    ids = event['token_ids']
    cut = min(len(ids), int(event['active_end']) + 8)
    ids = ids[:cut]
    event = dict(event)
    event['token_ids'] = ids
    # Pre-decode pieces so render needs no HF tokenizer / offline Hub.
    strs = []
    for tid in ids:
      if mask_id is not None and int(tid) == int(mask_id):
        strs.append('░')
      else:
        strs.append(tok.decode([int(tid)], skip_special_tokens=False))
    event['token_strs'] = strs
    events.append(event)

  sampler.step_hook = hook
  with torch.inference_mode():
    out = sampler.generate(
        model,
        num_samples=1,
        num_steps=int(args.num_steps),
        eps=None,
        inject_bos=False,
        prefix_ids=prefix,
        max_new_tokens=int(args.max_new_tokens),
        greedy=bool(args.greedy),
    )
  sampler.step_hook = None

  final_ids = out[0].detach().cpu().tolist()
  # trim trailing masks for final text
  mask_id = getattr(model, 'mask_id', None)
  if mask_id is not None:
    while final_ids and final_ids[-1] == mask_id:
      final_ids.pop()
  final_text = tok.decode(final_ids, skip_special_tokens=False)

  label = args.label or Path(args.ckpt).parents[1].name
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  meta = {
      'label': label,
      'checkpoint': str(Path(args.ckpt).resolve()),
      'decode_profile': args.decode_profile,
      'unmask_threshold': args.unmask_threshold,
      'hydra_overrides': overrides,
      'prompt': user,
      'prefix_text': prefix_text,
      'prefix_len': prefix_len,
      'max_new_tokens': int(args.max_new_tokens),
      'num_steps': int(args.num_steps),
      'greedy': bool(args.greedy),
      'seed': args.seed,
      'forward_process': str(getattr(cfg.algo, 'forward_process_name', '')),
      'block_size': int(getattr(model, 'block_size', 0) or 0),
      'mask_id': None if mask_id is None else int(mask_id),
      'n_events': len(events),
      'final_text': final_text,
      'final_token_ids': final_ids,
  }
  (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
  with (out_dir / 'events.jsonl').open('w') as f:
    for ev in events:
      f.write(json.dumps(ev, ensure_ascii=False) + '\n')
  (out_dir / 'final.txt').write_text(final_text + '\n')
  print(json.dumps({
      'out': str(out_dir),
      'n_events': len(events),
      'prefix_len': prefix_len,
      'final_chars': len(final_text),
  }, indent=2))


if __name__ == '__main__':
  main()
