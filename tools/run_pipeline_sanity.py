#!/usr/bin/env python3
"""Pipeline kill-ladder: model vs sampler vs plumbing.

Fork logic (print + JSON ``verdict``):
  A (one-step low-t on REAL text) fails  → model/undertraining (sampler not root)
  A passes, B (multi-step from same xt) fails → sampler / schedule bug
  A and B pass → look at free-gen / harness surface

Also reports plumbing: mask_id, block_size, ignore_bos, attn-hook contract,
train-style token NLL on the same batch.

Usage:
  python tools/run_pipeline_sanity.py --checkpoint path/to/last.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


FIXED_TEXTS = [
    'The capital of France is Paris. The answer is 7.',
    'User: What is 2+2?\nAssistant: The answer is 4.',
    'Once upon a time there was a small village by the river.',
]


def _load(checkpoint: str, device: str):
  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.train import register_config_resolvers
  from omegaconf import OmegaConf

  register_config_resolvers()
  model, config, tokenizer = load_block_trainer_checkpoint(
      checkpoint, torch.device(device))
  OmegaConf.resolve(config)
  return model, tokenizer, config


def _encode_batch(tokenizer, texts: list[str], seq: int, device: str) -> torch.Tensor:
  """Encode texts and **tile** to fill ``seq`` (avoid pad-dominated probes)."""
  pad_id = tokenizer.pad_token_id
  if pad_id is None:
    pad_id = tokenizer.eos_token_id
  rows = []
  for t in texts:
    ids = tokenizer.encode(t, add_special_tokens=True)
    if not ids:
      ids = [pad_id]
    # tile content to fill the model length (paper arms = 2048)
    tiled = (ids * ((seq // len(ids)) + 1))[:seq]
    rows.append(tiled)
  return torch.tensor(rows, device=device, dtype=torch.long)


@torch.no_grad()
def force_content_holes(
    model, x0: torch.Tensor, *, frac: float, content: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Deterministic holes on content positions (bypasses low-t empty-hole luck)."""
  xt = x0.clone()
  holes = torch.zeros_like(x0, dtype=torch.bool)
  for b in range(x0.shape[0]):
    idx = content[b].nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
      continue
    n = max(1, int(idx.numel() * frac))
    # evenly spaced for stability across runs
    sel = idx[torch.linspace(0, idx.numel() - 1, n).long()]
    holes[b, sel] = True
    if model.forward_process_name == 'masked':
      xt[b, sel] = model.mask_id
    else:
      # replace with a different vocab id (avoid identity)
      noise = torch.randint(
          0, model.vocab_size, (n,), device=x0.device, dtype=x0.dtype)
      same = noise == x0[b, sel]
      noise = torch.where(
          same, (noise + 1) % model.vocab_size, noise)
      xt[b, sel] = noise
  return xt, holes


@torch.no_grad()
def one_step_on_xt(
    model, x0: torch.Tensor, xt: torch.Tensor, holes: torch.Tensor,
) -> dict:
  logits = model._backbone_logits(xt, x0, block_size=model.block_size)
  pred = logits.argmax(dim=-1)
  n_holes = int(holes.sum().item())
  if n_holes == 0:
    return {'exact': 1.0, 'n_holes': 0, 'xt': xt, 'holes': holes, 'pred': pred}
  exact = float((pred[holes] == x0[holes]).float().mean().item())
  return {'exact': exact, 'n_holes': n_holes, 'xt': xt, 'holes': holes, 'pred': pred}


def _load_real_batch(
    model, tokenizer, config, seq: int, batch_size: int, device: str,
    *, prefer_val: bool,
):
  """Prefer val loader when requested; else fixed English strings."""
  del model
  if prefer_val:
    try:
      from discrete_diffusion.data import get_dataloaders
      _, valid = get_dataloaders(config, tokenizer, skip_train=True)
      batch = next(iter(valid))
      x0 = batch['input_ids'].to(device)[:batch_size, :seq]
      if x0.shape[1] < seq:
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        x0 = F.pad(x0, (0, seq - x0.shape[1]), value=pad)
      return x0, 'val_loader'
    except Exception as exc:  # noqa: BLE001 — probe must still run
      x0 = _encode_batch(tokenizer, FIXED_TEXTS[:batch_size], seq, device)
      return x0, f'fixed_texts_fallback ({type(exc).__name__}: {exc})'
  x0 = _encode_batch(tokenizer, FIXED_TEXTS[:batch_size], seq, device)
  return x0, 'fixed_texts'


@torch.no_grad()
def one_step_exact(model, x0: torch.Tensor, t_val: float, block_size: int,
                   *, content_mask: torch.Tensor | None = None) -> dict:
  bsz, seq = x0.shape
  t = torch.full((bsz, seq), t_val, device=x0.device, dtype=torch.float32)
  for bi in range(seq // block_size):
    sl = slice(bi * block_size, (bi + 1) * block_size)
    t[:, sl] = t_val
  xt = model._corrupt(x0, t, block_size=block_size)
  logits = model._backbone_logits(xt, x0, block_size=block_size)
  pred = logits.argmax(dim=-1)
  if model.forward_process_name == 'masked':
    holes = xt == model.mask_id
  else:
    holes = xt != x0
  if content_mask is not None:
    holes = holes & content_mask
  n_holes = int(holes.sum().item())
  if n_holes == 0:
    return {'exact': 1.0, 'n_holes': 0, 'xt': xt, 'holes': holes}
  exact = float((pred[holes] == x0[holes]).float().mean().item())
  return {'exact': exact, 'n_holes': n_holes, 'xt': xt, 'holes': holes, 'pred': pred}


@torch.no_grad()
def multi_step_from_xt(
    model, x0_true: torch.Tensor, xt: torch.Tensor, holes: torch.Tensor,
    *, num_steps: int, block_size: int, eps: float,
) -> dict:
  """Whole-sequence reverse on the *same* xt; restore locked tokens each step.

  Uses BlockSampler step_fn (masked/uniform). O(num_steps) forwards — not
  O(num_blocks * num_steps).
  """
  del block_size
  from discrete_diffusion.sampling.block_sampler import BlockSampler

  sampler = BlockSampler(model.config)
  bsz = xt.shape[0]
  cur = xt.clone()
  # Clean half: true tokens (train-like context). Same-index still masked.
  x0_ctx = x0_true.clone()
  dt = (1.0 - eps) / max(num_steps, 1)
  timesteps = torch.linspace(1.0, eps, num_steps + 1, device=xt.device)
  step_fn = sampler._masked_step if sampler.is_masked else sampler._uniform_step
  locked = ~holes

  for i in range(num_steps):
    cur = torch.where(locked, x0_true, cur)
    x0_ctx = torch.where(locked, x0_true, cur)
    t = timesteps[i].expand(bsz)
    cur = step_fn(model, cur, x0_ctx, t, dt)
  cur = torch.where(locked, x0_true, cur)
  x0_ctx = torch.where(locked, x0_true, cur)
  t_final = timesteps[-1].expand(bsz)
  cur = step_fn(model, cur, x0_ctx, t_final, None)
  cur = torch.where(locked, x0_true, cur)

  n_holes = int(holes.sum().item())
  if n_holes == 0:
    return {'exact': 1.0, 'n_holes': 0}
  exact = float((cur[holes] == x0_true[holes]).float().mean().item())
  return {'exact': exact, 'n_holes': n_holes}


@torch.no_grad()
def train_style_nll(model, x0: torch.Tensor, content: torch.Tensor) -> float:
  valid = content.to(dtype=torch.float32)
  if model.ignore_bos:
    valid = valid.clone()
    valid[:, 0] = 0
  loss = model._loss(x0, valid)
  return float(loss.loss.item())


def plumbing_report(model, config) -> dict:
  from discrete_diffusion.contracts.attention_hook import (
      assert_block_attention_hook_compatible,
  )
  hook_ok = None
  hook_err = None
  try:
    backbone = model.backbone
    inner = getattr(backbone, 'model', backbone)
    # QwenBlock wraps HF; hook installs on HF causal LM
    hf = getattr(backbone, 'transformer', None) or getattr(backbone, 'model', None)
    target = hf if hf is not None else inner
    assert_block_attention_hook_compatible(target)
    hook_ok = True
  except Exception as exc:  # noqa: BLE001
    hook_ok = False
    hook_err = f'{type(exc).__name__}: {exc}'
  return {
      'forward_process': model.forward_process_name,
      'block_size_model': int(model.block_size),
      'block_size_config': int(getattr(config, 'block_size', -1)),
      'num_tokens': int(model.num_tokens),
      'mask_id': int(getattr(model, 'mask_id', -1)),
      'ignore_bos': bool(model.ignore_bos),
      'attn_hook_compatible': hook_ok,
      'attn_hook_err': hook_err,
      'backbone_block_size': int(getattr(model.backbone, 'block_size', -1)),
  }


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--device', default='cuda')
  p.add_argument(
      '--seq-len', type=int, default=None,
      help='Token length. Default: model.num_tokens (Qwen block-diff '
           'requires concat length 2n == 2*num_tokens; shorter crashes).')
  p.add_argument('--batch-size', type=int, default=3)
  p.add_argument('--t', type=float, default=0.05,
                 help='Unused when --hole-frac is set (forced holes). Kept for logs.')
  p.add_argument('--hole-frac', type=float, default=0.15,
                 help='Fraction of content tokens to corrupt for A/B (forced).')
  p.add_argument('--steps', type=int, default=16)
  p.add_argument(
      '--use-val', action='store_true',
      help='Load a Nemotron val batch (slow/cold-cache). Default: fixed English strings.')
  p.add_argument('--output', default=None)
  args = p.parse_args()

  repo = Path(__file__).resolve().parents[1]
  if str(repo / 'src') not in sys.path:
    sys.path.insert(0, str(repo / 'src'))

  print('[pipeline_sanity] loading checkpoint…', flush=True)
  model, tokenizer, config = _load(args.checkpoint, args.device)
  bs = int(model.block_size)
  # Backbone hard-requires concat(xt,x0) length == 2 * n_tokens.
  n_model = int(model.num_tokens)
  seq = int(args.seq_len) if args.seq_len is not None else n_model
  if seq != n_model:
    raise SystemExit(
        f'seq-len={seq} incompatible with checkpoint n_tokens={n_model} '
        f'(backbone expects concat length {2 * n_model}). '
        f'Pass --seq-len {n_model} or omit --seq-len.')
  if seq % bs != 0:
    raise SystemExit(f'seq-len={seq} must divide block_size={bs}')

  print('[pipeline_sanity] plumbing…', flush=True)
  plumbing = plumbing_report(model, config)
  print('[pipeline_sanity] building batch…', flush=True)
  x0, data_src = _load_real_batch(
      model, tokenizer, config, seq, args.batch_size, args.device,
      prefer_val=bool(args.use_val))
  pad_id = tokenizer.pad_token_id
  if pad_id is None:
    pad_id = tokenizer.eos_token_id
  content = x0 != pad_id
  print(
      f'[pipeline_sanity] content_tokens={int(content.sum())}/{x0.numel()}',
      flush=True)

  print('[pipeline_sanity] forcing content holes…', flush=True)
  xt, holes = force_content_holes(
      model, x0, frac=float(args.hole_frac), content=content)
  print(f'[pipeline_sanity] holes={int(holes.sum())}', flush=True)

  print('[pipeline_sanity] A one-step…', flush=True)
  a = one_step_on_xt(model, x0, xt, holes)
  print(f"[pipeline_sanity] A exact={a['exact']:.4f} holes={a['n_holes']}", flush=True)
  print('[pipeline_sanity] B multi-step…', flush=True)
  b = multi_step_from_xt(
      model, x0, xt, holes,
      num_steps=args.steps, block_size=bs,
      eps=float(getattr(model, 'sampling_eps', 1e-3)))
  print(f"[pipeline_sanity] B exact={b['exact']:.4f}", flush=True)
  print('[pipeline_sanity] train-style NLL…', flush=True)
  nll = train_style_nll(model, x0, content)

  a_ok = a['exact'] >= 0.90
  b_ok = b['exact'] >= 0.90
  if not a_ok:
    verdict = 'MODEL_OR_UNDERTRAINING'
    detail = (
        'One-step low-t exact failed on real/fixed text with teacher x0. '
        'Root cause is not multi-step BlockSampler.')
  elif a_ok and not b_ok:
    verdict = 'SAMPLER_OR_SCHEDULE'
    detail = (
        'One-step OK but multi-step from same xt failed. '
        'Inspect BlockSampler reverse / dt / x0 updates.')
  else:
    verdict = 'PIPELINE_OK_AT_LOW_T'
    detail = (
        'Low-t one-step and multi-step recover. '
        'Soup/harness fail would be free-gen prior or surface, not low-t denoise.')

  report = {
      'checkpoint': args.checkpoint,
      'data_source': data_src,
      't': args.t,
      'hole_frac': args.hole_frac,
      'steps': args.steps,
      'seq': seq,
      'block_size': bs,
      'plumbing': plumbing,
      'train_style_token_nll': nll,
      'A_one_step_exact': a['exact'],
      'A_n_holes': a['n_holes'],
      'B_multi_step_exact': b['exact'],
      'B_n_holes': b['n_holes'],
      'A_pass': a_ok,
      'B_pass': b_ok,
      'verdict': verdict,
      'detail': detail,
  }
  print(json.dumps(report, indent=2))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
  return 0 if verdict == 'PIPELINE_OK_AT_LOW_T' else 3


if __name__ == '__main__':
  raise SystemExit(main())
