#!/usr/bin/env python3
"""Export a UNI-D² BlockTrainer ckpt into a Hub Fast-dLLM HF folder.

Copies Hub template (modeling.py / tokenizer / config) and replaces
``model.safetensors`` with our backbone weights (EMA when present), so
``third_party/Fast-dLLM/v2/eval.py`` can run Hub ``batch_sample`` on our
conversion checkpoint.

Usage:
  PYTHONPATH=src python tools/export_block_ckpt_to_fastdllm_hf.py \\
    --ckpt /path/to/last.ckpt \\
    --out /path/to/export_dir
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_HUB_TEMPLATE = (
    '/e/scratch/scifi/elsayed3/hf_cache/models/Fast_dLLM_v2_1.5B'
)
HUB_VOCAB = 151936
PREFIX = 'backbone.model.'


def _backbone_export_tensors(
    ckpt: dict, *, use_ema: bool, allow_embed_pad: bool = False,
) -> dict[str, torch.Tensor]:
  sd = ckpt.get('state_dict') or {}
  # Hub Fast_dLLM_Qwen keys are ``model.*`` (no lm_head; tied embeddings).
  keys = [
      k for k in sd
      if k.startswith(PREFIX) and 'lm_head' not in k
  ]
  if not keys:
    raise ValueError('No backbone.model.* tensors found in checkpoint')

  if use_ema and isinstance(ckpt.get('ema'), dict) and ckpt['ema'].get(
      'shadow_params'):
    shadows = ckpt['ema']['shadow_params']
    if len(shadows) < len(keys):
      raise ValueError(
          f'EMA shadow_params ({len(shadows)}) shorter than backbone '
          f'tensors ({len(keys)})')
    # Lightning EMA follows ``backbone.parameters()`` then noise; for our
    # Qwen conversion the first ``len(keys)`` shadows match ``keys`` order
    # (verified: 338 shapes, exact Hub key set, noise trainable empty).
    # Fail loud if shapes diverge — do not silently mis-zip.
    for i, k in enumerate(keys):
      if tuple(shadows[i].shape) != tuple(sd[k].shape):
        raise ValueError(
            f'EMA shadow[{i}] shape {tuple(shadows[i].shape)} != '
            f'{k} shape {tuple(sd[k].shape)} — refuse export')
    noise_float = [
        k for k, v in sd.items()
        if k.startswith('noise.') and getattr(v, 'dtype', None) is not None
        and v.dtype.is_floating_point
    ]
    if noise_float:
      raise ValueError(
          f'Export assumes no trainable noise params; found {noise_float[:4]}')
    tensors = {
        k[len(PREFIX):]: shadows[i].detach().to(torch.bfloat16).contiguous()
        for i, k in enumerate(keys)
    }
    src = 'ema'
  else:
    tensors = {
        k[len(PREFIX):]: sd[k].detach().to(torch.bfloat16).contiguous()
        for k in keys
    }
    src = 'state_dict'

  embed = tensors.get('model.embed_tokens.weight')
  if embed is None:
    raise ValueError('Missing model.embed_tokens.weight after remap')
  rows = int(embed.shape[0])
  if rows < HUB_VOCAB:
    if not allow_embed_pad:
      raise ValueError(
          f'embed rows {rows} < Hub vocab {HUB_VOCAB}; refuse silent '
          f'zero-pad (pass allow_embed_pad=True only for probes)')
    pad = torch.zeros(
        (HUB_VOCAB - rows, embed.shape[1]),
        dtype=embed.dtype,
    )
    tensors['model.embed_tokens.weight'] = torch.cat([embed, pad], dim=0)
    print(f'[export] padded embed {rows}→{HUB_VOCAB}', flush=True)
  elif rows > HUB_VOCAB:
    raise ValueError(f'embed rows {rows} > Hub vocab {HUB_VOCAB}')

  print(
      f'[export] source={src} n_tensors={len(tensors)} '
      f'embed={tuple(tensors["model.embed_tokens.weight"].shape)}',
      flush=True,
  )
  return tensors


def export(
    ckpt_path: Path,
    out_dir: Path,
    template: Path,
    *,
    use_ema: bool,
    allow_embed_pad: bool = False,
) -> None:
  if not ckpt_path.is_file():
    raise FileNotFoundError(ckpt_path)
  if not template.is_dir():
    raise FileNotFoundError(f'Hub template missing: {template}')

  print(f'[export] loading {ckpt_path}', flush=True)
  ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  tensors = _backbone_export_tensors(
      ckpt, use_ema=use_ema, allow_embed_pad=allow_embed_pad)

  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=False)

  # Copy Hub code + tokenizer; drop template weights (we rewrite).
  skip = {'model.safetensors', 'model.safetensors.index.json'}
  for p in template.iterdir():
    if p.name in skip or p.name.startswith('.'):
      continue
    dest = out_dir / p.name
    if p.is_dir():
      shutil.copytree(p, dest)
    else:
      shutil.copy2(p, dest)

  # Ensure config vocab / bd_size.
  cfg_path = out_dir / 'config.json'
  cfg = json.loads(cfg_path.read_text())
  cfg['vocab_size'] = HUB_VOCAB
  cfg.setdefault('bd_size', 32)
  cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')

  # Embed chat_template into tokenizer_config so transformers 4.45+ and 4.53
  # both see it (Hub ships chat_template.jinja; older TF ignores that file).
  tok_cfg_path = out_dir / 'tokenizer_config.json'
  if tok_cfg_path.is_file():
    tok_cfg = json.loads(tok_cfg_path.read_text())
    jinja = out_dir / 'chat_template.jinja'
    if jinja.is_file():
      tok_cfg['chat_template'] = jinja.read_text()
    elif 'chat_template' not in tok_cfg:
      from discrete_diffusion.data.conversion_baseline import (
          FAST_DLLM_SFT_CHAT_TEMPLATE,
      )
      tok_cfg['chat_template'] = FAST_DLLM_SFT_CHAT_TEMPLATE
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + '\n')
    print('[export] wrote tokenizer_config.chat_template', flush=True)

  weights_path = out_dir / 'model.safetensors'
  save_file(tensors, str(weights_path))
  print(f'[export] wrote {weights_path} ({len(tensors)} tensors)', flush=True)

  # Key-set sanity vs template (when present).
  tmpl_weights = template / 'model.safetensors'
  if tmpl_weights.is_file():
    from safetensors import safe_open
    with safe_open(str(tmpl_weights), framework='pt', device='cpu') as f:
      hub_keys = set(f.keys())
    ours = set(tensors)
    missing = sorted(hub_keys - ours)
    extra = sorted(ours - hub_keys)
    if missing or extra:
      raise RuntimeError(
          f'Key mismatch vs Hub template: missing={missing[:8]} '
          f'extra={extra[:8]}')
    print('[export] key set matches Hub template', flush=True)

  meta = {
      'source_ckpt': str(ckpt_path.resolve()),
      'template': str(template.resolve()),
      'use_ema': use_ema,
      'n_tensors': len(tensors),
      'vocab_size': HUB_VOCAB,
  }
  (out_dir / 'export_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
  print(f'[export] done → {out_dir}', flush=True)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument(
      '--template', type=Path, default=Path(DEFAULT_HUB_TEMPLATE))
  ap.add_argument(
      '--no-ema', action='store_true',
      help='Use raw state_dict instead of EMA shadows')
  ap.add_argument(
      '--allow-embed-pad', action='store_true',
      help='Allow zero-padding embed 151666→151936 (probe only)')
  args = ap.parse_args()
  export(
      args.ckpt, args.out, args.template,
      use_ema=not args.no_ema,
      allow_embed_pad=args.allow_embed_pad,
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
