#!/usr/bin/env python3
"""Collapse diagnosis control: Instruct AR vs block-sampler on AR-init weights.

No finetuned checkpoint. Answers:
  A) Is Qwen2.5-1.5B-Instruct fluent under normal AR generate?
  B) Does the block sampler already collapse on fresh AR→block weights
     (load_pretrained, zero block SFT)?

If A fluent and B collapsed → sampler / block decode path.
If A fluent and B fluent → collapse came from block training/recipe.
If A also bad → eval/env problem (unlikely for HF generate).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch


HUB = 'Qwen/Qwen2.5-1.5B-Instruct'
PROMPTS = [
    'Write three sentences about why the sky looks blue.',
    'Explain photosynthesis in simple terms for a child.',
    'List four healthy breakfast ideas and one short tip.',
]


def _collapse_stats(text: str) -> dict:
  toks = text.split()
  n = len(toks)
  if n == 0:
    return {'n_toks': 0, 'uniq_ratio': 0.0, 'top_frac': 1.0, 'collapsed': True}
  counts = Counter(toks)
  top_tok, top_n = counts.most_common(1)[0]
  uniq = len(counts) / n
  top_frac = top_n / n
  # CJK / no-space loops: also check repeated short char n-gram
  chars = re.sub(r'\s+', '', text)
  char_loop = False
  if len(chars) >= 40:
    gram = chars[:4]
    char_loop = chars.count(gram) >= max(8, len(chars) // (len(gram) * 3))
  collapsed = uniq < 0.05 or top_frac > 0.5 or char_loop or len(text.strip()) < 20
  return {
      'n_toks': n,
      'uniq_ratio': round(uniq, 4),
      'top_frac': round(top_frac, 4),
      'top_tok': top_tok[:40],
      'char_loop': char_loop,
      'collapsed': collapsed,
      'preview': text[:160].replace('\n', ' '),
  }


def _write_samples(path: Path, texts: list[str], meta: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as f:
    for i, (t, m) in enumerate(zip(texts, meta)):
      f.write(f'Sample {i}:\n{t}\n')
      f.write(f'[stats] collapsed={m["collapsed"]} '
              f'uniq={m["uniq_ratio"]} top_frac={m["top_frac"]} '
              f'top={m["top_tok"]!r}\n')
      f.write('-' * 80 + '\n')


def run_ar_instruct(out_dir: Path, num_samples: int, max_new_tokens: int,
                    device: torch.device) -> dict:
  from transformers import AutoModelForCausalLM, AutoTokenizer

  print(f'=== A) HF AR Instruct generate ({HUB}) ===', flush=True)
  tok = AutoTokenizer.from_pretrained(HUB, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
      HUB, torch_dtype=torch.bfloat16, trust_remote_code=True)
  model.to(device).eval()

  texts: list[str] = []
  metas: list[dict] = []
  for i in range(num_samples):
    prompt = PROMPTS[i % len(PROMPTS)]
    messages = [{'role': 'user', 'content': prompt}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors='pt').to(device)
    with torch.no_grad():
      out = model.generate(
          input_ids,
          max_new_tokens=max_new_tokens,
          do_sample=True,
          temperature=0.7,
          top_p=0.9,
          pad_token_id=tok.eos_token_id,
      )
    gen = out[0, input_ids.shape[-1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    texts.append(f'[prompt] {prompt}\n[reply] {text}')
    metas.append(_collapse_stats(text))
    print(f'  AR[{i}] collapsed={metas[-1]["collapsed"]} '
          f'preview={metas[-1]["preview"]!r}', flush=True)

  _write_samples(out_dir / 'ar_instruct_samples.txt', texts, metas)
  summary = {
      'n': len(metas),
      'n_collapsed': sum(1 for m in metas if m['collapsed']),
      'samples': metas,
  }
  (out_dir / 'ar_instruct_stats.json').write_text(
      json.dumps(summary, indent=2), encoding='utf-8')
  del model
  torch.cuda.empty_cache()
  return summary


def run_block_ar_init(
    out_dir: Path,
    *,
    algo: str,
    length: int,
    block_size: int,
    steps: int,
    num_samples: int,
    device: torch.device,
) -> dict:
  print(f'=== B) Block sampler on AR-init weights '
        f'(algo={algo}, len={length}, bs={block_size}, steps={steps}) ===',
        flush=True)

  config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra
  from discrete_diffusion.train import register_config_resolvers
  from discrete_diffusion.data import get_tokenizer
  from discrete_diffusion.algorithms.block_trainer import BlockTrainer
  from discrete_diffusion.sampling.block_sampler import BlockSampler

  register_config_resolvers()
  overrides = [
      'data=synthetic',
      'model=qwen_block',
      f'algo={algo}',
      'noise=log-linear',
      f'block_size={block_size}',
      f'model.length={length}',
      f'model.hub_id={HUB}',
      'model.load_pretrained=true',
      f'data.tokenizer_name_or_path={HUB}',
      f'sampling.steps={steps}',
      'sampling=block',
      'eval.generate_samples=false',
      'trainer.accelerator=cuda',
  ]
  GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    config = compose(config_name='config', overrides=overrides)

  tokenizer = get_tokenizer(config)
  model = BlockTrainer(config, tokenizer=tokenizer)
  model.eval()
  model = model.to(device)

  sampler = BlockSampler(config)
  texts: list[str] = []
  metas: list[dict] = []
  for i in range(num_samples):
    with torch.no_grad():
      samples = sampler.generate(
          model, num_samples=1, num_steps=steps, eps=1e-3, inject_bos=True)
    text = tokenizer.decode(samples[0].cpu(), skip_special_tokens=True)
    texts.append(text)
    metas.append(_collapse_stats(text))
    print(f'  block[{algo}][{i}] collapsed={metas[-1]["collapsed"]} '
          f'preview={metas[-1]["preview"]!r}', flush=True)

  tag = f'block_ar_init_{algo}_L{length}'
  _write_samples(out_dir / f'{tag}_samples.txt', texts, metas)
  summary = {
      'algo': algo,
      'length': length,
      'block_size': block_size,
      'steps': steps,
      'n': len(metas),
      'n_collapsed': sum(1 for m in metas if m['collapsed']),
      'samples': metas,
  }
  (out_dir / f'{tag}_stats.json').write_text(
      json.dumps(summary, indent=2), encoding='utf-8')
  del model
  torch.cuda.empty_cache()
  return summary


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--out-dir',
      default='outputs/block_qwen/controls/instruct_collapse',
  )
  parser.add_argument('--num-samples', type=int, default=4)
  parser.add_argument('--ar-max-new-tokens', type=int, default=128)
  parser.add_argument('--length', type=int, default=256,
                      help='Block sample length (keep modest for speed)')
  parser.add_argument('--block-size', type=int, default=32)
  parser.add_argument('--steps', type=int, default=32)
  parser.add_argument(
      '--algos', default='block_masked,block_uniform',
      help='Comma-separated BlockTrainer algos for arm B')
  args = parser.parse_args(argv)

  if not torch.cuda.is_available():
    print('CUDA required', file=sys.stderr)
    return 1

  out_dir = Path(args.out_dir).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  device = torch.device('cuda')

  report = {
      'hub': HUB,
      'ar_instruct': run_ar_instruct(
          out_dir, args.num_samples, args.ar_max_new_tokens, device),
      'block_ar_init': {},
  }
  for algo in [a.strip() for a in args.algos.split(',') if a.strip()]:
    report['block_ar_init'][algo] = run_block_ar_init(
        out_dir,
        algo=algo,
        length=args.length,
        block_size=args.block_size,
        steps=args.steps,
        num_samples=args.num_samples,
        device=device,
    )

  # Verdict
  ar_bad = report['ar_instruct']['n_collapsed']
  block_bad = {
      k: v['n_collapsed'] for k, v in report['block_ar_init'].items()
  }
  if ar_bad == 0 and all(v == 0 for v in block_bad.values()):
    verdict = (
        'AR Instruct fluent AND block-on-AR-init fluent → '
        'collapse likely from block training/recipe, not backbone/sampler alone.'
    )
  elif ar_bad == 0 and any(v > 0 for v in block_bad.values()):
    verdict = (
        'AR Instruct fluent but block-on-AR-init collapsed → '
        'block sampler / decode path is a prime suspect (before blaming data).'
    )
  elif ar_bad > 0:
    verdict = (
        'AR Instruct itself looks bad → check tokenizer/HF generate/env first.'
    )
  else:
    verdict = 'Mixed / inconclusive; inspect sample files.'

  report['verdict'] = verdict
  (out_dir / 'control_report.json').write_text(
      json.dumps(report, indent=2), encoding='utf-8')
  print('\n=== VERDICT ===', flush=True)
  print(verdict, flush=True)
  print(f'Report: {out_dir / "control_report.json"}', flush=True)
  return 0


if __name__ == '__main__':
  # Avoid Hydra/cwd surprises when launched via sbatch
  os.chdir(Path(__file__).resolve().parents[1])
  sys.exit(main())
