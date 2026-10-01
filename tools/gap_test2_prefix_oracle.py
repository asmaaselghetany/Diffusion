#!/usr/bin/env python3
"""Test 2 — prefix-oracle curve (gap diagnosis M−U).

Builds AR-teacher prefix cuts, then runs BlockSampler continuation under a
fixed decode profile. Cutoffs are preregistered in
``docs/research/GAP_DIAGNOSIS_M_U_2026-10-01.md``.

Build prefix set (from a JSONL of correct AR completions):
  python tools/gap_test2_prefix_oracle.py build-prefixes \\
    --ar-jsonl path/to/ar_correct.jsonl --tokenizer-from-ckpt .../2020048/.../last.ckpt \\
    --block-size 32 --fractions 0,0.25,0.5,0.75 --output prefixes.jsonl

Run one (arm, profile, fraction) cell:
  python tools/gap_test2_prefix_oracle.py run \\
    --ckpt .../2020049/checkpoints/last.ckpt \\
    --prefixes prefixes.jsonl --fraction 0.5 \\
    --decode-profile hierarchical_arpc \\
    --output cell_u_f50_arpc.json

Aggregate Δ(M−U) after both arms finished (gap CIs + cutoff on CI bounds):
  python tools/gap_test2_prefix_oracle.py summarize --cells cell_*.json --output curve.json

Pre-flight: run f=0% hierarchical_arpc on both arms first (must hit 55.6/33.4).
``build-prefixes`` drops problems too short for a real 75% cut so the set is
fixed across all f. Both arms load the same ``prefix_ids`` into
``sampler.generate`` — clean-stream half is identical by construction.

AR JSONL fields (one object per line):
  {"problem_id": "...", "question": "...", "solution_text": "...",
   "gold_number": "42", "correct": true}
``solution_text`` must be the AR teacher's completion body (not dataset gold).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


_NUM = re.compile(r'(-?\d+(?:\.\d+)?)')
_BOX = re.compile(r'\\boxed\{([^}]*)\}')


def _repo_src() -> None:
  root = Path(__file__).resolve().parents[1]
  src = str(root / 'src')
  if src not in sys.path:
    sys.path.insert(0, src)


def _extract_num(text: str) -> str | None:
  m = _BOX.search(text)
  if m:
    nums = _NUM.findall(m.group(1).replace(',', ''))
    return nums[-1] if nums else m.group(1).strip()
  nums = _NUM.findall(text.replace(',', ''))
  return nums[-1] if nums else None


def _load_model(ckpt: Path, device: torch.device, profile: str):
  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.evaluations.decode_profiles import profile_overrides
  from discrete_diffusion.sampling.block_sampler import BlockSampler
  from discrete_diffusion.train import register_config_resolvers
  from omegaconf import OmegaConf

  register_config_resolvers()
  overrides = profile_overrides(profile)
  model, config, tokenizer = load_block_trainer_checkpoint(
      ckpt, device, hydra_overrides=overrides)
  OmegaConf.resolve(config)
  model.eval()
  sampler = BlockSampler(config)
  return model, config, tokenizer, sampler


def _chat_prefix_ids(tokenizer, question: str, device) -> torch.Tensor:
  from discrete_diffusion.data.conversion_baseline import (
      apply_conversion_chat_template,
  )
  text = apply_conversion_chat_template(
      tokenizer,
      [{'role': 'user', 'content': str(question)}],
      add_generation_prompt=True,
      tokenize=False,
  )
  ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids']
  return ids.to(device)


def cmd_build_prefixes(args: argparse.Namespace) -> int:
  _repo_src()
  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.train import register_config_resolvers

  register_config_resolvers()
  device = torch.device('cpu')
  _, _, tokenizer = load_block_trainer_checkpoint(
      Path(args.tokenizer_from_ckpt), device)
  bs = int(args.block_size)
  fracs = [float(x) for x in args.fractions.split(',') if x.strip()]
  need_f75 = any(abs(f - 0.75) < 1e-9 for f in fracs)
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  n_keep = 0
  n_drop_short = 0
  with out.open('w', encoding='utf-8') as fout, Path(args.ar_jsonl).open(
      encoding='utf-8') as fin:
    for line in fin:
      if not line.strip():
        continue
      ex = json.loads(line)
      if not ex.get('correct', True):
        continue
      q = ex['question']
      sol = ex['solution_text']
      gold = str(ex.get('gold_number') or _extract_num(sol) or '')
      sol_ids = tokenizer(
          sol, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
      prompt_ids = _chat_prefix_ids(tokenizer, q, device)[0]
      # Fixed set: must support a real 75% cut (block-aligned) before the
      # last solution block — otherwise drop so f-mix stays constant.
      max_cut = ((len(sol_ids) - 1) // bs) * bs
      n_tok_75 = (int(len(sol_ids) * 0.75) // bs) * bs
      n_tok_75 = min(n_tok_75, max_cut)
      if need_f75 and n_tok_75 < bs:
        n_drop_short += 1
        continue
      if len(sol_ids) < bs:
        n_drop_short += 1
        continue
      cuts = {}
      for f in fracs:
        n_tok = int(len(sol_ids) * f)
        n_tok = (n_tok // bs) * bs
        n_tok = min(n_tok, max_cut)
        if f > 0 and n_tok < bs:
          # Should not happen after f75 filter for f in {0.25,0.5,0.75}
          n_tok = 0
        prefix = torch.cat([prompt_ids, sol_ids[:n_tok]], dim=0)
        prefix_text = tokenizer.decode(prefix, skip_special_tokens=False)
        leak = bool(gold and gold in prefix_text and f < 1.0)
        cuts[str(f)] = {
            'n_solution_tokens': int(n_tok),
            'prefix_len': int(prefix.numel()),
            'prefix_ids': prefix.tolist(),
            'possible_gold_in_prefix': leak,
        }
      # Verify all fractions present with consistent problem
      if need_f75 and cuts.get('0.75', {}).get('n_solution_tokens', 0) < bs:
        n_drop_short += 1
        continue
      rec = {
          'problem_id': ex.get('problem_id'),
          'question': q,
          'gold_number': gold,
          'solution_len': int(sol_ids.numel()),
          'prompt_len': int(prompt_ids.numel()),
          'block_size': bs,
          'cuts': cuts,
          'fixed_set': True,
      }
      fout.write(json.dumps(rec) + '\n')
      n_keep += 1
  meta = {
      'n_keep': n_keep,
      'n_drop_short_for_f75': n_drop_short,
      'fractions': fracs,
      'block_size': bs,
      'note': 'Same problems for all f; short sols dropped before write.',
  }
  Path(str(out) + '.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
  print(f'Wrote {n_keep} prefix records → {out} (dropped_short={n_drop_short})')
  return 0


@torch.no_grad()
def cmd_run(args: argparse.Namespace) -> int:
  _repo_src()
  from discrete_diffusion.evaluations.code_fingerprint import (
      assert_forward_process_utils_ok,
      code_fingerprint_header,
  )
  code_fp = assert_forward_process_utils_ok(require_expected_sha=False)
  print(
      f"code_fingerprint utils.py sha256={code_fp['sha256']} "
      f"n_lines={code_fp['n_lines']} ok={code_fp['ok']}",
      flush=True)
  device = torch.device(
      args.device if args.device != 'cuda' or torch.cuda.is_available()
      else 'cpu')
  model, config, tokenizer, sampler = _load_model(
      Path(args.ckpt), device, args.decode_profile)
  frac = str(float(args.fraction))
  # normalize key
  records = []
  with Path(args.prefixes).open(encoding='utf-8') as f:
    for line in f:
      if line.strip():
        records.append(json.loads(line))
  # map fraction keys
  def cut_for(rec):
    cuts = rec['cuts']
    if frac in cuts:
      return cuts[frac]
    # fuzzy
    for k, v in cuts.items():
      if abs(float(k) - float(frac)) < 1e-6:
        return v
    raise KeyError(f'fraction {frac} missing in record')

  results = []
  max_new = int(args.max_new_tokens)
  steps = int(args.num_steps or getattr(config.sampling, 'steps', 32))
  for i, rec in enumerate(records):
    if args.limit and i >= int(args.limit):
      break
    cut = cut_for(rec)
    prefix = torch.tensor(cut['prefix_ids'], dtype=torch.long, device=device)
    if prefix.numel() >= int(model.num_tokens) - 1:
      results.append({
          'problem_id': rec.get('problem_id'), 'error': 'prefix_too_long'})
      continue
    samples = sampler.generate(
        model,
        num_samples=1,
        num_steps=steps,
        eps=None,
        inject_bos=False,
        prefix_ids=prefix.unsqueeze(0),
        max_new_tokens=max_new,
        greedy=bool(getattr(sampler, '_greedy_decode', False)),
    )
    plen = int(prefix.numel())
    cont = samples[0, plen:plen + max_new]
    eos = tokenizer.eos_token_id
    if eos is not None:
      hits = (cont == eos).nonzero(as_tuple=False)
      if hits.numel():
        cont = cont[: int(hits[0]) + 1]
    text = tokenizer.decode(cont, skip_special_tokens=True)
    pred = _extract_num(text)
    gold = str(rec.get('gold_number') or '')
    nfe = getattr(sampler, 'last_nfe_stats', None)
    results.append({
        'problem_id': rec.get('problem_id'),
        'fraction': float(frac),
        'prefix_len': plen,
        'n_solution_tokens_in_prefix': cut['n_solution_tokens'],
        'pred': pred,
        'gold': gold,
        'correct': (pred is not None and gold != '' and pred == gold),
        'response_tail': text[-400:],
        'nfe': nfe,
        'possible_gold_in_prefix': cut.get('possible_gold_in_prefix'),
    })
    if (i + 1) % 10 == 0:
      print(f'… {i+1}/{len(records)}', flush=True)

  n = sum(1 for r in results if 'correct' in r)
  n_ok = sum(1 for r in results if r.get('correct'))
  gsm = None if n == 0 else n_ok / n
  # Per-problem correctness for paired gap bootstrap at summarize time
  doc = {
      'protocol': 'GAP_DIAGNOSIS_M_U_2026-10-01 Test 2',
      'code_fingerprint': code_fingerprint_header(),
      'ckpt': str(args.ckpt),
      'decode_profile': args.decode_profile,
      'fraction': float(frac),
      'gsm_exact': gsm,
      'n': n,
      'n_correct': n_ok,
      'anchor': None,
      'results': results,
  }
  # f=0% must reproduce hard-twin ARPC within band (run this cell first).
  if abs(float(frac) - 0.0) < 1e-9 and args.decode_profile == 'hierarchical_arpc':
    expect = None
    if 'masked_2020048' in str(args.ckpt) or '/ar2block_masked_2020048/' in str(
        args.ckpt):
      expect = 0.556
    elif 'uniform_2020049' in str(args.ckpt) or '/ar2block_uniform_2020049/' in str(
        args.ckpt):
      expect = 0.334
    if expect is not None and gsm is not None:
      delta_pp = 100.0 * (gsm - expect)
      ok = abs(delta_pp) <= float(args.anchor_tol_pp)
      doc['anchor'] = {
          'expected_gsm': expect,
          'got_gsm': gsm,
          'delta_pp': delta_pp,
          'tol_pp': float(args.anchor_tol_pp),
          'pass': ok,
          'note': (
              'f=0% must match existing harness within tol before other f'),
      }
      if not ok:
        print(
            f'ANCHOR FAIL: GSM={100*gsm:.1f}% expected={100*expect:.1f}% '
            f'(Δ={delta_pp:+.1f}pp > ±{args.anchor_tol_pp}pp). '
            'Fix runner before other fractions.',
            flush=True)
      else:
        print(
            f'ANCHOR OK: GSM={100*gsm:.1f}% vs {100*expect:.1f}% '
            f'(Δ={delta_pp:+.1f}pp)',
            flush=True)
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(doc, indent=2) + '\n')
  print(f"Wrote {out}  GSM={doc['gsm_exact']} ({n_ok}/{n})")
  return 0


def cmd_summarize(args: argparse.Namespace) -> int:
  cells = []
  for path in args.cells:
    cells.append(json.loads(Path(path).read_text()))
  from collections import defaultdict
  # (profile, frac) -> arm -> cell
  by = defaultdict(dict)
  for c in cells:
    ckpt = c['ckpt']
    arm = 'masked' if 'masked' in ckpt else (
        'uniform' if 'uniform' in ckpt else 'unknown')
    key = (c['decode_profile'], float(c['fraction']))
    by[key][arm] = c

  def _gap_ci(m_cell, u_cell, n_boot=400, seed=0):
    """Paired bootstrap on problem_id intersection."""
    if not m_cell or not u_cell:
      return None
    m_map = {
        r['problem_id']: 1.0 if r.get('correct') else 0.0
        for r in m_cell.get('results') or []
        if 'correct' in r and r.get('problem_id') is not None}
    u_map = {
        r['problem_id']: 1.0 if r.get('correct') else 0.0
        for r in u_cell.get('results') or []
        if 'correct' in r and r.get('problem_id') is not None}
    ids = sorted(set(m_map) & set(u_map))
    if len(ids) < 10:
      # fall back to mean difference without pair
      m = m_cell.get('gsm_exact')
      u = u_cell.get('gsm_exact')
      if m is None or u is None:
        return None
      return {
          'mean_pp': 100 * (m - u), 'lo_pp': None, 'hi_pp': None,
          'n_paired': 0, 'note': 'unpaired fallback'}
    import torch
    mv = torch.tensor([m_map[i] for i in ids], dtype=torch.float64)
    uv = torch.tensor([u_map[i] for i in ids], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    diffs = []
    n = len(ids)
    for _ in range(n_boot):
      idx = torch.randint(0, n, (n,), generator=g)
      diffs.append(float((mv[idx] - uv[idx]).mean() * 100))
    diffs.sort()
    return {
        'mean_pp': float((mv - uv).mean() * 100),
        'lo_pp': diffs[int(0.025 * (n_boot - 1))],
        'hi_pp': diffs[int(0.975 * (n_boot - 1))],
        'n_paired': n,
    }

  rows = []
  for (profile, frac), arms in sorted(by.items()):
    m = arms.get('masked')
    u = arms.get('uniform')
    gap = _gap_ci(m, u)
    rows.append({
        'profile': profile,
        'fraction': frac,
        'gsm_masked': None if not m else 100 * m.get('gsm_exact'),
        'gsm_uniform': None if not u else 100 * u.get('gsm_exact'),
        'gap': gap,
        'anchor_masked': None if not m else m.get('anchor'),
        'anchor_uniform': None if not u else u.get('anchor'),
    })

  # Cutoffs on CI upper bound for exposure (gap@75), lower bound for denoiser
  verdict = None
  verdict_detail = None
  arpc = [r for r in rows if r['profile'] == 'hierarchical_arpc']
  g0 = next((r['gap'] for r in arpc if r['fraction'] == 0.0), None)
  g75 = next((r['gap'] for r in arpc if r['fraction'] == 0.75), None)
  if g0 and g75 and g0.get('hi_pp') is not None and g75.get('hi_pp') is not None:
    # Exposure: gap@75 CI upper ≤ 5 AND gap@0 CI lower ≥ 15
    if g75['hi_pp'] <= 5 and g0['lo_pp'] >= 15:
      verdict = 'exposure-dominated'
    elif g75['lo_pp'] >= 15:
      verdict = 'denoiser-dominated'
    else:
      verdict = 'mixed'
    verdict_detail = {
        'gap0_pp': g0, 'gap75_pp': g75,
        'rule': (
            'exposure if gap75.hi≤5 and gap0.lo≥15; '
            'denoiser if gap75.lo≥15; else mixed'),
    }

  doc = {
      'protocol': 'GAP_DIAGNOSIS_M_U_2026-10-01 Test 2 summarize',
      'cutoffs': {
          'exposure': 'gap@75% CI upper ≤5pp AND gap@0% CI lower ≥15pp',
          'denoiser': 'gap@75% CI lower ≥15pp',
          'mixed': 'else',
          'noise_band_note': (
              'Apply cutoffs to bootstrap CIs on the paired gap, '
              'not point estimates alone'),
      },
      'verdict_hierarchical_arpc': verdict,
      'verdict_detail': verdict_detail,
      'rows': rows,
  }
  out = Path(args.output)
  out.write_text(json.dumps(doc, indent=2) + '\n')
  print(json.dumps({
      'verdict': verdict,
      'verdict_detail': verdict_detail,
      'rows': [
          {k: r[k] for k in (
              'profile', 'fraction', 'gsm_masked', 'gsm_uniform', 'gap')}
          for r in rows],
  }, indent=2))
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  sp = ap.add_subparsers(dest='cmd', required=True)

  b = sp.add_parser('build-prefixes')
  b.add_argument('--ar-jsonl', required=True)
  b.add_argument('--tokenizer-from-ckpt', required=True)
  b.add_argument('--block-size', type=int, default=32)
  b.add_argument('--fractions', default='0,0.25,0.5,0.75')
  b.add_argument('--output', required=True)
  b.set_defaults(func=cmd_build_prefixes)

  r = sp.add_parser('run')
  r.add_argument('--ckpt', required=True)
  r.add_argument('--prefixes', required=True)
  r.add_argument('--fraction', type=float, required=True)
  r.add_argument('--decode-profile', default='hierarchical_arpc')
  r.add_argument('--max-new-tokens', type=int, default=1024)
  r.add_argument('--num-steps', type=int, default=None)
  r.add_argument('--limit', type=int, default=0)
  r.add_argument('--device', default='cuda')
  r.add_argument('--anchor-tol-pp', type=float, default=2.0,
                 help='f=0%% hierarchical_arpc must match 55.6/33.4 within this')
  r.add_argument('--output', required=True)
  r.set_defaults(func=cmd_run)

  s = sp.add_parser('summarize')
  s.add_argument('--cells', nargs='+', required=True)
  s.add_argument('--output', required=True)
  s.set_defaults(func=cmd_summarize)

  args = ap.parse_args(argv)
  return int(args.func(args) or 0)


if __name__ == '__main__':
  raise SystemExit(main())
