#!/usr/bin/env python3
"""Test 3 — denoiser quality at matched α_t (gap diagnosis M−U).

Forward-only. Isolates the denoiser under GT clean-stream (N2C on).

Pre-flight (CPU, no ckpt):
  python tools/gap_test3_denoiser_quality.py self-check

GPU:
  python tools/gap_test3_denoiser_quality.py run \\
    --masked-ckpt .../ar2block_masked_2020048/checkpoints/last.ckpt \\
    --uniform-ckpt .../ar2block_uniform_2020049/checkpoints/last.ckpt \\
    --device cuda --max-blocks 1000 --output out/gap_test3.json

Corrupted sites are always ``xt != x0`` inside the active block (Unif collisions
where a redraw equals x0 are logged and excluded from the corrupt set).
Masked detection AUROC is N/A (MASK is visible).

See docs/research/GAP_DIAGNOSIS_M_U_2026-10-01.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_ALPHAS = (
    1.0, 0.999, 0.99, 0.98, 0.95, 0.90, 0.70, 0.50, 0.30, 0.10)


def _repo_src() -> None:
  root = Path(__file__).resolve().parents[1]
  src = str(root / 'src')
  if src not in sys.path:
    sys.path.insert(0, src)


def _t_from_alpha(alpha: float, eps: float) -> float:
  """LogLinear: α = 1 − (1−ε)t  →  t = (1−α)/(1−ε)."""
  return float((1.0 - alpha) / max(1.0 - eps, 1e-8))


def _alpha_from_t(t: float, eps: float) -> float:
  return float(1.0 - (1.0 - eps) * t)


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
  s = scores.detach().float().reshape(-1).cpu()
  y = labels.detach().bool().reshape(-1).cpu()
  pos = s[y]
  neg = s[~y]
  n_pos = int(pos.numel())
  n_neg = int(neg.numel())
  if n_pos == 0 or n_neg == 0:
    return None
  total = 0.0
  chunk = 2048
  neg_sorted, _ = torch.sort(neg)
  for i in range(0, n_pos, chunk):
    p = pos[i:i + chunk]
    less = torch.searchsorted(neg_sorted, p, right=False).to(torch.float64)
    more_or_eq = torch.searchsorted(neg_sorted, p, right=True).to(torch.float64)
    equal = more_or_eq - less
    total += float((less + 0.5 * equal).sum().item())
  return total / (n_pos * n_neg)


def _bootstrap_mean(xs: list[float], n_boot: int = 200, seed: int = 0) -> dict:
  if not xs:
    return {'mean': None, 'lo': None, 'hi': None, 'n': 0}
  g = torch.Generator().manual_seed(seed)
  t = torch.tensor(xs, dtype=torch.float64)
  n = t.numel()
  means = []
  for _ in range(n_boot):
    idx = torch.randint(0, n, (n,), generator=g)
    means.append(float(t[idx].mean()))
  means.sort()
  return {
      'mean': float(t.mean()),
      'lo': means[int(0.025 * (n_boot - 1))],
      'hi': means[int(0.975 * (n_boot - 1))],
      'n': int(n),
  }


def _bootstrap_paired_diff(
    a: list[float], b: list[float], n_boot: int = 200, seed: int = 1,
) -> dict:
  """Bootstrap CI on mean(a)−mean(b) with paired indices."""
  if not a or not b or len(a) != len(b):
    return {'mean': None, 'lo': None, 'hi': None, 'n': 0}
  g = torch.Generator().manual_seed(seed)
  ta = torch.tensor(a, dtype=torch.float64)
  tb = torch.tensor(b, dtype=torch.float64)
  n = ta.numel()
  diffs = []
  for _ in range(n_boot):
    idx = torch.randint(0, n, (n,), generator=g)
    diffs.append(float((ta[idx] - tb[idx]).mean()))
  diffs.sort()
  return {
      'mean': float((ta - tb).mean()),
      'lo': diffs[int(0.025 * (n_boot - 1))],
      'hi': diffs[int(0.975 * (n_boot - 1))],
      'n': int(n),
  }


def _noise_level_split(u_levels: list[dict], alphas: list[float]) -> dict:
  """Cause-6 probe: Unif at high α (low noise) vs low α (high noise)."""
  by_a = {float(lv['alpha_t']): lv for lv in u_levels}

  def _mean(lv, key):
    if lv is None:
      return None
    v = lv.get(key)
    if v is None:
      return None
    if isinstance(v, dict):
      return v.get('mean')
    return v

  high = [a for a in alphas if a >= 0.90]
  low = [a for a in alphas if a <= 0.30]
  high_clean = [x for x in (_mean(by_a.get(a), 'clean_acc') for a in high)
                if x is not None]
  low_clean = [x for x in (_mean(by_a.get(a), 'clean_acc') for a in low)
               if x is not None]
  high_auc = [x for x in (_mean(by_a.get(a), 'detect_auroc') for a in high)
              if x is not None]
  low_auc = [x for x in (_mean(by_a.get(a), 'detect_auroc') for a in low)
             if x is not None]

  def avg(xs):
    return None if not xs else sum(xs) / len(xs)

  hc, lc, ha, la = avg(high_clean), avg(low_clean), avg(high_auc), avg(low_auc)
  pattern = 'insufficient'
  if hc is not None and lc is not None:
    d_clean = hc - lc
    d_auc = None if ha is None or la is None else ha - la
    if d_clean < -0.05 or (d_auc is not None and d_auc < -0.05):
      pattern = 'worse_at_low_noise'  # supports cause 6
    elif abs(d_clean) < 0.03 and (d_auc is None or abs(d_auc) < 0.03):
      pattern = 'flat_across_alpha'
    elif d_clean > 0.05:
      pattern = 'better_at_low_noise'
    else:
      pattern = 'mixed'
  return {
      'high_alpha': high,
      'low_alpha': low,
      'clean_acc_high_alpha': hc,
      'clean_acc_low_alpha': lc,
      'detect_auroc_high_alpha': ha,
      'detect_auroc_low_alpha': la,
      'pattern': pattern,
      'note': (
          'worse_at_low_noise → Unif struggles when most tokens are right '
          '(fits no-t); flat_across_alpha → ignores noise level'),
  }


def _metrics_from_pred(
    pred: torch.Tensor,
    log_p: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    corrupted: torch.Tensor,
    cmask: torch.Tensor,
    *,
    report_auroc: bool,
    report_clean_acc: bool,
    exclude_ids: tuple[int, ...] | list[int] | None = None,
) -> dict:
  clean_sites = cmask & ~corrupted
  row: dict = {
      'n_corrupted': int(corrupted.sum().item()),
      'n_clean_in_block': int(clean_sites.sum().item()),
  }
  if corrupted.any():
    lp = log_p[0][corrupted[0]]
    tgt = x0[corrupted]
    row['corrupt_ce'] = float(F.nll_loss(lp, tgt, reduction='mean').item())
    row['corrupt_acc'] = float(
        (pred[corrupted] == x0[corrupted]).float().mean().item())
  else:
    row['corrupt_ce'] = None
    row['corrupt_acc'] = None
  if report_clean_acc and clean_sites.any():
    # Soft metrics on content sites only — banned specials as x0 make CE explode.
    p = log_p[0].exp()
    p_x0 = p.gather(-1, x0[0].unsqueeze(-1)).squeeze(-1)
    p_xt = p.gather(-1, xt[0].unsqueeze(-1)).squeeze(-1)
    ranks = (p > p_x0.unsqueeze(-1)).sum(-1) + 1
    cl = clean_sites[0]
    if exclude_ids:
      content = cl.clone()
      for mid in exclude_ids:
        content = content & (x0[0] != int(mid))
    else:
      content = cl
    if content.any():
      lp = log_p[0][content]
      tgt = x0[0, content]
      row['clean_ce'] = float(F.nll_loss(lp, tgt, reduction='mean').item())
      row['clean_acc'] = float(
          (pred[0, content] == x0[0, content]).float().mean().item())
      row['clean_copy_xt_acc'] = float(
          (pred[0, content] == xt[0, content]).float().mean().item())
      row['clean_mean_p_x0'] = float(p_x0[content].mean().item())
      row['clean_mean_p_xt'] = float(p_xt[content].mean().item())
      row['clean_median_rank_x0'] = float(ranks[content].float().median().item())
      row['clean_mean_rank_x0'] = float(ranks[content].float().mean().item())
      row['n_clean_content'] = int(content.sum().item())
    else:
      row['clean_ce'] = None
      row['clean_acc'] = None
      row['clean_copy_xt_acc'] = None
      row['clean_mean_p_x0'] = None
      row['clean_mean_p_xt'] = None
      row['clean_median_rank_x0'] = None
      row['clean_mean_rank_x0'] = None
      row['n_clean_content'] = 0
  else:
    row['clean_ce'] = None
    row['clean_acc'] = None
    row['clean_copy_xt_acc'] = None
    row['clean_mean_p_x0'] = None
    row['clean_mean_p_xt'] = None
    row['clean_median_rank_x0'] = None
    row['clean_mean_rank_x0'] = None
    row['n_clean_content'] = None
  if report_auroc and cmask.any() and corrupted.any() and clean_sites.any():
    p_xt = log_p[0].exp().gather(-1, xt[0].unsqueeze(-1)).squeeze(-1)
    scores = (1.0 - p_xt)[cmask[0]]
    labels = corrupted[0][cmask[0]]
    row['detect_auroc'] = _auroc(scores, labels)
  else:
    row['detect_auroc'] = None  # masked → N/A; or degenerate split
  return row


def self_check() -> int:
  """Oracle → acc=1/AUROC=1; uniform-random → acc≈1/V, AUROC≈0.5.

  Also checks **clean-site** metrics: oracle clean_acc=1, random ≈1/V.
  """
  torch.manual_seed(0)
  V, L = 50, 64
  x0 = torch.randint(0, V, (1, L))
  # Active block has both corrupt and clean sites (needed for AUROC).
  cmask = torch.zeros(1, L, dtype=torch.bool)
  cmask[:, 16:48] = True
  xt = x0.clone()
  xt[:, 16:32] = (x0[:, 16:32] + 7) % V  # corrupt half of block
  corrupted = (xt != x0) & cmask
  assert int(corrupted.sum()) == 16
  assert int((cmask & ~corrupted).sum()) == 16

  # Oracle: logits peak at x0
  logits = torch.full((1, L, V), -20.0)
  for i in range(L):
    logits[0, i, int(x0[0, i])] = 10.0
  log_p = F.log_softmax(logits, dim=-1)
  pred = log_p.argmax(-1)
  oracle = _metrics_from_pred(
      pred, log_p, x0, xt, corrupted, cmask,
      report_auroc=True, report_clean_acc=True, exclude_ids=())
  assert oracle['corrupt_acc'] == 1.0, oracle
  assert oracle['clean_acc'] == 1.0, oracle
  assert oracle['clean_copy_xt_acc'] == 1.0, oracle
  assert oracle['detect_auroc'] is not None and oracle['detect_auroc'] > 0.99, oracle

  # Random predictor
  flat = torch.zeros(1, L, V)
  log_p_r = F.log_softmax(flat, dim=-1)
  rnd = _metrics_from_pred(
      torch.randint(0, V, (1, L)), log_p_r, x0, xt, corrupted, cmask,
      report_auroc=True, report_clean_acc=True, exclude_ids=())
  aucs, accs_c, accs_cl = [], [], []
  for s in range(40):
    g = torch.Generator().manual_seed(s)
    pred_i = torch.randint(0, V, (1, L), generator=g)
    scores = torch.rand(L, generator=g)
    aucs.append(_auroc(scores[cmask[0]], corrupted[0][cmask[0]]))
    accs_c.append(float((pred_i[corrupted] == x0[corrupted]).float().mean()))
    clean = cmask & ~corrupted
    accs_cl.append(float((pred_i[clean] == x0[clean]).float().mean()))
  mean_auc = sum(aucs) / len(aucs)
  mean_acc = sum(accs_c) / len(accs_c)
  mean_clean = sum(accs_cl) / len(accs_cl)
  assert 0.35 < mean_auc < 0.65, mean_auc
  assert abs(mean_acc - 1.0 / V) < 0.05, mean_acc
  assert abs(mean_clean - 1.0 / V) < 0.05, mean_clean
  assert rnd['detect_auroc'] is None or 0.35 < rnd['detect_auroc'] < 0.65, rnd

  # Collision drop: xt==x0 must not enter corrupt set
  xt_c = x0.clone()
  xt_c[:, 16:32] = (x0[:, 16:32] + 1) % V
  xt_c[:, 20] = x0[:, 20]
  corr_c = (xt_c != x0) & cmask
  assert not bool(corr_c[0, 20].item())
  assert int(corr_c[:, 16:32].sum()) == 15

  # Masked path: clean_acc must be N/A
  masked_row = _metrics_from_pred(
      pred, log_p, x0, xt, corrupted, cmask,
      report_auroc=False, report_clean_acc=False, exclude_ids=())
  assert masked_row['clean_acc'] is None
  assert masked_row['detect_auroc'] is None

  eps = 1e-3
  for a in DEFAULT_ALPHAS:
    t = _t_from_alpha(a, eps)
    assert abs(_alpha_from_t(t, eps) - a) < 1e-9

  print('self-check PASS: oracle corrupt+clean acc/AUROC≈1; '
        'random ≈1/V / AUROC≈0.5; masked clean=N/A; collision drop OK')
  print(f'  oracle={oracle}')
  print(f'  random_mean_auc={mean_auc:.3f} corrupt_acc={mean_acc:.4f} '
        f'clean_acc={mean_clean:.4f}')
  return 0


def _load(ckpt: Path, device: torch.device):
  from discrete_diffusion.evaluations.checkpoint_utils import (
      load_block_trainer_checkpoint,
  )
  from discrete_diffusion.train import register_config_resolvers
  register_config_resolvers()
  model, config, tokenizer = load_block_trainer_checkpoint(ckpt, device)
  model.eval()
  return model, config, tokenizer


def _simplex_info(model) -> dict:
  from discrete_diffusion.forward_process.utils import (
      normalize_uniform_simplex_mode,
      resolve_uniform_exclude_ids,
      uniform_simplex_size,
  )
  mode = normalize_uniform_simplex_mode(
      getattr(model, 'uniform_simplex_mode', 'conversion'))
  v = int(model.vocab_size)
  mid = getattr(model, 'mask_id', None)
  exclude = ()
  v_eff = None
  if model.forward_process_name in ('uniform', 'hybrid'):
    exclude = tuple(resolve_uniform_exclude_ids(
        model.tokenizer, mask_id=mid, vocab_size=v, mode=mode))
    v_eff = int(uniform_simplex_size(v, None, exclude_ids=exclude))
  sched = type(getattr(model, 'noise', None)).__name__
  eps = float(getattr(model.noise, 'eps', 1e-3))
  return {
      'forward_process_name': model.forward_process_name,
      'uniform_simplex_mode': mode,
      'v_eff': v_eff,
      'vocab_size': v,
      'n_exclude': len(exclude),
      'exclude_ids_head': list(exclude[:8]),
      'mask_id': int(mid) if mid is not None else -1,
      'block_size': int(model.block_size),
      'noise_schedule': sched,
      'eps': eps,
      'shift_loss_targets': bool(getattr(model, 'shift_loss_targets', False)),
      'time_conditioning': bool(getattr(model, 'time_conditioning', False)),
      'alpha_map': {
          str(a): _t_from_alpha(a, eps) for a in DEFAULT_ALPHAS
      },
  }


def _ban_exclude_logits(model, logits: torch.Tensor) -> torch.Tensor:
  """Match train/decode: ban reserved specials from the predictive simplex."""
  if model.forward_process_name not in ('uniform', 'hybrid'):
    # Masked: still ban MASK/PAD as decode does
    out = logits.clone()
    neg = float(getattr(model, 'neg_infinity', -1e6))
    if getattr(model, 'mask_id', None) is not None:
      out[..., int(model.mask_id)] = neg
    pad = getattr(getattr(model, 'tokenizer', None), 'pad_token_id', None)
    if pad is not None:
      out[..., int(pad)] = float('-inf')
    return out
  from discrete_diffusion.forward_process.utils import (
      normalize_uniform_simplex_mode,
      resolve_uniform_exclude_ids,
  )
  mode = normalize_uniform_simplex_mode(
      getattr(model, 'uniform_simplex_mode', 'conversion'))
  exclude = resolve_uniform_exclude_ids(
      model.tokenizer, mask_id=model.mask_id,
      vocab_size=int(model.vocab_size), mode=mode)
  out = logits.clone()
  neg = float(getattr(model, 'neg_infinity', -1e6))
  for mid in exclude:
    if 0 <= int(mid) < out.size(-1):
      out[..., int(mid)] = neg
  return out


def _v_eff_for(model) -> int:
  from discrete_diffusion.forward_process.utils import (
      normalize_uniform_simplex_mode,
      resolve_uniform_exclude_ids,
      uniform_simplex_size,
  )
  mode = normalize_uniform_simplex_mode(
      getattr(model, 'uniform_simplex_mode', 'conversion'))
  exclude = resolve_uniform_exclude_ids(
      model.tokenizer, mask_id=model.mask_id,
      vocab_size=int(model.vocab_size), mode=mode)
  return int(uniform_simplex_size(
      int(model.vocab_size), None, exclude_ids=exclude))


def _retention_from_log_p(
    log_p: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    clean: torch.Tensor,
    *,
    alpha: float,
    temps: tuple[float, ...],
    v_eff: int,
) -> dict:
  """Noise-removal keep-xt from p_θ — same algebra as sampler fast path."""
  if not clean.any():
    return {}
  out: dict = {}
  a_t = float(alpha)
  cl = clean[0]
  idxs = cl.nonzero(as_tuple=False).flatten()
  for temp in temps:
    tag = f'T{temp:g}'.replace('.', 'p')
    p = log_p[0].exp()
    if abs(temp - 1.0) > 1e-8:
      p = p.clamp(min=1e-12).pow(1.0 / float(temp))
      p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
    p_xt = p.gather(-1, xt[0].unsqueeze(-1)).squeeze(-1)
    denom = (a_t * v_eff * p_xt + (1.0 - a_t)).clamp(min=1e-12)
    keep_prob = (a_t * v_eff * p_xt / denom).clamp(0.0, 1.0)
    soft = float(keep_prob[cl].mean().item())
    thr = torch.rand(idxs.numel(), device=log_p.device)
    xs = xt[0].clone()
    for j, i in enumerate(idxs.tolist()):
      if thr[j] >= keep_prob[i]:
        xs[i] = torch.multinomial(p[i], 1).item()
    hard_keep = float((xs[cl] == xt[0, cl]).float().mean().item())
    still_x0 = float((xs[cl] == x0[0, cl]).float().mean().item())
    out[f'retention_soft_keep_xt_{tag}'] = soft
    out[f'retention_hard_keep_xt_{tag}'] = hard_keep
    out[f'retention_still_x0_{tag}'] = still_x0
  return out


@torch.no_grad()
def _eval_level(
    model,
    batches: list[torch.Tensor],
    *,
    alpha: float,
    eps: float,
    block_size: int,
    active_block: int,
) -> tuple[dict, list[dict]]:
  """Returns (aggregate, per_problem rows) for paired gap bootstrap."""
  t_val = _t_from_alpha(alpha, eps)
  is_masked = model.forward_process_name == 'masked'
  per_prob = []
  n_attempted = 0
  n_collision = 0  # move_mask & (xt==x0): redraw landed on original
  n_kept = 0       # ~move_mask inside block
  n_xt_ne_x0 = 0
  for x0 in batches:
    assert x0.shape[0] == 1
    seq = x0.shape[1]
    n_blocks = seq // block_size
    blk = min(max(int(active_block), 0), n_blocks - 1)
    if n_blocks > 1 and blk == 0:
      blk = 1
    start, end = blk * block_size, (blk + 1) * block_size
    t = torch.full((1, seq), t_val, device=x0.device, dtype=torch.float32)
    cmask = torch.zeros(1, seq, dtype=torch.bool, device=x0.device)
    cmask[:, start:end] = True
    xt, move = model._corrupt(
        x0, t, block_size=block_size, return_move_mask=True,
        corruption_mask=cmask)
    # Corrupt set for metrics: always xt != x0 (excludes Unif 1/V_eff collisions).
    corrupted = (xt != x0) & cmask
    if move is not None:
      attempted = move.bool() & cmask
      collision = attempted & (xt == x0)
      kept = (~move.bool()) & cmask
    elif is_masked:
      attempted = (xt == model.mask_id) & cmask
      collision = torch.zeros_like(cmask)
      kept = cmask & ~attempted
    else:
      # Fallback if FP has no move_mask: cannot separate keep vs collision.
      attempted = cmask
      collision = torch.zeros_like(cmask)
      kept = cmask & ~corrupted
    n_attempted += int(attempted.sum().item())
    n_collision += int(collision.sum().item())
    n_kept += int(kept.sum().item())
    n_xt_ne_x0 += int(corrupted.sum().item())

    logits = model._backbone_logits(xt, x0, block_size=block_size)
    if logits.size(1) < seq:
      pad = torch.zeros(
          1, seq - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    logits = _ban_exclude_logits(model, logits)
    log_p = F.log_softmax(logits.float(), dim=-1)
    pred = log_p.argmax(dim=-1)
    excl = ()
    if not is_masked:
      from discrete_diffusion.forward_process.utils import (
          normalize_uniform_simplex_mode, resolve_uniform_exclude_ids,
      )
      mode = normalize_uniform_simplex_mode(
          getattr(model, 'uniform_simplex_mode', 'conversion'))
      excl = tuple(resolve_uniform_exclude_ids(
          model.tokenizer, mask_id=model.mask_id,
          vocab_size=int(model.vocab_size), mode=mode))
    row = _metrics_from_pred(
        pred, log_p, x0, xt, corrupted, cmask,
        report_auroc=not is_masked,
        report_clean_acc=not is_masked,
        exclude_ids=excl)
    # One-step Unif retention from the *same* forward (noise-removal keep-xt).
    if not is_masked:
      clean_m = cmask & ~corrupted
      if clean_m.any():
        row.update(_retention_from_log_p(
            log_p, x0, xt, clean_m, alpha=alpha,
            temps=(1.0, 0.1), v_eff=_v_eff_for(model)))
    row.update({
        'alpha_t': alpha,
        't_schedule': t_val,
        'block': blk,
        'n_attempted_moves': int(attempted.sum().item()),
        'n_collision_xt_eq_x0': int(collision.sum().item()),
        'n_kept_no_move': int(kept.sum().item()),
        'n_xt_ne_x0': int(corrupted.sum().item()),
        'move_mask_available': move is not None,
    })
    per_prob.append(row)

  def _boot(key):
    xs = [p[key] for p in per_prob if p.get(key) is not None]
    return _bootstrap_mean(xs)

  agg = {
      'alpha_t': alpha,
      't_schedule': t_val,
      'n_problems': len(per_prob),
      'corrupt_acc': _boot('corrupt_acc'),
      'corrupt_ce': _boot('corrupt_ce'),
      'clean_acc': (
          {'mean': None, 'lo': None, 'hi': None, 'n': 0, 'note': 'N/A'}
          if is_masked else _boot('clean_acc')),
      'clean_ce': (
          {'mean': None, 'lo': None, 'hi': None, 'n': 0, 'note': 'N/A'}
          if is_masked else _boot('clean_ce')),
      'clean_copy_xt_acc': (
          None if is_masked else _boot('clean_copy_xt_acc')),
      'clean_mean_p_x0': (
          None if is_masked else _boot('clean_mean_p_x0')),
      'clean_mean_p_xt': (
          None if is_masked else _boot('clean_mean_p_xt')),
      'clean_median_rank_x0': (
          None if is_masked else _boot('clean_median_rank_x0')),
      'clean_mean_rank_x0': (
          None if is_masked else _boot('clean_mean_rank_x0')),
      'clean_acc_note': (
          'N/A — absorbing/SUBS does not train unmasked sites'
          if is_masked else
          'top-1 == x0 on sites with xt==x0 (after dropping collisions)'),
      'detect_auroc': (
          None if is_masked else _boot('detect_auroc')),
      'detect_auroc_note': (
          'N/A — MASK is visible by construction' if is_masked
          else 'score=1-p(xt) on active block'),
      'n_attempted_moves_total': n_attempted,
      'n_collision_xt_eq_x0_total': n_collision,
      'n_kept_no_move_total': n_kept,
      'n_xt_ne_x0_total': n_xt_ne_x0,
      'collision_rate_among_moves': (
          None if n_attempted == 0 else n_collision / n_attempted),
      'fraction_dropped_as_not_corrupt': (
          None if (n_attempted + n_kept) == 0
          else n_collision / max(n_attempted + n_kept, 1)),
      'mean_n_corrupted': (
          None if not per_prob else
          sum(p['n_corrupted'] for p in per_prob) / len(per_prob)),
      'retention_soft_keep_xt_T1': (
          None if is_masked else _boot('retention_soft_keep_xt_T1')),
      'retention_hard_keep_xt_T1': (
          None if is_masked else _boot('retention_hard_keep_xt_T1')),
      'retention_still_x0_T1': (
          None if is_masked else _boot('retention_still_x0_T1')),
      'retention_soft_keep_xt_T0p1': (
          None if is_masked else _boot('retention_soft_keep_xt_T0p1')),
      'retention_hard_keep_xt_T0p1': (
          None if is_masked else _boot('retention_hard_keep_xt_T0p1')),
      'retention_still_x0_T0p1': (
          None if is_masked else _boot('retention_still_x0_T0p1')),
  }
  return agg, per_prob


def _gather_batches(model, tokenizer, config, device, *, max_blocks: int):
  from discrete_diffusion.data import get_dataloaders
  from omegaconf import OmegaConf
  bs = int(model.block_size)
  L = int(getattr(model, 'num_tokens', None) or config.model.length)
  OmegaConf.set_struct(config, False)
  config.loader.eval_batch_size = 1
  config.loader.batch_size = 1
  _, valid = get_dataloaders(config, tokenizer, skip_train=True)
  out = []
  for batch in valid:
    ids = batch['input_ids'].to(device)
    if ids.size(1) < L:
      pad_id = tokenizer.pad_token_id or 0
      pad = torch.full(
          (1, L - ids.size(1)), pad_id, device=device, dtype=ids.dtype)
      ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, : (L // bs) * bs]
    out.append(ids.cpu())  # store CPU; move per-arm
    if len(out) >= max_blocks:
      break
  if not out:
    raise RuntimeError('no evaluation batches gathered')
  return out, bs, L


def _train_alpha_range(model) -> dict:
  """Max/min α under train t∼U[eps,1] with LogLinear α=1-(1-ε)t."""
  eps = float(getattr(model, 'sampling_eps', 1e-3))
  noise_eps = float(getattr(model.noise, 'eps', 1e-3))
  # t_min = sampling_eps → α_max; t_max = 1 → α_min
  a_max = _alpha_from_t(eps, noise_eps)
  a_min = _alpha_from_t(1.0, noise_eps)
  return {
      'sampling_eps': eps,
      'noise_eps': noise_eps,
      'train_alpha_max_at_t_eps': a_max,
      'train_alpha_min_at_t_1': a_min,
      'note': (
          'Train samples t∈[sampling_eps,1]; α=1.0 is slightly above '
          f'train α_max≈{a_max:.6f}. Prefer 0.999/0.99 for in-range.'),
  }


@torch.no_grad()
def _train_nll_by_alpha(
    model,
    batches: list[torch.Tensor],
    alphas: list[float],
    *,
    active_block: int,
) -> list[dict]:
  """Token CE -log p_θ(x0) on the active block, bucketed by α.

  Same forward as the probe. Compare ``ce_clean_sites`` to probe ``clean_ce``.
  """
  if model.forward_process_name != 'uniform':
    return []
  from discrete_diffusion.forward_process.utils import (
      resolve_uniform_exclude_ids,
  )
  bs = int(model.block_size)
  eps = float(model.noise.eps)
  rows = []
  for alpha in alphas:
    t_val = _t_from_alpha(alpha, eps)
    clean_nlls, all_nlls, cor_nlls = [], [], []
    for x0 in batches:
      seq = x0.shape[1]
      n_blocks = seq // bs
      blk = min(max(int(active_block), 0), n_blocks - 1)
      if n_blocks > 1 and blk == 0:
        blk = 1
      start, end = blk * bs, (blk + 1) * bs
      t = torch.full((1, seq), t_val, device=x0.device, dtype=torch.float32)
      cmask = torch.zeros(1, seq, dtype=torch.bool, device=x0.device)
      cmask[:, start:end] = True
      xt, _ = model._corrupt(
          x0, t, block_size=bs, return_move_mask=True, corruption_mask=cmask)
      logits = model._backbone_logits(xt, x0, block_size=bs)
      if logits.size(1) < seq:
        pad = torch.zeros(
            1, seq - logits.size(1), logits.size(-1),
            device=logits.device, dtype=logits.dtype)
        logits = torch.cat([logits, pad], dim=1)
      logits_u = _ban_exclude_logits(model, logits)
      log_p = F.log_softmax(logits_u.float(), dim=-1)
      ce = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
      corrupted = (xt != x0) & cmask
      clean = cmask & ~corrupted
      if clean.any():
        clean_nlls.append(float(ce[clean].mean().item()))
      if corrupted.any():
        cor_nlls.append(float(ce[corrupted].mean().item()))
      all_nlls.append(float(ce[cmask].mean().item()))
    rows.append({
        'alpha_t': alpha,
        'ce_clean_sites': _bootstrap_mean(clean_nlls),
        'ce_corrupt_sites': _bootstrap_mean(cor_nlls),
        'ce_all_block': _bootstrap_mean(all_nlls),
        'note': 'token CE -log p_θ(x0); same forward as probe',
    })
  return rows


@torch.no_grad()
def _one_step_retention(
    model,
    batches: list[torch.Tensor],
    *,
    alpha: float,
    temps: tuple[float, ...] = (1.0, 0.1),
    active_block: int,
    max_batches: int | None = None,
) -> list[dict]:
  """One ancestral Unif reverse step; fraction of clean tokens kept.

  Decision-relevant vs top-1: sampler mixes keep-xt with p_x0.
  """
  from omegaconf import OmegaConf
  from discrete_diffusion.sampling.block_sampler import BlockSampler

  if model.forward_process_name != 'uniform':
    return []
  bs = int(model.block_size)
  eps = float(model.noise.eps)
  t_val = _t_from_alpha(alpha, eps)
  # One step toward clean: α_s from a slightly smaller t (or noise-removal)
  # Use dt so α_s ≈ min(1, α + 0.05) via schedule, else noise-removal.
  use = batches if max_batches is None else batches[: int(max_batches)]
  out = []
  for temp in temps:
    cfg = OmegaConf.create({
        'algo': {'forward_process_name': 'uniform'},
        'sampling': {
            'use_arpc': False,
            'hierarchical_kv': True,
            'posterior_sampler': 'fast',
            'unmask_threshold': None,
            'uniform_confidence_sticky': False,
            'x0_temperature': float(temp),
            'p_nucleus': 1.0,
            'greedy': False,
        },
    })
    sampler = BlockSampler(cfg)
    kept_fracs, still_x0_fracs, n_clean_tot = [], [], 0
    for x0 in use:
      seq = x0.shape[1]
      n_blocks = seq // bs
      blk = min(max(int(active_block), 0), n_blocks - 1)
      if n_blocks > 1 and blk == 0:
        blk = 1
      start, end = blk * bs, (blk + 1) * bs
      t = torch.full((1, seq), t_val, device=x0.device, dtype=torch.float32)
      cmask = torch.zeros(1, seq, dtype=torch.bool, device=x0.device)
      cmask[:, start:end] = True
      xt, _ = model._corrupt(
          x0, t, block_size=bs, return_move_mask=True, corruption_mask=cmask)
      clean0 = cmask & (xt == x0)
      if not clean0.any():
        continue
      # Noise-removal step (α_s=1): the keep-vs-resample commit
      t_scalar = torch.full((1,), t_val, device=x0.device)
      xs = sampler._uniform_step(
          model, xt.clone(), x0, t_scalar, None,
          window=(start, end))
      # Only active window written in some paths — apply window result
      after = xt.clone()
      after[:, start:end] = xs[:, start:end]
      n_c = int(clean0.sum().item())
      n_clean_tot += n_c
      kept = float((after[clean0] == xt[clean0]).float().mean().item())
      still = float((after[clean0] == x0[clean0]).float().mean().item())
      kept_fracs.append(kept)
      still_x0_fracs.append(still)
    out.append({
        'alpha_t': alpha,
        'x0_temperature': float(temp),
        'step': 'noise_removal_dt_None',
        'retention_keep_xt': _bootstrap_mean(kept_fracs),
        'retention_still_x0': _bootstrap_mean(still_x0_fracs),
        'n_clean_tokens': n_clean_tot,
        'n_problems': len(kept_fracs),
        'note': (
            'keep_xt = fraction of pre-step clean sites unchanged; '
            'still_x0 = fraction still equal to ground truth'),
    })
  return out


@torch.no_grad()
def _model_prefix_eval(
    model,
    batches: list[torch.Tensor],
    *,
    alpha: float,
    active_block: int,
    max_batches: int,
) -> dict:
  """Eval on block B with block B-1 filled by model p_θ argmax (own errors).

  Avoids a second BlockSampler full-seq materialize (login VRAM).
  """
  if model.forward_process_name != 'uniform':
    return {'skipped': True}
  if device_is_cuda(model):
    torch.cuda.empty_cache()
  bs = int(model.block_size)
  eps = float(model.noise.eps)
  t_val = _t_from_alpha(alpha, eps)
  per = []
  for x0_gt in batches[: int(max_batches)]:
    seq = x0_gt.shape[1]
    n_blocks = seq // bs
    blk = min(max(int(active_block), 1), n_blocks - 1)
    prev = blk - 1
    # Fill prev block: pure-noise xt → one denoise forward → argmax x0_hat
    noise = model._hard_pure_noise_xt(x0_gt)
    xt_prev = x0_gt.clone()
    xt_prev[:, prev * bs:(prev + 1) * bs] = noise[
        :, prev * bs:(prev + 1) * bs]
    logits_prev = model._backbone_logits(xt_prev, x0_gt, block_size=bs)
    if logits_prev.size(1) < seq:
      logits_prev = torch.cat([
          logits_prev,
          torch.zeros(
              1, seq - logits_prev.size(1), logits_prev.size(-1),
              device=logits_prev.device, dtype=logits_prev.dtype)], dim=1)
    logits_prev = _ban_exclude_logits(model, logits_prev)
    fill = logits_prev.argmax(-1)
    x0_ctx = x0_gt.clone()
    x0_ctx[:, prev * bs:(prev + 1) * bs] = fill[
        :, prev * bs:(prev + 1) * bs]
    start, end = blk * bs, (blk + 1) * bs
    t = torch.full((1, seq), t_val, device=x0_gt.device, dtype=torch.float32)
    cmask = torch.zeros(1, seq, dtype=torch.bool, device=x0_gt.device)
    cmask[:, start:end] = True
    xt, _ = model._corrupt(
        x0_gt, t, block_size=bs, return_move_mask=True, corruption_mask=cmask)
    corrupted = (xt != x0_gt) & cmask
    logits = model._backbone_logits(xt, x0_ctx, block_size=bs)
    if logits.size(1) < seq:
      logits = torch.cat([
          logits,
          torch.zeros(
              1, seq - logits.size(1), logits.size(-1),
              device=logits.device, dtype=logits.dtype)], dim=1)
    logits = _ban_exclude_logits(model, logits)
    log_p = F.log_softmax(logits.float(), dim=-1)
    pred = log_p.argmax(-1)
    from discrete_diffusion.forward_process.utils import (
        normalize_uniform_simplex_mode, resolve_uniform_exclude_ids,
    )
    mode = normalize_uniform_simplex_mode(
        getattr(model, 'uniform_simplex_mode', 'conversion'))
    excl = tuple(resolve_uniform_exclude_ids(
        model.tokenizer, mask_id=model.mask_id,
        vocab_size=int(model.vocab_size), mode=mode))
    row = _metrics_from_pred(
        pred, log_p, x0_gt, xt, corrupted, cmask,
        report_auroc=True, report_clean_acc=True, exclude_ids=excl)
    prev_sl = slice(prev * bs, (prev + 1) * bs)
    row['prev_block_match_gt'] = float(
        (x0_ctx[:, prev_sl] == x0_gt[:, prev_sl]).float().mean().item())
    per.append(row)
    del logits_prev, logits, log_p, noise, xt_prev

  def boot(key):
    return _bootstrap_mean([p[key] for p in per if p.get(key) is not None])

  return {
      'alpha_t': alpha,
      'n': len(per),
      'corrupt_acc': boot('corrupt_acc'),
      'clean_acc': boot('clean_acc'),
      'detect_auroc': boot('detect_auroc'),
      'clean_mean_p_x0': boot('clean_mean_p_x0'),
      'prev_block_match_gt': boot('prev_block_match_gt'),
      'note': (
          'Prev block = argmax denoise from pure noise (light proxy for '
          'ancestral fill). Active targets=GT; AUROC under own prefix.'),
  }


def device_is_cuda(model) -> bool:
  try:
    return next(model.parameters()).device.type == 'cuda'
  except StopIteration:
    return False


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
  alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

  # Sequential load (one arm on GPU) — login nodes often share VRAM.
  print('Loading masked (batch source of truth)…', flush=True)
  m_model, m_cfg, m_tok = _load(Path(args.masked_ckpt), device)
  m_info = _simplex_info(m_model)
  batches_cpu, _, _ = _gather_batches(
      m_model, m_tok, m_cfg, device, max_blocks=int(args.max_blocks))
  t_recv_m = bool(m_info.get('time_conditioning'))
  if t_recv_m:
    raise SystemExit('time_conditioning unexpectedly True on masked')
  m_bs = int(m_model.block_size)
  batches_m = [b.to(device) for b in batches_cpu]
  m_levels, m_per = [], []
  for a in alphas:
    print(f'  α={a} masked…', flush=True)
    agg, per = _eval_level(
        m_model, batches_m, alpha=a, eps=m_info['eps'],
        block_size=m_bs, active_block=int(args.active_block))
    m_levels.append(agg)
    m_per.append(per)
  del m_model, batches_m
  if device.type == 'cuda':
    torch.cuda.empty_cache()

  print('Loading uniform…', flush=True)
  u_model, u_cfg, u_tok = _load(Path(args.uniform_ckpt), device)
  u_info = _simplex_info(u_model)
  if abs(m_info['eps'] - u_info['eps']) > 1e-9:
    raise SystemExit(
        f"eps mismatch masked={m_info['eps']} uniform={u_info['eps']}")
  if m_info['noise_schedule'] != u_info['noise_schedule']:
    raise SystemExit(
        f"schedule mismatch {m_info['noise_schedule']} vs "
        f"{u_info['noise_schedule']}")
  if m_bs != int(u_model.block_size):
    raise SystemExit('block_size mismatch between arms')
  if u_info.get('v_eff') is None or not isinstance(u_info['v_eff'], int):
    raise SystemExit(f"Unif V_eff unresolved: {u_info.get('v_eff')!r}")
  if bool(u_info.get('time_conditioning')):
    raise SystemExit('time_conditioning unexpectedly True on uniform')
  print(
      f"Unif simplex mode={u_info['uniform_simplex_mode']} "
      f"V={u_info['vocab_size']} V_eff={u_info['v_eff']} "
      f"n_exclude={u_info['n_exclude']} shift={u_info['shift_loss_targets']}",
      flush=True)
  batches_u = [b.to(device) for b in batches_cpu]
  u_levels, u_per = [], []
  for a in alphas:
    print(f'  α={a} uniform…', flush=True)
    agg, per = _eval_level(
        u_model, batches_u, alpha=a, eps=u_info['eps'],
        block_size=int(u_model.block_size),
        active_block=int(args.active_block))
    u_levels.append(agg)
    u_per.append(per)

  deltas = []
  for i, a in enumerate(alphas):
    ma, ua, mc, uc = [], [], [], []
    for pm, pu in zip(m_per[i], u_per[i]):
      if pm['corrupt_acc'] is not None and pu['corrupt_acc'] is not None:
        ma.append(pm['corrupt_acc'])
        ua.append(pu['corrupt_acc'])
      if pm['corrupt_ce'] is not None and pu['corrupt_ce'] is not None:
        mc.append(pm['corrupt_ce'])
        uc.append(pu['corrupt_ce'])
    deltas.append({
        'alpha_t': a,
        't_schedule': m_levels[i]['t_schedule'],
        'd_corrupt_acc_M_minus_U': _bootstrap_paired_diff(ma, ua),
        'd_corrupt_ce_U_minus_M': _bootstrap_paired_diff(uc, mc),
        'unif_detect_auroc': u_levels[i]['detect_auroc'],
        'masked_detect_auroc': None,
        'unif_clean_acc': u_levels[i]['clean_acc'],
        'masked_clean_acc': None,
        'unif_collision_rate_among_moves': u_levels[i][
            'collision_rate_among_moves'],
        'unif_n_collision_dropped': u_levels[i][
            'n_collision_xt_eq_x0_total'],
        'unif_n_xt_ne_x0': u_levels[i]['n_xt_ne_x0_total'],
    })

  noise_split = _noise_level_split(u_levels, alphas)
  train_range = _train_alpha_range(u_model)
  print('Train α range / NLL buckets…', flush=True)
  train_nll_u = _train_nll_by_alpha(
      u_model, batches_u,
      [a for a in alphas if a >= 0.90 or a <= 0.30],
      active_block=int(args.active_block))

  print('One-step retention folded into Unif α levels (soft+hard T=1/0.1).',
        flush=True)
  retention = [
      {
          'alpha_t': lv['alpha_t'],
          'retention_soft_keep_xt_T1': lv.get('retention_soft_keep_xt_T1'),
          'retention_hard_keep_xt_T1': lv.get('retention_hard_keep_xt_T1'),
          'retention_still_x0_T1': lv.get('retention_still_x0_T1'),
          'retention_soft_keep_xt_T0p1': lv.get('retention_soft_keep_xt_T0p1'),
          'retention_hard_keep_xt_T0p1': lv.get('retention_hard_keep_xt_T0p1'),
          'retention_still_x0_T0p1': lv.get('retention_still_x0_T0p1'),
      }
      for lv in u_levels
      if float(lv['alpha_t']) >= 0.90
  ]

  model_prefix = None
  if not args.skip_model_prefix:
    print('Model-generated-prefix pass…', flush=True)
    model_prefix = _model_prefix_eval(
        u_model, batches_u, alpha=0.95,
        active_block=int(args.active_block),
        max_batches=int(
            args.prefix_max_blocks or min(256, int(args.max_blocks))))

  doc = {
      'protocol': 'GAP_DIAGNOSIS_M_U_2026-10-01 Test 3',
      'code_fingerprint': code_fingerprint_header(),
      'note': (
          'Shared sequences. Corrupted := (xt!=x0)&block. '
          'Masked detect_auroc/clean=N/A. Gaps: paired bootstrap CIs. '
          'Cause 6: no time conditioning. Soft clean stats + retention + '
          'model-prefix for decode-time readout.'),
      'time_conditioning_received': False,
      'time_conditioning_note': (
          'BlockTrainer passes sigma=None; Qwen forward deletes sigma. '
          't is used only for corruption + ELBO weights.'),
      'train_alpha_range': train_range,
      'alpha_match': {
          'eps': m_info['eps'],
          'schedule': m_info['noise_schedule'],
          'map_t_from_alpha': m_info['alpha_map'],
          'matched': True,
      },
      'masked': {
          'checkpoint': str(args.masked_ckpt),
          'simplex': m_info,
          'n_sequences': len(batches_cpu),
          'levels': m_levels,
      },
      'uniform': {
          'checkpoint': str(args.uniform_ckpt),
          'simplex': u_info,
          'n_sequences': len(batches_cpu),
          'levels': u_levels,
      },
      'deltas_M_vs_U': deltas,
      'noise_level_split_uniform': noise_split,
      'train_nll_buckets_uniform': train_nll_u,
      'one_step_retention_uniform': retention,
      'model_generated_prefix_uniform': model_prefix,
  }
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(doc, indent=2) + '\n')
  print(f'Wrote {out}')
  print(f"time_conditioning_received={doc['time_conditioning_received']}")
  print(
      f"V_eff={u_info['v_eff']} mode={u_info['uniform_simplex_mode']} "
      f"train_α_max≈{train_range['train_alpha_max_at_t_eps']:.6f}")
  print(
      f"noise_level_split pattern={noise_split['pattern']} "
      f"clean_hi={noise_split['clean_acc_high_alpha']} "
      f"clean_lo={noise_split['clean_acc_low_alpha']}")
  print(
      'alpha  d_acc(M-U)  U_AUROC  U_clean  U_p(x0)  U_rank  '
      'U_clean_ce  coll_moves')
  for i, d in enumerate(deltas):
    da = d['d_corrupt_acc_M_minus_U']
    au = d['unif_detect_auroc']
    au_m = None if au is None else au.get('mean')
    uc = d['unif_clean_acc']
    uc_m = None if not isinstance(uc, dict) else uc.get('mean')
    ul = u_levels[i]
    px = None if not ul.get('clean_mean_p_x0') else ul['clean_mean_p_x0'].get('mean')
    rk = None if not ul.get('clean_median_rank_x0') else ul['clean_median_rank_x0'].get('mean')
    ce = None if not ul.get('clean_ce') else ul['clean_ce'].get('mean')
    print(
        f"  {d['alpha_t']:.4g}  {da['mean']}  {au_m}  {uc_m}  "
        f"{px}  {rk}  {ce}  {d['unif_collision_rate_among_moves']}")
  print('retention (noise-removal, from same forward):')
  for r in retention:
    s1 = (r.get('retention_soft_keep_xt_T1') or {}).get('mean')
    h1 = (r.get('retention_hard_keep_xt_T1') or {}).get('mean')
    x1 = (r.get('retention_still_x0_T1') or {}).get('mean')
    s01 = (r.get('retention_soft_keep_xt_T0p1') or {}).get('mean')
    h01 = (r.get('retention_hard_keep_xt_T0p1') or {}).get('mean')
    print(
        f"  α={r['alpha_t']} soft_keep T1={s1} hard={h1} still_x0={x1} | "
        f"T0.1 soft={s01} hard={h01}")
  if model_prefix and not model_prefix.get('skipped'):
    print(
        f"model_prefix α=0.95 AUROC="
        f"{(model_prefix.get('detect_auroc') or {}).get('mean')} "
        f"clean={ (model_prefix.get('clean_acc') or {}).get('mean') } "
        f"prev_match_gt="
        f"{(model_prefix.get('prev_block_match_gt') or {}).get('mean')}")
  return 0


def cmd_probe_check(args: argparse.Namespace) -> int:
  """GPU: α=1.0 copy check + low-noise Unif vs train NLL context."""
  _repo_src()
  from discrete_diffusion.evaluations.code_fingerprint import (
      assert_forward_process_utils_ok,
      code_fingerprint_header,
  )
  assert_forward_process_utils_ok(require_expected_sha=False)
  device = torch.device(
      args.device if args.device != 'cuda' or torch.cuda.is_available()
      else 'cpu')
  print('Loading uniform…', flush=True)
  u_model, u_cfg, u_tok = _load(Path(args.uniform_ckpt), device)
  u_info = _simplex_info(u_model)
  if not isinstance(u_info.get('v_eff'), int):
    raise SystemExit(f"V_eff fail: {u_info.get('v_eff')}")
  print(
      f"V_eff={u_info['v_eff']} V={u_info['vocab_size']} "
      f"mode={u_info['uniform_simplex_mode']} "
      f"shift={u_info['shift_loss_targets']} "
      f"time_cond={u_info['time_conditioning']}",
      flush=True)
  batches_cpu, _, _ = _gather_batches(
      u_model, u_tok, u_cfg, device, max_blocks=int(args.max_blocks))
  batches = [b.to(device) for b in batches_cpu]
  # α=1.0 — no corruption; clean_acc should be ~1 if model copies
  agg1, _ = _eval_level(
      u_model, batches, alpha=1.0, eps=u_info['eps'],
      block_size=int(u_model.block_size),
      active_block=int(args.active_block))
  # α=0.95 — low noise
  agg95, _ = _eval_level(
      u_model, batches, alpha=0.95, eps=u_info['eps'],
      block_size=int(u_model.block_size),
      active_block=int(args.active_block))
  # Train-time context from wandb summary if present
  train_ctx = None
  summ = (
      Path(args.uniform_ckpt).resolve().parents[1]
      / 'hydra' / 'wandb')
  # walk for wandb-summary.json
  hits = list(summ.glob('**/wandb-summary.json')) if summ.exists() else []
  if hits:
    import json as _json
    s = _json.loads(hits[0].read_text())
    train_ctx = {
        'val_nll': s.get('val/nll'),
        'val_ppl': s.get('val/ppl'),
        'train_nll': s.get('train/nll'),
        'note': (
            'No token-acc in wandb; compare probe clean_acc to whether '
            'copy would be near 1. Soft NLL≠top-1.'),
        'summary_path': str(hits[0]),
    }
  copy_acc = None if not agg1['clean_acc'] else agg1['clean_acc'].get('mean')
  verdict = 'inconclusive'
  if copy_acc is not None:
    if copy_acc >= 0.85:
      verdict = 'copies_at_alpha1'
    elif copy_acc <= 0.40:
      verdict = 'does_not_copy_at_alpha1'
    else:
      verdict = 'partial_copy_at_alpha1'
  doc = {
      'protocol': 'GAP_DIAGNOSIS_M_U_2026-10-01 Test 3 probe-check',
      'code_fingerprint': code_fingerprint_header(),
      'simplex': u_info,
      'n_sequences': len(batches),
      'alpha_1_0': agg1,
      'alpha_0_95': agg95,
      'train_context': train_ctx,
      'verdict_copy': verdict,
      'note': (
          'If α=1 clean_acc≪1, model does not identity-copy; '
          'smoke clean~0.25 can be real. Metric self-check is separate.'),
  }
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(doc, indent=2) + '\n')
  print(json.dumps({
      'verdict_copy': verdict,
      'alpha_1_clean_acc': copy_acc,
      'alpha_1_n_corrupted_mean': agg1.get('mean_n_corrupted'),
      'alpha_095_corrupt_acc': (
          None if not agg95['corrupt_acc'] else
          agg95['corrupt_acc'].get('mean')),
      'alpha_095_clean_acc': (
          None if not agg95['clean_acc'] else
          agg95['clean_acc'].get('mean')),
      'V_eff': u_info['v_eff'],
      'collision_among_moves_a095': agg95.get('collision_rate_among_moves'),
      'n_collision_dropped_a095': agg95.get('n_collision_xt_eq_x0_total'),
      'train_context': train_ctx,
  }, indent=2))
  print(f'Wrote {out}')
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  sp = ap.add_subparsers(dest='cmd', required=True)
  sc = sp.add_parser('self-check', help='Oracle/random metric sanity (CPU)')
  sc.set_defaults(func=lambda _: self_check())
  pc = sp.add_parser(
      'probe-check',
      help='GPU: α=1 copy check + V_eff + low-noise Unif (before full run)')
  pc.add_argument('--uniform-ckpt', required=True)
  pc.add_argument('--device', default='cuda')
  pc.add_argument('--max-blocks', type=int, default=64)
  pc.add_argument('--active-block', type=int, default=1)
  pc.add_argument('--output', required=True)
  pc.set_defaults(func=cmd_probe_check)
  r = sp.add_parser('run')
  r.add_argument('--masked-ckpt', required=True)
  r.add_argument('--uniform-ckpt', required=True)
  r.add_argument('--device', default='cuda')
  r.add_argument('--max-blocks', type=int, default=1000)
  r.add_argument('--active-block', type=int, default=1)
  r.add_argument('--alphas', default=','.join(str(a) for a in DEFAULT_ALPHAS))
  r.add_argument('--retention-max-blocks', type=int, default=0,
                 help='0 = use --max-blocks for retention')
  r.add_argument('--prefix-max-blocks', type=int, default=256)
  r.add_argument('--skip-model-prefix', action='store_true')
  r.add_argument('--output', required=True)
  r.set_defaults(func=cmd_run)
  args = ap.parse_args(argv)
  return int(args.func(args) or 0)


if __name__ == '__main__':
  raise SystemExit(main())
