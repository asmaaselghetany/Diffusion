#!/usr/bin/env python3
"""Audit U2 size-1 uniform ELBO vs U0 (same batches / fixed t)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def load(checkpoint: Path, device: torch.device):
  import hydra.utils
  from discrete_diffusion.data import get_dataloaders, get_tokenizer
  from discrete_diffusion.train import register_config_resolvers

  register_config_resolvers()
  ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)
  OmegaConf.set_struct(config, False)
  config.trainer.devices = 1
  config.trainer.num_nodes = 1
  config.trainer.accumulate_grad_batches = 1
  config.loader.batch_size = 1
  config.loader.eval_batch_size = 1
  config.loader.eval_global_batch_size = 1
  config.loader.global_batch_size = 1
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  model = algo_cls.load_from_checkpoint(
      str(checkpoint), config=config, tokenizer=tokenizer, map_location=device)
  model.to(device)
  model.eval()
  _, valid_loader = get_dataloaders(config, tokenizer, skip_train=True)
  return model, valid_loader, config


@torch.no_grad()
def diagnose(model, batch, block_size: int, device, fixed_t: float | None):
  from discrete_diffusion.forward_process.block_masked import sample_block_timesteps

  x0 = batch['input_ids'].to(device)
  valid = batch['attention_mask'].to(device)
  bsz, T = x0.shape
  bs = block_size
  if fixed_t is None:
    t = sample_block_timesteps(
        bsz, T, bs, device, sampling_eps=model.sampling_eps,
        antithetic=False, stratified_gamma=0.0)
  else:
    t = torch.full((bsz, T), float(fixed_t), device=device)
  alpha_t, dalpha_t = model._elbo_schedule_weights(t)
  xt = model._corrupt(x0, t, block_size=bs)
  if model.ignore_bos:
    xt = xt.clone()
    xt[:, 0] = x0[:, 0]
  logits = model._backbone_logits(xt, x0, block_size=bs, attention_mask=valid)
  loss = model._loss_for_block(logits, xt, x0, alpha_t, dalpha_t, block_size=bs)
  loss_m, valid_m = model._apply_ignore_bos_mask(loss, valid)
  if (model.shift_loss_targets and valid_m.size(-1) == loss_m.size(-1) + 1):
    valid_m = valid_m[:, 1:]
  per = loss_m * valid_m
  toks = float(valid_m.sum().item())
  mean = float(per.sum().item()) / max(toks, 1.0)
  probs = logits.float().softmax(-1)
  ent = -(probs * (probs.clamp_min(1e-12).log())).sum(-1)
  return {
      'block_size': bs,
      'fixed_t': fixed_t,
      'mean_nll': mean,
      'ppl': math.exp(min(mean, 40)),
      'loss_min': float(loss_m.min()),
      'loss_max': float(loss_m.max()),
      'loss_p99': float(torch.quantile(loss_m.float().reshape(-1), 0.99)),
      'logit_abs_mean': float(logits.float().abs().mean()),
      'logit_max': float(logits.float().max()),
      'entropy_mean': float(ent.mean()),
      'alpha_mean': float(alpha_t.mean()),
      'xt_eq_x0_rate': float((xt == x0).float().mean()),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--u2', required=True)
  ap.add_argument('--u0', required=True)
  ap.add_argument('--batches', type=int, default=2)
  ap.add_argument('--device', default='cuda')
  args = ap.parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

  print('Loading U2…', flush=True)
  u2, loader, _ = load(Path(args.u2), device)
  print('Loading U0…', flush=True)
  u0, _, _ = load(Path(args.u0), device)

  batches = []
  for i, batch in enumerate(loader):
    batches.append(batch)
    if i + 1 >= args.batches:
      break

  for label, model in [('U2', u2), ('U0', u0)]:
    print(f'\n==== {label} joint_ar={model.joint_ar_alpha} '
          f'causal={model.causal_clean_stream} ====')
    for bs in (1, 32):
      for ft in (None, 0.1, 0.5, 0.9):
        stats = []
        for batch in batches:
          stats.append(diagnose(model, batch, bs, device, ft))
        # average key fields
        keys = ['mean_nll', 'ppl', 'loss_max', 'loss_p99', 'logit_abs_mean',
                'entropy_mean', 'xt_eq_x0_rate']
        avg = {k: sum(s[k] for s in stats) / len(stats) for k in keys}
        print(
            f'  bs={bs:2d} t={str(ft):4s}  nll={avg["mean_nll"]:.3f}  '
            f'ppl={avg["ppl"]:.2f}  loss_max={avg["loss_max"]:.1f}  '
            f'p99={avg["loss_p99"]:.2f}  |logit|={avg["logit_abs_mean"]:.2f}  '
            f'H={avg["entropy_mean"]:.2f}  eq={avg["xt_eq_x0_rate"]:.3f}',
            flush=True)

  # Ablation: zero joint flags on U2 (nll path should be unchanged)
  print('\n==== U2 with joint_ar_alpha forced 0 (should match above if nll-pure) ====')
  u2.joint_ar_alpha = 0.0
  u2.causal_clean_stream = False
  for bs in (1, 32):
    stats = [diagnose(u2, b, bs, device, 0.5) for b in batches]
    mean = sum(s['mean_nll'] for s in stats) / len(stats)
    print(f'  bs={bs} t=0.5 nll={mean:.3f} ppl={math.exp(min(mean,40)):.2f}')


if __name__ == '__main__':
  sys.exit(main() or 0)
