"""Unified block-diffusion sampler for masked and uniform models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process import is_absorbing_forward_process_config
from ..forward_process.utils import sample_categorical
from .base import Sampler


class BlockDiffusionSampler(Sampler):
  """Semi-autoregressive block sampler delegating posterior steps by mode."""

  def __init__(self, config, forward_process=None) -> None:
    del forward_process
    self.config = config
    self._is_absorbing = is_absorbing_forward_process_config(config)

  def _nucleus_sample(self, model, p_x0: torch.Tensor) -> torch.Tensor:
    p = getattr(self.config.sampling, 'p_nucleus', 1.0)
    if p == 1.0:
      return p_x0
    block_size = model.block_size
    p_x0_block = p_x0[:, -block_size:].clone()
    sorted_probs, sorted_indices = p_x0_block.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_block.scatter_(-1, sorted_indices, sorted_probs)
    p_x0_block /= p_x0_block.sum(-1, keepdim=True)
    p_x0[:, -block_size:] = p_x0_block
    return p_x0

  def _model_p_x0_block(self, model, x, t):
    alpha_t = model.noise.alpha_t(t)
    sigma_t = model._sigma_from_alphat(alpha_t)
    if getattr(self.config.sampling, 'kv_cache', False):
      log_p_x0 = model.forward(
        x[:, -model.block_size:], sigma_t, sample_mode=True)
    else:
      log_p_x0 = model.forward(x, sigma_t, sample_mode=True)
      log_p_x0 = log_p_x0[:, -model.block_size:]
    if self.config.sampling.use_float64:
      log_p_x0 = log_p_x0.to(torch.float64)
    p_x0 = log_p_x0.exp()
    return self._nucleus_sample(model, p_x0)

  def _absorbing_block_posterior(self, model, x, t, dt, p_x0=None,
                                 noise_removal_step=False):
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)
    if p_x0 is None:
      p_x0 = self._model_p_x0_block(model, x, t)

    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    sampled_x0 = sample_categorical(p_x0)
    should_denoise_draw = (
      torch.rand_like(x[:, -model.block_size:], dtype=torch.float64,
                      device=x.device) < prob_denoise)
    x_block = x[:, -model.block_size:]
    is_masked = (x_block == model.mask_id)
    should_denoise_mask = is_masked & should_denoise_draw
    _x_block = torch.where(should_denoise_mask, sampled_x0, x_block)
    x_block = torch.where(x_block != model.mask_id, x_block, _x_block)
    x_new = torch.cat((x[:, :-model.block_size], x_block), dim=-1)

    if getattr(self.config.sampling, 'kv_cache', False):
      sigma_t = model._sigma_from_alphat(alpha_t)
      if model.mask_id not in x_block:
        _ = model.forward(
          x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    return p_x0, x_new

  def _uniform_block_posterior(self, model, x, t, dt, p_x0=None,
                               noise_removal_step=False):
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)

    if p_x0 is None:
      p_x0 = self._model_p_x0_block(model, x, t)

    v = model.vocab_size
    alpha_t3 = alpha_t[..., None]
    alpha_s3 = alpha_s[..., None]
    alpha_ts = alpha_t3 / alpha_s3
    x_block = x[:, -model.block_size:]
    xt_one_hot = F.one_hot(x_block, v)
    limiting = model.limiting_distribution.view(1, 1, -1)

    numerator = (
      (alpha_t3 * v * p_x0 * xt_one_hot)
      + ((alpha_ts - alpha_t3) * xt_one_hot)
      + ((alpha_s3 - alpha_t3) * p_x0)
      + ((1 - alpha_ts) * (1 - alpha_s3) * limiting)
    )
    denom = (
      (alpha_t3 * v * torch.gather(p_x0, -1, x_block[..., None]))
      + (1 - alpha_t3)
    )
    q_xs = numerator / denom
    x_block = sample_categorical(q_xs)
    x_new = torch.cat((x[:, :-model.block_size], x_block), dim=-1)

    if getattr(self.config.sampling, 'kv_cache', False):
      sigma_t = model._sigma_from_alphat(alpha_t)
      _ = model.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    return p_x0, x_new

  def compute_posterior(self, model, x, t, dt, p_x0=None,
                        noise_removal_step=False):
    if self._is_absorbing:
      return self._absorbing_block_posterior(
        model, x, t, dt, p_x0=p_x0,
        noise_removal_step=noise_removal_step)
    return self._uniform_block_posterior(
      model, x, t, dt, p_x0=p_x0, noise_removal_step=noise_removal_step)

  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    return torch.special.entr(counts.float() / counts.sum()).sum()

  def _check_stop_conds(self, model, x):
    stop = False
    truncate_idx = None
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True
    if getattr(self.config.sampling, 'var_length', False):
      eos_positions = torch.where(x == model.tokenizer.eos_token_id)
      if len(eos_positions[0]) > 1:
        stop = True
        truncate_idx = min(eos_positions[1][1] + 1, x.shape[1])
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)
    return stop, x

  def _semi_ar_sampler(self, model, num_steps, seqlen, inject_bos, eps):
    n_samples = 1
    ones = torch.ones((n_samples, 1), dtype=model.dtype, device=model.device)
    if getattr(self.config.sampling, 'kv_cache', False):
      reset_fn = getattr(model.backbone, 'reset_kv_cache', None)
      if callable(reset_fn):
        reset_fn()

    sampling_steps = 0
    num_strides = seqlen // model.block_size
    x_accum = None

    for stride_num in range(num_strides):
      if stride_num == 0:
        x_accum = model.prior_sample(n_samples, model.block_size)
        if inject_bos:
          x_accum[:, 0] = model.tokenizer.bos_token_id
      else:
        x_block = model.prior_sample(n_samples, model.block_size)
        x_accum = torch.cat((x_accum, x_block), dim=1)

      end_idx = (stride_num + 1) * model.block_size
      start_idx = max(end_idx - 1024, 0)
      fwd_idx = torch.arange(start_idx, end_idx, device=model.device)

      dt = (1 - eps) / max(num_steps, 1)
      p_x0_cache = None
      timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)

      for idx in range(num_steps):
        current_t = timesteps[idx] * ones
        p_x0_cache, x_next = self.compute_posterior(
          model=model,
          x=x_accum[:, fwd_idx],
          t=current_t,
          dt=dt,
          p_x0=p_x0_cache)
        if p_x0_cache is None:
          sampling_steps += 1
        x_accum[:, fwd_idx] = x_next

      t0 = timesteps[-1] * ones
      _, x_accum[:, fwd_idx] = self.compute_posterior(
        model=model,
        x=x_accum[:, fwd_idx],
        t=t0,
        dt=dt,
        p_x0=p_x0_cache,
        noise_removal_step=True)

      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(model, x_accum)
        if stop:
          return None, None
    return x_accum, sampling_steps

  def _sample_once(self, model, num_steps, eps, inject_bos):
    seqlen = self.config.model.length
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
      sample, nfes = self._semi_ar_sampler(
        model=model,
        num_steps=num_steps,
        seqlen=seqlen,
        inject_bos=inject_bos,
        eps=eps)
      if sample is not None:
        return sample, nfes
      attempts += 1
    raise ValueError('Sampling failed.')

  def _update_metrics(self, model, nfes):
    if hasattr(model.metrics, 'nfes'):
      model.metrics.nfes.update(nfes)
    if hasattr(model.metrics, 'gen_nfes'):
      model.metrics.gen_nfes.append(nfes)

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    if num_samples is None:
      num_samples = self.config.loader.eval_batch_size
    if num_steps is None:
      num_steps = getattr(self.config.algo, 'T', 0) or self.config.sampling.steps
    if eps is None:
      eps = 1e-5
    if inject_bos is None:
      inject_bos = getattr(self.config.sampling, 'inject_bos', True)

    samples = []
    for _ in range(num_samples):
      sample, nfes = self._sample_once(
        model=model,
        num_steps=num_steps,
        eps=eps,
        inject_bos=inject_bos)
      samples.append(sample)
      if nfes is not None:
        self._update_metrics(model, nfes)
    return torch.cat(samples, dim=0)


__all__ = ['BlockDiffusionSampler']
