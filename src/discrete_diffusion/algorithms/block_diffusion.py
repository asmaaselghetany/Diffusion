"""Unified block-diffusion trainer for masked and uniform forward processes."""

from __future__ import annotations

from dataclasses import dataclass

import hydra.utils
import numpy as np
import torch
from omegaconf import OmegaConf

from ..evaluations import BD3Metrics
from ..forward_process import (
  AbsorbingForwardProcess,
  UniformForwardProcess,
  is_absorbing_forward_process_config,
)
from ..noise_schedules import LogLinear
from .base import Diffusion, ensure_mask_token
from .mdlm import MDLM


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class BlockDiffusion(Diffusion):
  """Block diffusion with per-block noise and ELBO objectives.

  Masked (absorbing) and uniform modes differ only in forward process,
  model output processing, and ``nll_per_token`` — block training is shared.
  """

  def __init__(self, config, tokenizer):
    self._is_absorbing = is_absorbing_forward_process_config(config)
    if self._is_absorbing:
      OmegaConf.set_struct(config.algo, False)
      config.algo.parameterization = 'subs'
      OmegaConf.set_struct(config.algo, True)
      self.mask_id, vocab_size = ensure_mask_token(tokenizer)
      super().__init__(config, tokenizer, vocab_size=vocab_size)
    else:
      super().__init__(config, tokenizer)
      self.register_buffer(
        'limiting_distribution',
        torch.full((self.vocab_size,), 1.0 / float(self.vocab_size)))
      self.zero_recon_loss = getattr(config.algo, 'zero_recon_loss', True)

    self._forward_process = hydra.utils.instantiate(
      self.config.algo.forward_process,
      tokenizer=self.tokenizer,
      schedule=self.noise,
    )
    if self._is_absorbing:
      if not isinstance(self._forward_process, AbsorbingForwardProcess):
        raise ValueError(
          'Masked block diffusion requires AbsorbingForwardProcess')
    elif not isinstance(self._forward_process, UniformForwardProcess):
      raise ValueError('Uniform block diffusion requires UniformForwardProcess')

    self._init_block_diffusion(config)
    if self.config.model.length % self.block_size != 0:
      raise ValueError(
        f'model.length ({self.config.model.length}) must be divisible by '
        f'block_size ({self.block_size})'
      )
    self._validate_configuration()

  def _init_block_diffusion(self, config):
    self.cross_attn = getattr(self.config.algo, 'cross_attn', False)
    self.block_size = getattr(config, 'block_size', self.config.model.length)
    self.var_min = getattr(self.config.algo, 'var_min', False)
    self.metrics = BD3Metrics(config)

    if not isinstance(self.noise, LogLinear):
      raise ValueError(
        'BlockDiffusion requires LogLinear noise schedule, got '
        f'{type(self.noise).__name__}'
      )

    self.sigma_max = -torch.log(self.noise.alpha_t(torch.tensor(1.0)))
    self.sigma_min = torch.tensor(self.noise.eps, dtype=torch.float32)

    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min, dtype=torch.float32))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max, dtype=torch.float32))

    self.time_conditioning = getattr(self.config.algo, 'time_conditioning', False)
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _validate_configuration(self):
    super()._validate_configuration()
    if not self._is_absorbing and self.time_conditioning:
      raise ValueError('Uniform block diffusion expects time_conditioning=False')

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if hasattr(self.backbone, 'block_diff_mask'):
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    return self

  def on_train_epoch_start(self):
    super().on_train_epoch_start()
    self._train_mode()

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    return losses.loss

  def on_validation_epoch_start(self):
    super().on_validation_epoch_start()
    if self.var_min:
      self.sampling_eps = self.config.training.sampling_eps

  def validation_step(self, batch, batch_idx):
    del batch_idx

    if self.var_min:
      valid_loss = None
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        losses_clip = self._loss(
          batch['input_ids'],
          batch['attention_mask'],
          sampling_eps_min=sampling_eps_min,
          sampling_eps_max=sampling_eps_max)
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max):
          valid_loss = losses_clip
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          nlls = losses_clip.nlls
          per_block = nlls.reshape(nlls.shape[0], -1, self.block_size).mean(-1)
          self.metrics.valid_vars[noise_clip_start].append(per_block)
      if valid_loss is not None:
        self.metrics.valid_nlls.update(valid_loss.nlls, valid_loss.token_mask)
      return valid_loss.loss if valid_loss is not None else losses_clip.loss

    losses = self._loss(
      batch['input_ids'],
      batch['attention_mask'],
      sampling_eps_min=1e-3 if self.block_size > 1 else 1,
      sampling_eps_max=1 if self.block_size > 1 else 1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def on_validation_epoch_end(self):
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    self._train_mode()

  def prior_sample(self, *batch_dims):
    if self._is_absorbing:
      size = (batch_dims[0] if len(batch_dims) == 1
              and isinstance(batch_dims[0], (tuple, list)) else batch_dims)
      return torch.full(tuple(size), self.mask_id,
                        dtype=torch.int64, device=self.device)
    return torch.randint(
      low=0, high=self.vocab_size, size=batch_dims,
      device=self.device, dtype=torch.int64)

  def _build_block_model_input(self, xt, x0):
    if self.cross_attn:
      return torch.cat((xt, x0), dim=-1)
    return xt

  def _sigma_from_p(self, p):
    return torch.min(-torch.log(1 - p), self.sigma_max)

  def _sample_t(self, batch_dims, device, sampling_eps_min, sampling_eps_max,
                block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)
    if self.antithetic_sampling:
      offset_b = torch.arange(
        batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]
      insert_special = getattr(self.config.data, 'insert_train_special', False)
      insert_eos = getattr(self.config.data, 'insert_train_eos', False)
      if insert_special or insert_eos:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None,
            sampling_eps_max=None):
    if sampling_eps_min is None and self.var_min:
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif sampling_eps_min is None:
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0

    (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(
      x0, attention_mask)
    del output_tokens
    loss = self._forward_pass_diffusion(
      input_tokens,
      t=t,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max)

    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0

    nlls = loss * attention_mask
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for value in var:
        agg_var = value.to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
               all_vars / max(len(var), 1),
               on_epoch=True,
               on_step=False,
               sync_dist=True)
    if getattr(self.config.algo, 'fix_clipping', False) is False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    if (sampling_eps_min == 1e-3 and sampling_eps_max == 1
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True
    if self.block_size == 1 and sampling_eps_min >= 1:
      return True
    return False

  def _subs_parameterization(self, logits, xt):
    logits = logits.clone()
    logits[:, :, self.mask_id] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked_indices = (xt != self.mask_id)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.backbone(
        x, sigma, sample_mode=sample_mode, store_kv=store_kv)
    xt = x[:, :self.config.model.length] if self.cross_attn else x
    if self._is_absorbing:
      return self._subs_parameterization(logits, xt=xt)
    return torch.log_softmax(logits, dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    if self._is_absorbing:
      return MDLM.nll_per_token(
        self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=low_var)
    return self._uniform_nll_per_token(
      log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=low_var)

  def _uniform_nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                             low_var=False):
    """Uniform Duo ELBO — same formulation as BlockGen, active Lightning dtype."""
    if alpha_t.ndim == 3:
      alpha_t = alpha_t.squeeze(-1)
    if dalpha_t.ndim == 3:
      dalpha_t = dalpha_t.squeeze(-1)
    if alpha_t.ndim == 1:
      alpha_t = alpha_t.unsqueeze(-1)
    if dalpha_t.ndim == 1:
      dalpha_t = dalpha_t.unsqueeze(-1)

    vocab_size = self.vocab_size
    x_reconst = log_x_theta.exp()
    alpha_t3 = alpha_t.unsqueeze(-1)
    x_bar_theta = vocab_size * alpha_t3 * x_reconst + 1 - alpha_t3
    coeff = dalpha_t / (vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).to(log_x_theta.dtype)
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    if low_var:
      term1 = torch.zeros_like(xbar_xt)
    else:
      term1 = vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)
    const = (1 - alpha_t) / (vocab_size * alpha_t + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = (
      (vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt
    ) * const.log()
    term2_theta = -term2_coefs * (
      x_bar_theta.log().sum(-1) - vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - vocab_size * alpha_t / (1 - alpha_t)
      * (xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    if self.zero_recon_loss:
      return diffusion_loss
    return diffusion_loss + self._reconstruction_loss(x0)

  def _noised_tokens(self, x, t):
    out = self._forward_process(x, t)
    return out[0] if isinstance(out, (tuple, list)) else out

  def _block_noise_fraction(self, xt, x, block_size):
    x_blocks = x.reshape(x.shape[0], -1, block_size)
    xt_blocks = xt.reshape(x.shape[0], -1, block_size)
    if self._is_absorbing:
      return (xt_blocks == self.mask_id).float().sum(-1) / block_size
    return (xt_blocks != x_blocks).float().sum(-1) / block_size

  def _resample_xt(self, x, xt, move_indices, p, block_size,
                   sampling_eps_min, sampling_eps_max):
    perc_noised = self._block_noise_fraction(xt, x, block_size)

    while (perc_noised < sampling_eps_min).any() or (
        perc_noised > sampling_eps_max).any():
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_noised > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_noised < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      else:
        regen_idx = ((perc_noised < sampling_eps_min)
                     | (perc_noised > sampling_eps_max))
      regen_idx = regen_idx.repeat_interleave(block_size, dim=-1)
      move_indices[regen_idx] = (
        torch.rand(*x.shape, device=x.device) < p)[regen_idx]
      xt = self._forward_process.corrupt_with_mask(x, move_indices)
      xt = xt.reshape(x.shape[0], -1, block_size)
      perc_noised = self._block_noise_fraction(xt, x, block_size)
    return xt

  def _block_corrupt(self, x, t, sampling_eps_min, sampling_eps_max):
    block_size = self.block_size
    xt = self._noised_tokens(x, t)

    if block_size == 1 and sampling_eps_min == 1.0:
      return self.prior_sample(x.shape)

    if self.config.training.resample and not (
        sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      alpha_t = self.noise.alpha_t(t)
      if alpha_t.ndim == 1:
        alpha_t = alpha_t.unsqueeze(-1)
      p = (1.0 - alpha_t).to(dtype=torch.float32)
      move_indices = torch.rand(*x.shape, device=x.device) <= p
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_xt(
        x, xt, move_indices, p, block_size,
        sampling_eps_min, sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None,
                              sampling_eps_max=None):
    if sampling_eps_min is None:
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    if t is None:
      t = self._sample_t(
        x0.shape, x0.device, sampling_eps_min, sampling_eps_max)

    xt = self._block_corrupt(
      x0, t, sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max)

    alpha_t = self.noise.alpha_t(t)
    dalpha_t = self.noise.alpha_prime_t(t)
    if alpha_t.ndim == 1:
      alpha_t = alpha_t.unsqueeze(-1)
    if dalpha_t.ndim == 1:
      dalpha_t = dalpha_t.unsqueeze(-1)

    p = 1.0 - alpha_t
    sigma = self._sigma_from_p(p[:, 0].unsqueeze(-1))

    if self.config.algo.ignore_bos:
      xt[:, 0] = x0[:, 0]

    log_x_theta = self.forward(
      self._build_block_model_input(xt, x0), sigma=sigma)
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=self.loss_type == 'low_var')


__all__ = ['BlockDiffusion', 'Loss']
