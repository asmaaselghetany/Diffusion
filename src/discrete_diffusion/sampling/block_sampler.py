"""Block-wise sampler for Qwen BlockTrainer (masked unmask | uniform redraw)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .base import Sampler


class BlockSampler(Sampler):
  """Semi-autoregressive block generator for ``BlockTrainer``.

  Processes the sequence block-by-block. Within each block, runs reverse
  diffusion steps using ``concat(xt, x0)`` through the Qwen block backbone.

  Mode is taken from ``config.algo.forward_process_name``:
  - ``masked``: absorbing unmask steps (Fast-dLLM path)
  - ``uniform``: uniform-state redraw steps (BlockGen path)
  """

  def __init__(self, config, forward_process=None) -> None:
    del forward_process
    self.config = config
    self.mode = getattr(config.algo, 'forward_process_name', 'masked')

  @property
  def is_masked(self) -> bool:
    return self.mode == 'masked'

  def _logits(self, model, xt: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return model.backbone_logits(xt, x0)

  def _expand_alpha(self, model, t_scalar: torch.Tensor, length: int) -> torch.Tensor:
    b = t_scalar.shape[0]
    t = t_scalar.view(b, 1).expand(b, length)
    return model.noise.alpha_t(t)

  def _masked_step(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      t_scalar: torch.Tensor,
      dt: float | None,
  ) -> torch.Tensor:
    b, seq_len = xt.shape
    alpha_t = self._expand_alpha(model, t_scalar, seq_len)
    if dt is None:
      alpha_s = torch.ones_like(alpha_t)
    else:
      t_prev = (t_scalar - dt).clamp(min=0.0)
      alpha_s = self._expand_alpha(model, t_prev, seq_len)

    logits = self._logits(model, xt, x0)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    sampled = sample_categorical(p_x0)
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t).clamp(min=1e-8)
    should_denoise = torch.rand_like(xt, dtype=torch.float32) < prob_denoise
    is_masked = xt == model.mask_id
    should_update = is_masked & should_denoise
    out = torch.where(should_update, sampled, xt)
    return torch.where(xt != model.mask_id, xt, out)

  def _uniform_step(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      t_scalar: torch.Tensor,
      dt: float | None,
  ) -> torch.Tensor:
    b, seq_len = xt.shape
    v = model.vocab_size
    alpha_t = self._expand_alpha(model, t_scalar, seq_len).unsqueeze(-1)
    if dt is None:
      alpha_s = torch.ones_like(alpha_t)
    else:
      t_prev = (t_scalar - dt).clamp(min=0.0)
      alpha_s = self._expand_alpha(model, t_prev, seq_len).unsqueeze(-1)

    logits = self._logits(model, xt, x0)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    if getattr(self.config.sampling, 'use_float64', False):
      p_x0 = p_x0.to(torch.float64)

    alpha_ts = alpha_t / alpha_s.clamp(min=1e-8)
    xt_one_hot = F.one_hot(xt, v).to(p_x0.dtype)
    uniform = torch.full((1, 1, v), 1.0 / v, device=xt.device, dtype=p_x0.dtype)

    numerator = (
        (alpha_t * v * p_x0 * xt_one_hot)
        + ((alpha_ts - alpha_t) * xt_one_hot)
        + ((alpha_s - alpha_t) * p_x0)
        + ((1 - alpha_ts) * (1 - alpha_s) * uniform)
    )
    denom = (alpha_t * v * torch.gather(p_x0, -1, xt.unsqueeze(-1))) + (1 - alpha_t)
    q_xs = numerator / denom.clamp(min=1e-12)
    return sample_categorical(q_xs)

  def _init_block(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.is_masked:
      xt[:, start:end] = model.mask_id
    else:
      xt[:, start:end] = torch.randint(
          0, model.vocab_size, (xt.shape[0], end - start),
          device=xt.device, dtype=xt.dtype)
    x0 = xt.clone()
    return xt, x0

  def _denoise_block(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
      num_steps: int,
      eps: float,
  ) -> torch.Tensor:
    step_fn = self._masked_step if self.is_masked else self._uniform_step
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=xt.device)
    dt = (1.0 - eps) / max(num_steps, 1)

    for i in range(num_steps):
      xt[:, :start] = x0[:, :start]
      t = timesteps[i].expand(xt.shape[0])
      xt = step_fn(model, xt, x0, t, dt)
      x0 = xt.clone()

    xt[:, :start] = x0[:, :start]
    t_final = timesteps[-1].expand(xt.shape[0])
    xt = step_fn(model, xt, x0, t_final, None)
    return xt

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    if num_steps is None:
      num_steps = int(self.config.sampling.steps)
    if eps is None:
      eps = float(getattr(model, 'sampling_eps', 1e-3))
    if inject_bos is None:
      inject_bos = bool(getattr(self.config.sampling, 'inject_bos', True))

    n = model.num_tokens
    bs = model.block_size
    num_blocks = n // bs
    if n % bs != 0:
      raise ValueError(f'num_tokens={n} must divide block_size={bs}')

    xt = model.prior_sample(num_samples, n)
    x0 = xt.clone()
    if inject_bos:
      bos = model.tokenizer.bos_token_id
      xt[:, 0] = bos
      x0[:, 0] = bos

    for block_idx in range(num_blocks):
      start = block_idx * bs
      end = start + bs
      xt, x0 = self._init_block(model, xt, x0, start, end)
      xt = self._denoise_block(model, xt, x0, start, end, num_steps, eps)
      x0 = xt.clone()

    return xt


__all__ = ['BlockSampler']
