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
  - ``masked``: absorbing unmask steps
  - ``uniform``: uniform-state redraw steps (BlockGen path)
  """

  def __init__(self, config, forward_process=None) -> None:
    del forward_process
    self.config = config
    self.mode = getattr(config.algo, 'forward_process_name', 'masked')
    sampling = getattr(config, 'sampling', None)
    self.use_arpc = bool(getattr(sampling, 'use_arpc', False))
    self.arpc_prefix_frac = float(getattr(sampling, 'arpc_prefix_frac', 0.25))
    self.arpc_resample_tau = float(getattr(sampling, 'arpc_resample_tau', 0.5))

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
    # Match training SUBS: never sample the mask token as content.
    if getattr(model, 'mask_id', None) is not None:
      logits = logits.clone()
      neg = float(getattr(model, 'neg_infinity', -1e6))
      logits[..., model.mask_id] = neg
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

  def _arpc_prefix_fill(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """AR-informed prefix inside the current block (BlockGen ARPC)."""
    block_len = end - start
    prefix_len = max(1, int(block_len * self.arpc_prefix_frac))
    if start > 0:
      context = x0[:, :start]
      for pos in range(start, start + prefix_len):
        logits = model.backbone.causal_logits(context)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        xt[:, pos] = next_tok
        x0[:, pos] = next_tok
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
    else:
      for pos in range(1, min(prefix_len, block_len)):
        logits = model.backbone.causal_logits(x0[:, :pos])
        next_tok = logits[:, -1, :].argmax(dim=-1)
        xt[:, pos] = next_tok
        x0[:, pos] = next_tok
    return xt, x0

  def _arpc_correct_block(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
  ) -> torch.Tensor:
    """Resample low-confidence tokens after block denoising."""
    logits = self._logits(model, xt, x0)[:, start:end]
    probs = F.log_softmax(logits, dim=-1).exp()
    conf = probs.gather(-1, xt[:, start:end].unsqueeze(-1)).squeeze(-1)
    low = conf < self.arpc_resample_tau
    if not low.any():
      return xt
    sampled = sample_categorical(probs)
    block = xt[:, start:end].clone()
    block = torch.where(low, sampled, block)
    xt[:, start:end] = block
    return xt

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
    # Keep position-0 BOS when training uses ignore_bos (never corrupts / never
    # trains index 0). Wiping it left block 0 fully masked → unconstrained prior.
    ignore_bos = bool(getattr(model, 'ignore_bos', False)) or bool(
        getattr(getattr(model, 'config', None), 'algo', None)
        and getattr(model.config.algo, 'ignore_bos', False))
    if ignore_bos and start == 0:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        xt[:, 0] = bos
    x0 = xt.clone()
    if self.use_arpc and not self.is_masked:
      xt, x0 = self._arpc_prefix_fill(model, xt, x0, start, end)
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
      if self.use_arpc and not self.is_masked:
        xt = self._arpc_correct_block(model, xt, x0, start, end)
      x0 = xt.clone()

    return xt


__all__ = ['BlockSampler']
