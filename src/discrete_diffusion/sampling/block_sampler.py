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

  @staticmethod
  def _ignore_bos(model) -> bool:
    if bool(getattr(model, 'ignore_bos', False)):
      return True
    algo = getattr(getattr(model, 'config', None), 'algo', None)
    return bool(algo is not None and getattr(algo, 'ignore_bos', False))

  def _init_block(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep already-committed prefix; only reset the active block to prior.
    committed = x0[:, :start].clone()
    if self.is_masked:
      xt[:, start:end] = model.mask_id
    else:
      xt[:, start:end] = torch.randint(
          0, model.vocab_size, (xt.shape[0], end - start),
          device=xt.device, dtype=xt.dtype)
    # Keep position-0 BOS when training uses ignore_bos (never corrupts / never
    # trains index 0). Wiping it left block 0 fully masked → unconstrained prior.
    if self._ignore_bos(model) and start == 0:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        xt[:, 0] = bos
        x0[:, 0] = bos
    xt[:, :start] = committed
    x0[:, :start] = committed
    x0[:, start:end] = xt[:, start:end]
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
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-diffuse only ``[start:end]``; freeze prefix and future.

    Uniform reverse draws the full sequence; writing that back into ``x0``
    used to re-noise committed blocks. BlockGen generates
    ``[clean prefix | current xt]`` only.
    """
    step_fn = self._masked_step if self.is_masked else self._uniform_step
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=xt.device)
    dt = (1.0 - eps) / max(num_steps, 1)
    prefix = xt[:, :start].clone()
    future = xt[:, end:].clone()
    bos_id = None
    if self._ignore_bos(model) and start == 0:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        bos_id = bos

    def _restore() -> None:
      xt[:, :start] = prefix
      x0[:, :start] = prefix
      xt[:, end:] = future
      x0[:, end:] = future
      if bos_id is not None:
        xt[:, 0] = bos_id
        x0[:, 0] = bos_id

    def _apply(t_scalar: torch.Tensor, step_dt: float | None) -> None:
      _restore()
      x0[:, start:end] = xt[:, start:end]
      xt_new = step_fn(model, xt, x0, t_scalar, step_dt)
      xt[:, start:end] = xt_new[:, start:end]
      _restore()
      x0[:, start:end] = xt[:, start:end]

    for i in range(num_steps):
      t = timesteps[i].expand(xt.shape[0])
      _apply(t, dt)

    t_final = timesteps[-1].expand(xt.shape[0])
    _apply(t_final, None)
    return xt, x0

  @torch.no_grad()
  def generate(
      self,
      model,
      *,
      num_samples,
      num_steps,
      eps,
      inject_bos,
      prefix_ids: torch.Tensor | None = None,
      max_new_tokens: int | None = None,
  ):
    """Block-wise free-gen, optionally conditioned on a clean ``prefix_ids``.

    ``prefix_ids`` shape ``(B, L)`` or ``(L,)`` is written into the sequence and
    frozen; generation continues from the first unfinished position (Fast-dLLM-
    style prompt continuation for lm-eval).

    If ``max_new_tokens`` is set, only enough trailing blocks to cover that
    many new tokens are denoised (rounded up to ``block_size``).
    """
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
    prefix_len = 0
    if prefix_ids is not None:
      if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0).expand(num_samples, -1)
      if prefix_ids.shape[0] != num_samples:
        raise ValueError(
            f'prefix_ids batch {prefix_ids.shape[0]} != num_samples={num_samples}')
      # Leave at least one token free so we can generate a continuation.
      prefix_len = int(min(prefix_ids.shape[1], n - 1))
      if prefix_len > 0:
        xt[:, :prefix_len] = prefix_ids[:, :prefix_len].to(xt.device)
        x0[:, :prefix_len] = xt[:, :prefix_len]
    elif inject_bos:
      bos = model.tokenizer.bos_token_id
      xt[:, 0] = bos
      x0[:, 0] = bos

    first_block = prefix_len // bs
    if max_new_tokens is not None and max_new_tokens > 0:
      target_end = min(n, prefix_len + int(max_new_tokens))
      # Round up to a block boundary so the last partial block is fully decoded.
      last_block = (target_end + bs - 1) // bs
      last_block = min(last_block, num_blocks)
    else:
      last_block = num_blocks

    for block_idx in range(first_block, last_block):
      start = block_idx * bs
      end = start + bs
      denoise_start = max(start, prefix_len)
      if denoise_start >= end:
        continue
      xt, x0 = self._init_block(model, xt, x0, denoise_start, end)
      xt, x0 = self._denoise_block(
          model, xt, x0, denoise_start, end, num_steps, eps)
      if self.use_arpc and not self.is_masked:
        xt = self._arpc_correct_block(model, xt, x0, denoise_start, end)
        x0[:, denoise_start:end] = xt[:, denoise_start:end]
      x0[:, :end] = xt[:, :end]
      if prefix_len > 0:
        xt[:, :prefix_len] = prefix_ids[:, :prefix_len].to(xt.device)
        x0[:, :prefix_len] = xt[:, :prefix_len]

    return xt


__all__ = ['BlockSampler']
