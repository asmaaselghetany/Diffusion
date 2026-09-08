"""Block-wise sampler for Qwen BlockTrainer (masked unmask | uniform redraw)."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .arpc import (
    causal_log_probs_for_span,
    corruption_indices,
)
from .base import Sampler
from .shift_logits import align_shift_logits


logger = logging.getLogger(__name__)


_ARPC_MODES = ('simplified', 'blockgen')
_CORRUPTION_MODES = ('random', 'divergence', 'diffusion_metric', 'ar_metric')
_DIVERGENCE = ('kld', 'reverse_kld', 'tvd')
_DIFF_METRICS = ('entropy', 'confidence', 'margin')
_AR_METRICS = ('nll', 'gap_to_top1', 'entropy')


class BlockSampler(Sampler):
  """Semi-autoregressive block generator for ``BlockTrainer``.

  Processes the sequence block-by-block. Within each block, runs reverse
  diffusion steps. Default path uses ``concat(xt, x0)`` (train graph);
  ``sampling.single_stream_decode`` uses Hub ``eval_block_diff_mask``.

  Mode is taken from ``config.algo.forward_process_name``:
  - ``masked`` / ``hybrid``: absorbing unmask steps (hybrid trains with
    mask+uniform mix but decodes like masked for the first paper cell)
  - ``uniform``: uniform-state redraw steps (BlockGen path)

  Decode accelerations (default off):
  - ``hierarchical_kv``: progressive truncated forward (prefix+active only)
  - ``single_stream_decode``: Hub single-stream block-causal (+ DualCache)
  - ``sub_block_size``: Fast-dLLM small-block windows inside each attention block.
  """

  def __init__(self, config, forward_process=None) -> None:
    del forward_process
    self.config = config
    self.forward_process_name = getattr(
        config.algo, 'forward_process_name', 'masked')
    self.mode = self.forward_process_name
    if self.mode == 'hybrid':
      # Eval-as-masked for B4 v1 (prior = all MASK).
      self.mode = 'masked'
    sampling = getattr(config, 'sampling', None)
    self.use_arpc = bool(getattr(sampling, 'use_arpc', False))
    self.arpc_mode = str(getattr(sampling, 'arpc_mode', 'simplified') or 'simplified')
    self.arpc_prefix_frac = float(getattr(sampling, 'arpc_prefix_frac', 0.25))
    self.arpc_resample_tau = float(getattr(sampling, 'arpc_resample_tau', 0.5))
    self.arpc_corruption_mode = str(
        getattr(sampling, 'arpc_corruption_mode', 'divergence') or 'divergence')
    self.arpc_divergence_measure = str(
        getattr(sampling, 'arpc_divergence_measure', 'kld') or 'kld')
    self.arpc_diffusion_metric = str(
        getattr(sampling, 'arpc_diffusion_metric', 'confidence') or 'confidence')
    self.arpc_ar_metric = str(
        getattr(sampling, 'arpc_ar_metric', 'nll') or 'nll')
    self.arpc_warmup_steps = int(getattr(sampling, 'arpc_warmup_steps', 0) or 0)
    self.arpc_guide_every = int(getattr(sampling, 'arpc_guide_every', 1) or 1)
    self.arpc_temperature = float(
        getattr(sampling, 'arpc_temperature', 1.0) or 1.0)
    raw_prefix = getattr(sampling, 'arpc_use_prefix_fill', None)
    if raw_prefix is None or raw_prefix == 'null':
      self.arpc_use_prefix_fill = self.arpc_mode == 'simplified'
    else:
      self.arpc_use_prefix_fill = bool(raw_prefix)
    # None = auto (follow model.shift_loss_targets); bool overrides for A/B eval.
    align = getattr(sampling, 'align_shift_logits', None) if sampling else None
    self._align_shift_logits_override = align
    sub = getattr(sampling, 'sub_block_size', None) if sampling else None
    self.sub_block_size = int(sub) if sub not in (None, 0, 'null') else None
    self.hierarchical_kv = bool(getattr(sampling, 'hierarchical_kv', False))
    self.use_block_cache = bool(getattr(sampling, 'use_block_cache', False))
    self.single_stream_decode = bool(
        getattr(sampling, 'single_stream_decode', False))
    self._dual_cache = None  # set per denoise window when use_block_cache
    self.p_nucleus = float(getattr(sampling, 'p_nucleus', 1.0) if sampling else 1.0)
    raw_thr = getattr(sampling, 'unmask_threshold', None) if sampling else None
    self.unmask_threshold = (
        None if raw_thr in (None, 'null', '') else float(raw_thr))
    # Codex-fixes: pad tokens after first EOS (stops post-EOS soup in token space).
    self.pad_after_eos = bool(getattr(sampling, 'pad_after_eos', True))
    # Stop scheduling later blocks once every row has emitted EOS in the
    # generated span (audit fix: was filling full 2048 after early collapse).
    self.stop_on_eos = bool(getattr(sampling, 'stop_on_eos', True))
    self._arpc_warned = False
    self._validate_sampling_flags()

  def _validate_sampling_flags(self) -> None:
    if self.use_arpc and self.forward_process_name != 'uniform':
      raise ValueError(
          'sampling.use_arpc=true requires algo.forward_process_name=uniform '
          f'(got {self.forward_process_name!r}; ARPC is uniform-only)')
    if self.arpc_mode not in _ARPC_MODES:
      raise ValueError(
          f'sampling.arpc_mode={self.arpc_mode!r} not in {_ARPC_MODES}')
    if self.arpc_corruption_mode not in _CORRUPTION_MODES:
      raise ValueError(
          f'sampling.arpc_corruption_mode={self.arpc_corruption_mode!r} '
          f'not in {_CORRUPTION_MODES}')
    if self.arpc_divergence_measure not in _DIVERGENCE:
      raise ValueError(
          f'sampling.arpc_divergence_measure='
          f'{self.arpc_divergence_measure!r} not in {_DIVERGENCE}')
    if self.arpc_diffusion_metric not in _DIFF_METRICS:
      raise ValueError(
          f'sampling.arpc_diffusion_metric={self.arpc_diffusion_metric!r} '
          f'not in {_DIFF_METRICS}')
    if self.arpc_ar_metric not in _AR_METRICS:
      raise ValueError(
          f'sampling.arpc_ar_metric={self.arpc_ar_metric!r} not in {_AR_METRICS}')
    if self.arpc_guide_every < 1:
      raise ValueError('sampling.arpc_guide_every must be >= 1')
    if self.arpc_warmup_steps < 0:
      raise ValueError('sampling.arpc_warmup_steps must be >= 0')
    if self.arpc_temperature <= 0:
      raise ValueError(
          f'sampling.arpc_temperature must be > 0, got {self.arpc_temperature}')
    if self.use_block_cache and not self.hierarchical_kv:
      raise ValueError(
          'sampling.use_block_cache=true requires sampling.hierarchical_kv=true')
    if self.use_block_cache and not self.single_stream_decode:
      logger.warning(
          'sampling.use_block_cache=true without single_stream_decode: '
          'using dual-stream DualCache splice. Prefer '
          'sampling.single_stream_decode=true for Hub DualCache semantics.')
    if (self.forward_process_name == 'hybrid'
        and str(getattr(self.config.algo, 'hybrid_decode', 'masked')) != 'masked'):
      raise ValueError(
          'algo.hybrid_decode must be \"masked\" for B4v1 '
          '(DESIGN_LOCKS B4v1); got '
          f'{getattr(self.config.algo, "hybrid_decode", None)!r}')

  def _maybe_warn_arpc_mixture(self, model) -> None:
    """Codex-fixes: BlockGen ARPC needs size-1 in the train mixture."""
    if self._arpc_warned or not self.use_arpc:
      return
    mixture = list(getattr(model, 'block_size_mixture', None) or [])
    weights = getattr(model, 'block_weights', None)
    cfg = getattr(model, 'config', None)
    if not mixture and cfg is not None:
      algo = getattr(cfg, 'algo', None)
      mixture = list(getattr(algo, 'block_size_mixture', None) or [])
      if weights is None:
        weights = getattr(algo, 'block_weights', None)
    has_size1 = 1 in mixture
    if not has_size1 and weights is not None:
      # Weighted 2^k: index 0 → size 1.
      try:
        w = list(weights)
        has_size1 = len(w) > 0 and float(w[0]) > 0
      except (TypeError, ValueError):
        has_size1 = False
    if not has_size1:
      logger.warning(
          'BlockGen ARPC needs algo.block_size_mixture including 1 '
          '(or block_weights with mass on size 1); got mixture=%r weights=%r. '
          'AR verify (L\'=1) is untrained on this ckpt.',
          mixture or [], weights)
    self._arpc_warned = True

  @staticmethod
  def _pad_after_eos(
      samples: torch.Tensor,
      *,
      start: int,
      eos_id: int | None,
      pad_id: int | None,
  ) -> torch.Tensor:
    """Keep the first EOS and replace every later token with padding.

    Ported from Diffusion-codex-fixes: stops post-EOS garbage in token space
    (text ``stop_at_im_end`` alone still leaves junk ids in the tensor).
    """
    if eos_id is None:
      return samples
    fill_id = eos_id if pad_id is None else int(pad_id)
    out = samples
    for row in out:
      hits = row[start:].eq(eos_id).nonzero(as_tuple=False)
      if hits.numel() > 0:
        eos_pos = start + int(hits[0, 0])
        row[eos_pos + 1:] = fill_id
    return out

  @property
  def is_masked(self) -> bool:
    return self.mode == 'masked'

  def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """Apply ``arpc_temperature`` (1.0 = identity) before ARPC softmax."""
    if self.arpc_temperature == 1.0:
      return logits
    return logits / self.arpc_temperature

  def _logits(
      self, model, xt: torch.Tensor, x0: torch.Tensor,
      *, active_end: int | None = None,
      window: tuple[int, int] | None = None,
  ) -> torch.Tensor:
    """Decode logits; Hub single-stream or dual-stream train graph."""
    bs = getattr(model, 'block_size', None)
    use_dc = (
        self.use_block_cache and self.hierarchical_kv
        and active_end is not None
        and window is not None)

    if self.single_stream_decode:
      if use_dc and hasattr(model.backbone, 'block_eval_prefill'):
        w0, w1 = window
        if self._dual_cache is None:
          logits, self._dual_cache = model.backbone.block_eval_prefill(
              xt, active_len=active_end, block_size=bs)
          return logits
        return model.backbone.block_eval_replace(
            xt, active_len=active_end, window=(w0, w1),
            cache=self._dual_cache, block_size=bs)
      a = active_end if (
          self.hierarchical_kv and active_end is not None
          and active_end < xt.shape[1]) else None
      if hasattr(model.backbone, 'block_eval_logits'):
        return model.backbone.block_eval_logits(
            xt, active_len=a, block_size=bs)
      # Fallback: dual-stream with x0 tracking xt (quality-equivalent).
      return model.backbone_logits(
          xt, x0, active_len=a if a is not None else active_end)

    use_dc_dual = use_dc and hasattr(model.backbone, 'block_diff_prefill')
    if use_dc_dual:
      w0, w1 = window
      if self._dual_cache is None:
        logits, self._dual_cache = model.backbone.block_diff_prefill(
            xt, x0, active_len=active_end, block_size=bs)
        return logits
      return model.backbone.block_diff_replace(
          xt, x0, active_len=active_end, window=(w0, w1),
          cache=self._dual_cache, block_size=bs)
    if (self.hierarchical_kv and active_end is not None
        and active_end < xt.shape[1]):
      return model.backbone_logits(xt, x0, active_len=active_end)
    return model.backbone_logits(xt, x0)

  def _shift_align_enabled(self, model) -> bool:
    if self._align_shift_logits_override is not None:
      return bool(self._align_shift_logits_override)
    return bool(getattr(model, 'shift_loss_targets', False))

  @staticmethod
  def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
      return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    drop = cumsum - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(
        min=1e-8)
    out = torch.zeros_like(probs)
    return out.scatter(-1, sorted_idx, sorted_probs)

  def _prepare_masked_logits(self, model, logits: torch.Tensor) -> torch.Tensor:
    logits = align_shift_logits(logits, enabled=self._shift_align_enabled(model))
    if getattr(model, 'mask_id', None) is not None:
      logits = logits.clone()
      neg = float(getattr(model, 'neg_infinity', -1e6))
      logits[..., model.mask_id] = neg
    pad_id = getattr(getattr(model, 'tokenizer', None), 'pad_token_id', None)
    if pad_id is not None:
      logits = logits.clone()
      logits[..., int(pad_id)] = float('-inf')
    return logits

  def _expand_alpha(self, model, t_scalar: torch.Tensor, length: int) -> torch.Tensor:
    b = t_scalar.shape[0]
    t = t_scalar.view(b, 1).expand(b, length)
    # Match training ELBO when Fast-dLLM mask schedule was used.
    if getattr(model, 'mask_schedule', 'alpha') == 'fast_dllm':
      eps = float(getattr(model.noise, 'eps', 1e-3))
      p_mask = (1.0 - eps) * t + eps
      return (1.0 - p_mask).to(dtype=torch.float32)
    return model.noise.alpha_t(t)

  def _masked_step(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      t_scalar: torch.Tensor,
      dt: float | None,
      *,
      active_end: int | None = None,
      window: tuple[int, int] | None = None,
  ) -> torch.Tensor:
    b, seq_len = xt.shape
    alpha_t = self._expand_alpha(model, t_scalar, seq_len)
    if dt is None:
      alpha_s = torch.ones_like(alpha_t)
    else:
      t_prev = (t_scalar - dt).clamp(min=0.0)
      alpha_s = self._expand_alpha(model, t_prev, seq_len)

    logits = self._prepare_masked_logits(
        model, self._logits(
            model, xt, x0, active_end=active_end, window=window))
    # Truncated / DualCache forward may return length active_end — pad back.
    if logits.size(1) < seq_len:
      pad = torch.zeros(
          b, seq_len - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    p_x0 = self._apply_top_p(p_x0, self.p_nucleus)
    if getattr(self, '_greedy_decode', False):
      sampled = p_x0.argmax(dim=-1)
    else:
      sampled = sample_categorical(p_x0)
    is_masked = xt == model.mask_id
    if self.unmask_threshold is not None:
      # Fast-dLLM confidence commit: unmask sites with conf >= threshold,
      # and always commit the highest-confidence masked token (Hub generate).
      conf = p_x0.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
      commit = is_masked & (conf >= self.unmask_threshold)
      # Force at least one unmask per row among still-masked positions.
      conf_masked = conf.masked_fill(~is_masked, float('-inf'))
      max_idx = conf_masked.argmax(dim=-1)
      rows = torch.arange(b, device=xt.device)
      still = is_masked.any(dim=-1)
      commit = commit.clone()
      commit[rows[still], max_idx[still]] = True
      commit = commit & is_masked
      out = torch.where(commit, sampled, xt)
      return torch.where(~is_masked, xt, out)
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t).clamp(min=1e-8)
    should_denoise = torch.rand_like(xt, dtype=torch.float32) < prob_denoise
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
      *,
      active_end: int | None = None,
      window: tuple[int, int] | None = None,
  ) -> torch.Tensor:
    b, seq_len = xt.shape
    v = model.vocab_size
    alpha_t = self._expand_alpha(model, t_scalar, seq_len).unsqueeze(-1)
    if dt is None:
      alpha_s = torch.ones_like(alpha_t)
    else:
      t_prev = (t_scalar - dt).clamp(min=0.0)
      alpha_s = self._expand_alpha(model, t_prev, seq_len).unsqueeze(-1)

    logits = self._logits(
        model, xt, x0, active_end=active_end, window=window)
    if logits.size(1) < seq_len:
      pad = torch.zeros(
          b, seq_len - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
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
    q_xs = self._apply_top_p(q_xs, self.p_nucleus)
    if getattr(self, '_greedy_decode', False):
      return q_xs.argmax(dim=-1)
    return sample_categorical(q_xs)

  def _arpc_prefix_fill(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """AR-informed prefix inside the current block (simplified ARPC)."""
    block_len = end - start
    prefix_len = max(1, int(block_len * self.arpc_prefix_frac))
    use_cache = self.hierarchical_kv and hasattr(
        model.backbone, 'causal_next_with_cache')
    if start > 0:
      context = x0[:, :start]
      past = None
      for pos in range(start, start + prefix_len):
        if use_cache:
          logits, past = model.backbone.causal_next_with_cache(context, past)
          next_tok = logits.argmax(dim=-1)
        else:
          logits = model.backbone.causal_logits(context)
          next_tok = logits[:, -1, :].argmax(dim=-1)
        xt[:, pos] = next_tok
        x0[:, pos] = next_tok
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
    else:
      past = None
      for pos in range(1, min(prefix_len, block_len)):
        if use_cache:
          logits, past = model.backbone.causal_next_with_cache(
              x0[:, :pos], past)
          next_tok = logits.argmax(dim=-1)
        else:
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
    """Resample low-confidence tokens after block denoising (simplified)."""
    logits = self._scale_logits(self._logits(
        model, xt, x0, active_end=end, window=(start, end))[:, start:end])
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

  def _ar_log_probs_block(
      self, model, x0: torch.Tensor, start: int, end: int,
  ) -> torch.Tensor:
    """``[B, end-start, V]`` causal log-probs for the active block (AR verify)."""
    logits = model.backbone.causal_logits(x0[:, :end])
    return causal_log_probs_for_span(
        logits, start, end, vocab_size=model.vocab_size)

  def _arpc_guided_step(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
      t_scalar: torch.Tensor,
      dt: float,
      *,
      block_prefix_len: int,
  ) -> None:
    """BlockGen predictor–corrector: clean proposal → score → re-noise top-k."""
    block_len = end - start
    alpha_t = self._expand_alpha(model, t_scalar, xt.shape[1])
    t_prev = (t_scalar - dt).clamp(min=0.0)
    alpha_s = self._expand_alpha(model, t_prev, xt.shape[1])

    logits = self._scale_logits(
        self._logits(model, xt, x0, active_end=end, window=(start, end)))
    if self.is_masked:
      logits = self._prepare_masked_logits(model, logits)
    log_p = F.log_softmax(logits[:, start:end], dim=-1)
    # Predictor: sample clean proposal for the active block.
    proposal = sample_categorical(log_p.exp())
    if block_prefix_len > 0:
      proposal = proposal.clone()
      proposal[:, :block_prefix_len] = xt[:, start:start + block_prefix_len]
    xt[:, start:end] = proposal
    x0[:, start:end] = proposal

    log_p_ar = None
    if self.arpc_corruption_mode in ('divergence', 'ar_metric'):
      log_p_ar = self._ar_log_probs_block(model, x0, start, end)

    frac = float((1.0 - alpha_s[:, start:end].mean()).clamp(min=0.0, max=1.0))
    num_to_corrupt = max(1, round(frac * block_len))
    idxs = corruption_indices(
        log_p_x0=log_p,
        log_p_ar=log_p_ar,
        x_current=xt[:, start:end],
        num_to_corrupt=num_to_corrupt,
        block_prefix_len=block_prefix_len,
        corruption_mode=self.arpc_corruption_mode,
        divergence_measure=self.arpc_divergence_measure,
        diffusion_metric=self.arpc_diffusion_metric,
        ar_metric=self.arpc_ar_metric,
    )
    if self.is_masked:
      noisy = torch.full(
          (xt.shape[0], idxs.shape[1]), model.mask_id,
          device=xt.device, dtype=xt.dtype)
    else:
      noisy = torch.randint(
          0, model.vocab_size, (xt.shape[0], idxs.shape[1]),
          device=xt.device, dtype=xt.dtype)
    block = xt[:, start:end].clone()
    block.scatter_(1, idxs, noisy)
    xt[:, start:end] = block
    x0[:, start:end] = block

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
      *,
      inject_bos: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep already-committed prefix; only reset the active block to prior.
    committed = x0[:, :start].clone()
    if self.is_masked:
      xt[:, start:end] = model.mask_id
    else:
      xt[:, start:end] = torch.randint(
          0, model.vocab_size, (xt.shape[0], end - start),
          device=xt.device, dtype=xt.dtype)
    # When inject_bos=False (lm-eval continuation), do not force BOS at pos 0.
    if self._ignore_bos(model) and start == 0 and inject_bos:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        xt[:, 0] = bos
        x0[:, 0] = bos
    xt[:, :start] = committed
    x0[:, :start] = committed
    x0[:, start:end] = xt[:, start:end]
    if (self.use_arpc and not self.is_masked and self.arpc_use_prefix_fill):
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
      *,
      inject_bos: bool = True,
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
    if self._ignore_bos(model) and start == 0 and inject_bos:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        bos_id = bos

    # Fresh DualCache for this window (invalidate after prior redraw in init).
    self._dual_cache = None
    window = (start, end)

    # Tokens already filled by ARPC prefix fill stay protected across reverse
    # steps (and in blockgen guided corruption).
    block_prefix_len = 0
    if (self.use_arpc and not self.is_masked and self.arpc_use_prefix_fill):
      block_prefix_len = max(1, int((end - start) * self.arpc_prefix_frac))
      # BOS at pos 0 is already frozen separately when ignore_bos.
      if start == 0 and self._ignore_bos(model):
        block_prefix_len = min(block_prefix_len, end - start)
      # Snapshot AR-filled tokens so _restore can reinstate them after steps.
      ar_prefix = xt[:, start:start + block_prefix_len].clone()
    else:
      ar_prefix = None

    def _restore() -> None:
      xt[:, :start] = prefix
      x0[:, :start] = prefix
      xt[:, end:] = future
      x0[:, end:] = future
      if bos_id is not None:
        xt[:, 0] = bos_id
        x0[:, 0] = bos_id
      if ar_prefix is not None and block_prefix_len > 0:
        xt[:, start:start + block_prefix_len] = ar_prefix
        x0[:, start:start + block_prefix_len] = ar_prefix

    def _apply(t_scalar: torch.Tensor, step_dt: float | None) -> None:
      _restore()
      x0[:, start:end] = xt[:, start:end]
      xt_new = step_fn(
          model, xt, x0, t_scalar, step_dt,
          active_end=end, window=window)
      xt[:, start:end] = xt_new[:, start:end]
      _restore()
      x0[:, start:end] = xt[:, start:end]

    use_blockgen = (
        self.use_arpc and not self.is_masked and self.arpc_mode == 'blockgen')

    # Fast-dLLM confidence decode: iterate until the block has no masks
    # (Hub generate while-loop), not a fixed α-schedule of length ``steps``.
    # Bound = remaining mask count (codex-fixes): each pass must make progress.
    if self.is_masked and self.unmask_threshold is not None:
      t_full = torch.ones(xt.shape[0], device=xt.device)
      max_iters = max(int((xt[:, start:end] == model.mask_id).sum().item()), 1)
      max_iters = max(max_iters, end - start)
      for _ in range(max_iters):
        if not (xt[:, start:end] == model.mask_id).any():
          break
        before = int((xt[:, start:end] == model.mask_id).sum().item())
        _apply(t_full, None)
        after = int((xt[:, start:end] == model.mask_id).sum().item())
        if after >= before:
          raise RuntimeError(
              'masked confidence decoding failed to make progress '
              f'(masks {before} → {after} in [{start},{end}))')
      else:
        if (xt[:, start:end] == model.mask_id).any():
          raise RuntimeError(
              'masked confidence decoding exhausted its forced-progress bound '
              f'in [{start},{end})')
      _restore()
      x0[:, start:end] = xt[:, start:end]
      return xt, x0

    for i in range(num_steps):
      t = timesteps[i].expand(xt.shape[0])
      is_guided = (
          use_blockgen
          and i >= self.arpc_warmup_steps
          and (i - self.arpc_warmup_steps) % self.arpc_guide_every == 0
          and i < num_steps - 1)
      if is_guided:
        _restore()
        x0[:, start:end] = xt[:, start:end]
        self._arpc_guided_step(
            model, xt, x0, start, end, t, dt,
            block_prefix_len=block_prefix_len)
        _restore()
        x0[:, start:end] = xt[:, start:end]
      else:
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
      inject_bos: bool = True,
      prefix_ids: torch.Tensor | None = None,
      max_new_tokens: int | None = None,
      greedy: bool | None = None,
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
    if greedy is None:
      greedy = bool(getattr(self.config.sampling, 'greedy', False))
    self._greedy_decode = bool(greedy)
    self._maybe_warn_arpc_mixture(model)

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

    sub = self.sub_block_size
    if sub is not None:
      if sub <= 0 or bs % sub != 0:
        raise ValueError(
            f'sampling.sub_block_size={sub} must be >0 and divide '
            f'block_size={bs}')

    for block_idx in range(first_block, last_block):
      start = block_idx * bs
      end = start + bs
      denoise_start = max(start, prefix_len)
      if denoise_start >= end:
        continue
      # Sub-block windows (Fast-dLLM small_block_size analogue).
      windows: list[tuple[int, int]] = []
      if sub is None:
        windows.append((denoise_start, end))
      else:
        # Align windows to attention-block grid, then clip to denoise_start.
        for s in range(start, end, sub):
          e = min(s + sub, end)
          s2 = max(s, denoise_start)
          if s2 < e:
            windows.append((s2, e))
      for w_start, w_end in windows:
        xt, x0 = self._init_block(
            model, xt, x0, w_start, w_end, inject_bos=inject_bos)
        xt, x0 = self._denoise_block(
            model, xt, x0, w_start, w_end, num_steps, eps,
            inject_bos=inject_bos)
        if (self.use_arpc and not self.is_masked
            and self.arpc_mode == 'simplified'):
          xt = self._arpc_correct_block(model, xt, x0, w_start, w_end)
          x0[:, w_start:w_end] = xt[:, w_start:w_end]
      x0[:, :end] = xt[:, :end]
      if prefix_len > 0:
        xt[:, :prefix_len] = prefix_ids[:, :prefix_len].to(xt.device)
        x0[:, :prefix_len] = xt[:, :prefix_len]

      # Early-stop: all rows already produced EOS in the generated region.
      if self.stop_on_eos:
        tok = getattr(model, 'tokenizer', None)
        eos_id = getattr(tok, 'eos_token_id', None) if tok is not None else None
        gen_start = (
            prefix_len if prefix_ids is not None else (1 if inject_bos else 0))
        if eos_id is not None and end > gen_start:
          has_eos = (xt[:, gen_start:end] == eos_id).any(dim=-1)
          if bool(has_eos.all()):
            break

    generated_start = (
        prefix_len if prefix_ids is not None else (1 if inject_bos else 0))
    if self.pad_after_eos:
      tok = getattr(model, 'tokenizer', None)
      xt = self._pad_after_eos(
          xt,
          start=generated_start,
          eos_id=getattr(tok, 'eos_token_id', None),
          pad_id=getattr(tok, 'pad_token_id', None),
      )
    return xt


__all__ = ['BlockSampler']
