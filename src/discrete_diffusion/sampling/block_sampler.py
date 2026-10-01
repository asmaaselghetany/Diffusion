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
from .shift_logits import ShiftMode, align_shift_logits


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
  - ``ar_block_bridge``: Hub post-block AR append (auto ON for masked+shift).
  - ``use_arpc`` + ``arpc_mode=blockgen``: BlockGen ARPC on masked (remask) or
    uniform (redraw). Requires size-1 in the train mixture / block_weights.
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
        getattr(sampling, 'arpc_corruption_mode', 'ar_metric') or 'ar_metric')
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
    self.x0_temperature = float(
        getattr(sampling, 'x0_temperature', 1.0) or 1.0)
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
    # Ablation escape: keep legacy full-seq dual open-loop (C0 ancestral 2.27%
    # path). Default False — auto-force hierarchical_ss packing instead.
    self.allow_full_seq_decode = bool(
        getattr(sampling, 'allow_full_seq_decode', False)
        if sampling else False)
    # Uniform Confidence Commit (UCC): Hub DualCache *commit* analog on
    # USDM (Unif undecided, not MASK). Not DualCache K/V. Optional revise
    # is off by default (Hub has none). See DESIGN_LOCKS U0-UNIFORM-COMMIT.
    self.uniform_confidence_sticky = bool(
        getattr(sampling, 'uniform_confidence_sticky', False)
        if sampling else False)
    # UCC: refuse force-max unless peak conf ≥ this (0 = always force-max,
    # matching Hub generate force-max).
    raw_min = (
        getattr(sampling, 'sticky_min_conf', 0.0) if sampling else 0.0)
    self.sticky_min_conf = float(raw_min if raw_min not in (None, 'null') else 0.0)
    # Optional BlockGen-style corrector (NOT Hub DualCache). Default off.
    self.uniform_commit_revise = bool(
        getattr(sampling, 'uniform_commit_revise', False)
        if sampling else False)
    raw_rev = (
        getattr(sampling, 'uniform_commit_revise_tau', 0.25)
        if sampling else 0.25)
    self.uniform_commit_revise_tau = float(
        raw_rev if raw_rev not in (None, 'null') else 0.25)
    # Ablation: pick force-max / thr sites uniformly at random among candidates
    # instead of by confidence rank. Token value on commit stays argmax(p_x0).
    # For thr=1 UCC (``uniform_dual``), thr almost never fires → random force-max
    # is the honest "order doesn't matter" control (bake D5).
    self.uniform_commit_random = bool(
        getattr(sampling, 'uniform_commit_random', False)
        if sampling else False)
    # Commit site order for UCC force-max: confidence | random | ltr.
    # ``uniform_commit_random=true`` forces ``random`` (bake D5).
    raw_order = (
        getattr(sampling, 'uniform_commit_order', None) if sampling else None)
    if self.uniform_commit_random:
      self.uniform_commit_order = 'random'
    elif raw_order in (None, 'null', ''):
      self.uniform_commit_order = 'confidence'
    else:
      self.uniform_commit_order = str(raw_order).strip().lower()
      if self.uniform_commit_order not in ('confidence', 'random', 'ltr'):
        raise ValueError(
            'sampling.uniform_commit_order must be '
            f'confidence|random|ltr, got {self.uniform_commit_order!r}')
    # Per-generate NFE / commit accounting (reset in generate()).
    self._nfe_stats: dict[str, int] = {}
    self.last_nfe_stats: dict[str, int] | None = None
    self._reset_nfe_stats()
    self._block_sticky_frozen: torch.Tensor | None = None
    # DualCache lifetime matches Hub ``block_past_key_values``: keep across
    # sub-windows inside one attention block; invalidate on new attention
    # block or Hub refresh (first small-block token still MASK).
    self._dual_cache = None
    # Optional per-reverse-step callback for decode videos / traces.
    # Signature: hook(event: dict) -> None  (see ``_emit_step_hook``).
    self.step_hook = None
    self._decode_trace_step = 0
    self.p_nucleus = float(getattr(sampling, 'p_nucleus', 1.0) if sampling else 1.0)
    raw_thr = getattr(sampling, 'unmask_threshold', None) if sampling else None
    self.unmask_threshold = (
        None if raw_thr in (None, 'null', '') else float(raw_thr))
    # Hub AR block-bridge (generation_functions.py ~81-85). null = auto:
    # ON for masked + shift (Fast-dLLM parity), OFF for uniform/ARPC.
    raw_bridge = (
        getattr(sampling, 'ar_block_bridge', None) if sampling else None)
    if raw_bridge is None or raw_bridge == 'null':
      self._ar_block_bridge_override = None
    else:
      self._ar_block_bridge_override = bool(raw_bridge)
    # Ban MASK/PAD in denoise logits (safety). Hub does not ban; set false for
    # bit-closer Hub parity when probing decode mismatches.
    self.ban_mask_pad_logits = bool(
        getattr(sampling, 'ban_mask_pad_logits', True) if sampling else True)
    # Codex-fixes: pad tokens after first EOS (stops post-EOS soup in token space).
    self.pad_after_eos = bool(getattr(sampling, 'pad_after_eos', True))
    # Stop scheduling later blocks once every row has emitted EOS in the
    # generated span (audit fix: was filling full 2048 after early collapse).
    self.stop_on_eos = bool(getattr(sampling, 'stop_on_eos', True))
    # Unifusion editability probe: count token changes under uniform reverse.
    # Default off — enable via sampling.track_revisions=true (isolated eval).
    self.track_revisions = bool(
        getattr(sampling, 'track_revisions', False) if sampling else False)
    self._revision_stats: dict[str, float] | None = None
    # BlockGen default posterior sampler is ``fast`` (no full V×L materialize).
    # ``naive`` materializes q_xs (our previous path / Duo-style).
    raw_ps = str(
        getattr(sampling, 'posterior_sampler', 'fast') or 'fast').strip().lower()
    if raw_ps not in ('fast', 'naive'):
      raise ValueError(
          f'sampling.posterior_sampler={raw_ps!r} not in (fast|naive)')
    self.posterior_sampler = raw_ps
    self._arpc_warned = False
    self._validate_sampling_flags()

  def reset_revision_stats(self) -> None:
    self._revision_stats = {
        'token_changes': 0.0,
        'token_slots': 0.0,
        'steps_tracked': 0.0,
    }

  def pop_revision_stats(self) -> dict[str, float] | None:
    """Return revision rates (or None if tracking disabled / empty)."""
    if not self.track_revisions or not self._revision_stats:
      return None
    s = self._revision_stats
    slots = float(s.get('token_slots') or 0.0)
    out = {
        'revision_token_changes': float(s.get('token_changes') or 0.0),
        'revision_token_slots': slots,
        'revision_rate': (
            float(s['token_changes']) / slots if slots > 0 else 0.0),
        'revision_steps_tracked': float(s.get('steps_tracked') or 0.0),
        'note': (
            'Fraction of (batch,pos) updates that changed the token under '
            'uniform reverse (editable-token probe). Masked commits are N/A.'),
    }
    return out

  def _use_block_scope(self) -> bool:
    """Hub densifies only the current attention block (not future MASKs).

    True for hierarchical / DualCache / single-stream / confidence decode.
    Open-loop ancestral auto-forces ``hierarchical_kv`` (truncation); only
    ``allow_full_seq_decode`` keeps legacy full-seq dual.
    """
    return bool(
        self.hierarchical_kv
        or self.use_block_cache
        or self.single_stream_decode
        or self.unmask_threshold is not None
        or self.uniform_confidence_sticky)

  def _validate_sampling_flags(self) -> None:
    # BlockGen ARPC is corruption-agnostic (absorb remask | uniform redraw).
    # See third_party/blockgen scripts blockgen_{absorb,uniform}_ar_then_arpc*.sh.
    if self.use_arpc and self.forward_process_name not in (
        'uniform', 'masked', 'hybrid'):
      raise ValueError(
          'sampling.use_arpc=true requires forward_process_name in '
          f'{{uniform, masked, hybrid}}; got {self.forward_process_name!r}')
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
    if self.x0_temperature <= 0:
      raise ValueError(
          f'sampling.x0_temperature must be > 0, got {self.x0_temperature}')
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
    # Refuse full-seq dual open-loop (attend future MASK/Unif) — C0 ancestral
    # GSM 2.27% on that path. Truncation (hierarchical_kv) is skeleton fairness.
    # Do NOT force Hub single_stream: dual-trained BlockGen-spine models decode
    # with truncated dual (``hierarchical``); Fast-dLLM remask uses ss via
    # hubmatch/dual_cache profiles, not this open-loop auto-force.
    _open_loop = (
        not self.allow_full_seq_decode
        and not self.use_block_cache
        and self.unmask_threshold is None
        and not self.uniform_confidence_sticky)
    if (self.forward_process_name in ('masked', 'uniform', 'hybrid')
        and _open_loop
        and not self.hierarchical_kv):
      logger.warning(
          '%s open-loop decode without hierarchical_kv (full-seq dual). '
          'C0 ancestral was 2.27%% GSM on that path. Auto-enabling '
          'hierarchical_kv. Pass it explicitly to silence.',
          self.forward_process_name)
      self.hierarchical_kv = True
    if self.uniform_confidence_sticky and self.forward_process_name == 'masked':
      raise ValueError(
          'sampling.uniform_confidence_sticky is for uniform/hybrid only '
          '(masked uses DualCache + unmask_threshold)')
    # UCC works under dual OR single-stream packing. Do NOT force ss=true —
    # that made the remask twin unfair vs masked B1 (ss=false, sub_block=null).
    # Opt into Hub ss packing via profile ``uniform_commit_ss`` if needed.

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

  @staticmethod
  def _attention_block_end(end: int, block_size: int, seq_len: int) -> int:
    """Ceil ``end`` to the enclosing attention-block boundary (Hub span)."""
    if block_size <= 0:
      raise ValueError(f'block_size must be >0, got {block_size}')
    return min(seq_len, ((end + block_size - 1) // block_size) * block_size)

  def _truncated_active_end(
      self, model, end: int, seq_len: int,
  ) -> int:
    """Forward span for Hub-style block scope; else ``end`` (legacy).

    Hub densifies the full current attention block then slices the denoise
    window. Sub-window ``end`` must therefore be rounded up to the block
    boundary when truncated / DualCache / confidence forwards are active.

    Uniform exception: never extend past ``end``. Masked can attend future
    MASK in the same attention block; Unif future sites are random noise and
    pollute single-stream / sticky sub-block decode (``sub_block_size=8``).
    """
    if not self._use_block_scope():
      return end
    if not self.is_masked:
      return end
    bs = getattr(model, 'block_size', None)
    if bs is None:
      return end
    return self._attention_block_end(end, int(bs), seq_len)

  def _reset_nfe_stats(self) -> None:
    self._nfe_stats = {
        'n_forwards': 0,
        'n_thr_commits': 0,
        'n_force_max_commits': 0,
        'n_commits': 0,
    }
    self.last_nfe_stats = None

  def _bump_nfe(self, key: str, n: int = 1) -> None:
    if not self._nfe_stats:
      self._reset_nfe_stats()
    self._nfe_stats[key] = int(self._nfe_stats.get(key, 0)) + int(n)

  def _logits(
      self, model, xt: torch.Tensor, x0: torch.Tensor,
      *, active_end: int | None = None,
      window: tuple[int, int] | None = None,
  ) -> tuple[torch.Tensor, ShiftMode]:
    """Decode logits; BlockGen generate packing, Hub ss, or dual train graph.

    Returns ``(logits, shift_mode)``. ``shift_mode='window'`` only on the
    DualCache *replace* path (zeros outside the denoise window); prefill /
    dense / hierarchical full forwards use ``'full'``.
    """
    self._bump_nfe('n_forwards', 1)
    bs = getattr(model, 'block_size', None)
    use_dc = (
        self.use_block_cache and self.hierarchical_kv
        and active_end is not None
        and window is not None)

    # Truncate to attention-block end whenever Hub block-scope is active
    # (not only hierarchical_kv) so baseline+threshold / hubmatch do not
    # attend future MASK tokens outside the current block.
    a = active_end if (
        self._use_block_scope()
        and active_end is not None
        and active_end < xt.shape[1]) else None

    if self.single_stream_decode:
      if use_dc and hasattr(model.backbone, 'block_eval_prefill'):
        w0, w1 = window
        self._maybe_refresh_dual_cache(
            model, xt, window_start=w0, active_end=active_end)
        if self._dual_cache is None:
          logits, self._dual_cache = model.backbone.block_eval_prefill(
              xt, active_len=active_end, block_size=bs)
          return logits, 'full'
        return model.backbone.block_eval_replace(
            xt, active_len=active_end, window=(w0, w1),
            cache=self._dual_cache, block_size=bs), 'window'
      if hasattr(model.backbone, 'block_eval_logits'):
        return model.backbone.block_eval_logits(
            xt, active_len=a, block_size=bs), 'full'
      # Fallback: dual-stream with x0 tracking xt (quality-equivalent).
      return model.backbone_logits(
          xt, x0, active_len=a if a is not None else active_end), 'full'

    use_dc_dual = use_dc and hasattr(model.backbone, 'block_diff_prefill')
    if use_dc_dual:
      w0, w1 = window
      self._maybe_refresh_dual_cache(
          model, xt, window_start=w0, active_end=active_end)
      if self._dual_cache is None:
        logits, self._dual_cache = model.backbone.block_diff_prefill(
            xt, x0, active_len=active_end, block_size=bs)
        return logits, 'full'
      return model.backbone.block_diff_replace(
          xt, x0, active_len=active_end, window=(w0, w1),
          cache=self._dual_cache, block_size=bs), 'window'

    # BlockGen generate packing (baseline / hierarchical open-loop):
    # forward(x0=clean_prefix, xt=noisy_window) — not equal-length dual.
    # Scientific match to third_party/blockgen samplers._run_block_model.
    gen_fn = getattr(model.backbone, 'block_gen_logits', None)
    if (gen_fn is not None and window is not None
        and self.hierarchical_kv and not self.use_block_cache):
      w0, w1 = window
      # Prefer clean committed tokens from x0 for the prefix; fall back to xt.
      prefix = x0[:, :w0] if w0 > 0 else xt[:, :0]
      xt_b = xt[:, w0:w1]
      # Match BlockGen: packing must use the trained x0_causal flag.
      # Hard bake trains token-causal clean stream; default False here
      # silently decoded block-causal and nullified that lever.
      x0_causal = bool(getattr(model.backbone, 'x0_causal', False))
      if not x0_causal:
        x0_causal = bool(getattr(model, 'x0_causal', False))
      if not x0_causal:
        algo = getattr(getattr(model, 'config', None), 'algo', None)
        x0_causal = bool(getattr(algo, 'x0_causal', False))
      block_logits = gen_fn(
          prefix, xt_b, block_size=bs, x0_causal=x0_causal)
      # Scatter into full-seq buffer so step_fn window indexing is unchanged.
      bsz, seq_len = xt.shape
      v = block_logits.shape[-1]
      full = torch.zeros(
          bsz, seq_len, v, device=block_logits.device, dtype=block_logits.dtype)
      full[:, w0:w1] = block_logits
      # Outside window is zero-padded (like DualCache replace) — window shift.
      return full, 'window'

    if a is not None:
      return model.backbone_logits(xt, x0, active_len=a), 'full'
    return model.backbone_logits(xt, x0), 'full'

  def _shift_align_enabled(self, model) -> bool:
    if self._align_shift_logits_override is not None:
      return bool(self._align_shift_logits_override)
    return bool(getattr(model, 'shift_loss_targets', False))

  def _ar_block_bridge_enabled(self, model) -> bool:
    """Whether to run Hub's post-block AR append (masked Fast-dLLM path)."""
    if self._ar_block_bridge_override is not None:
      return bool(self._ar_block_bridge_override)
    # Auto: Hub parity for masked + shift; never on uniform / ARPC by default
    # (BlockGen absorb ARPC remasks inside the block; bridge is a different recipe).
    if self.use_arpc:
      return False
    return self.is_masked and self._shift_align_enabled(model)

  def _maybe_refresh_dual_cache(
      self,
      model,
      xt: torch.Tensor,
      *,
      window_start: int,
      active_end: int | None,
  ) -> None:
    """Invalidate DualCache like Hub ``block_past_key_values`` refresh.

    Hub ``generation_functions.py`` ~102: refresh when cache is missing or the
    first token of the current small block is still MASK. Also drop on
    ``active_len`` mismatch (attention-block span changed).
    """
    if self._dual_cache is None or not self.use_block_cache:
      return
    cache_len = getattr(self._dual_cache, 'active_len', None)
    # Only compare when the cache object carries ``active_len`` (real DualCache).
    if (active_end is not None and cache_len is not None
        and int(cache_len) != int(active_end)):
      self._dual_cache = None
      return
    # Uniform redraws change tokens without a MASK signal — Hub's "still MASK
    # at window start" refresh never fires, leaving stale K/V for rewritten
    # sites. Always invalidate on bare uniform/hybrid.
    if not self.is_masked:
      self._dual_cache = None
      return
    mask_id = getattr(model, 'mask_id', None)
    if (mask_id is not None
        and (xt[:, window_start] == mask_id).any()):
      self._dual_cache = None

  def _apply_x0_temperature(self, probs: torch.Tensor) -> torch.Tensor:
    """Sharpen / flatten categorical ``p(x0)`` before ancestral draws."""
    temp = float(self.x0_temperature)
    if abs(temp - 1.0) < 1e-8:
      return probs
    # p' ∝ p^(1/T); T<1 → sharper (quieter ancestral).
    sharpened = probs.clamp(min=1e-12).pow(1.0 / temp)
    return sharpened / sharpened.sum(dim=-1, keepdim=True).clamp(min=1e-12)

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

  def _prepare_masked_logits(
      self,
      model,
      logits: torch.Tensor,
      *,
      window: tuple[int, int] | None = None,
      shift_mode: ShiftMode = 'full',
  ) -> torch.Tensor:
    # Window-local shift ONLY for DualCache replace (zero-padded outsides).
    # Dense / prefill keep full-tensor shift even when a denoise window is set.
    logits = align_shift_logits(
        logits,
        enabled=self._shift_align_enabled(model),
        window=window if shift_mode == 'window' else None,
        mode=shift_mode,
    )
    if not self.ban_mask_pad_logits:
      return logits
    logits = logits.clone()
    neg = float(getattr(model, 'neg_infinity', -1e6))
    # Ban reserved specials that must not appear as content predictions.
    # EOS/im_end stays allowed so the model can stop.
    for mid in self._uniform_exclude_ids(model):
      if 0 <= int(mid) < logits.size(-1):
        logits[..., int(mid)] = neg
    # Masked / hubmatch may leave exclude empty of pad; still ban pad+mask.
    if getattr(model, 'mask_id', None) is not None:
      logits[..., model.mask_id] = neg
    pad_id = getattr(getattr(model, 'tokenizer', None), 'pad_token_id', None)
    if pad_id is not None:
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

    raw, shift_mode = self._logits(
        model, xt, x0, active_end=active_end, window=window)
    logits = self._prepare_masked_logits(
        model, raw, window=window, shift_mode=shift_mode)
    # Truncated / DualCache forward may return length active_end — pad back.
    if logits.size(1) < seq_len:
      pad = torch.zeros(
          b, seq_len - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    p_x0 = self._apply_x0_temperature(p_x0)
    p_x0 = self._apply_top_p(p_x0, self.p_nucleus)
    if getattr(self, '_greedy_decode', False):
      sampled = p_x0.argmax(dim=-1)
    else:
      sampled = sample_categorical(p_x0)
    is_masked = xt == model.mask_id
    # Hub confidence unmask is scoped to the active small-block / denoise
    # window. Full-sequence force-max can "commit" future MASK positions that
    # are then discarded by ``xt[:, start:end] = …`` → zero progress (hubmatch
    # crash) or DualCache zero-logit ``!`` at the window edge.
    if window is not None:
      w0, w1 = window
      active = torch.zeros_like(is_masked)
      active[:, w0:w1] = True
      is_masked = is_masked & active
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

  def _uniform_simplex_mode(self, model) -> str:
    from ..forward_process.utils import normalize_uniform_simplex_mode
    cfg = getattr(model, 'config', None)
    algo = getattr(cfg, 'algo', None) if cfg is not None else None
    raw = getattr(algo, 'uniform_simplex_mode', None)
    if raw is None:
      raw = getattr(model, 'uniform_simplex_mode', 'conversion')
    return normalize_uniform_simplex_mode(raw)

  def _uniform_exclude_ids(self, model) -> tuple[int, ...]:
    from ..forward_process.utils import resolve_uniform_exclude_ids
    tok = getattr(model, 'tokenizer', None)
    mode = self._uniform_simplex_mode(model)
    if tok is None:
      if mode == 'blockgen':
        return ()
      mid = getattr(model, 'mask_id', None)
      return (int(mid),) if mid is not None else ()
    return resolve_uniform_exclude_ids(
        tok,
        mask_id=getattr(model, 'mask_id', None),
        vocab_size=int(model.vocab_size),
        mode=mode,
    )

  def _uniform_noise(
      self,
      model,
      shape: tuple[int, ...] | torch.Size,
      *,
      device: torch.device,
      dtype: torch.dtype,
  ) -> torch.Tensor:
    """Unif over content tokens (exclude MASK/PAD/…); blockgen → full V."""
    from ..forward_process.utils import (
        resolve_uniform_noise_redraw_exclude_ids,
        sample_uniform_excluding_mask,
    )
    tok = getattr(model, 'tokenizer', None)
    mid = getattr(model, 'mask_id', None)
    mode = self._uniform_simplex_mode(model)
    if tok is None:
      exclude = () if mode == 'blockgen' else (
          (int(mid),) if mid is not None else ())
    else:
      exclude = resolve_uniform_noise_redraw_exclude_ids(
          tok, mask_id=mid, vocab_size=int(model.vocab_size), mode=mode)
    draw_mid = None if mode == 'blockgen' else mid
    return sample_uniform_excluding_mask(
        shape,
        vocab_size=int(model.vocab_size),
        mask_id=draw_mid,
        device=device,
        dtype=dtype,
        exclude_ids=exclude,
    )

  def _uniform_v_eff(self, model) -> int:
    """Simplex size for DUO coefficients."""
    from ..forward_process.utils import uniform_simplex_size
    exclude = self._uniform_exclude_ids(model)
    return uniform_simplex_size(
        int(model.vocab_size),
        None,
        exclude_ids=exclude,
    )

  def _uniform_limiting(self, model, *, device, dtype) -> torch.Tensor:
    """``1/V_eff`` on content ids; 0 on reserved specials."""
    v = int(model.vocab_size)
    v_eff = self._uniform_v_eff(model)
    exclude = self._uniform_exclude_ids(model)
    u = torch.full((1, 1, v), 1.0 / max(v_eff, 1), device=device, dtype=dtype)
    for mid in exclude:
      if 0 <= int(mid) < v:
        u[..., int(mid)] = 0.0
    return u

  def _sample_uniform_posterior_fast(
      self,
      p_x0: torch.Tensor,
      xt: torch.Tensor,
      alpha_s: torch.Tensor,
      alpha_t: torch.Tensor,
      *,
      v_eff: int,
      noise_removal_step: bool,
      model,
  ) -> torch.Tensor:
    """BlockGen ``sample_uniform_posterior`` (no full V×L materialize).

    ``alpha_s`` / ``alpha_t`` are ``[B, L, 1]``; ``p_x0`` is ``[B, L, V]``.
    Uniform redraws use ``Unif(V\\{MASK})``.
    """
    # Squeeze channel dim for gather arithmetic.
    a_t = alpha_t.squeeze(-1)
    a_s = alpha_s.squeeze(-1)
    p_xt = torch.gather(p_x0, -1, xt.unsqueeze(-1)).squeeze(-1)
    denom = (a_t * v_eff * p_xt + (1.0 - a_t)).clamp(min=1e-12)
    sampled_x0 = sample_categorical(p_x0)
    thr = torch.rand_like(p_xt)
    if noise_removal_step:
      keep_xt = (a_t * v_eff * p_xt / denom).clamp(0.0, 1.0)
      return torch.where(thr < keep_xt, xt, sampled_x0)
    alpha_ts = a_t / a_s.clamp(min=1e-8)
    sample_unif = ((1.0 - alpha_ts) * (1.0 - a_s) / denom).clamp(0.0, 1.0)
    keep_xt = (
        (a_t * v_eff * p_xt + alpha_ts - a_t) / denom).clamp(0.0, 1.0)
    unif = self._uniform_noise(
        model, xt.shape, device=xt.device, dtype=xt.dtype)
    keep_or_x0 = torch.where(
        (sample_unif + keep_xt).clamp(max=1.0) > thr, xt, sampled_x0)
    return torch.where(sample_unif > thr, unif, keep_or_x0)

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
    v = int(model.vocab_size)
    v_eff = self._uniform_v_eff(model)
    alpha_t = self._expand_alpha(model, t_scalar, seq_len).unsqueeze(-1)
    noise_removal = dt is None
    if noise_removal:
      alpha_s = torch.ones_like(alpha_t)
    else:
      t_prev = (t_scalar - dt).clamp(min=0.0)
      alpha_s = self._expand_alpha(model, t_prev, seq_len).unsqueeze(-1)

    raw, shift_mode = self._logits(
        model, xt, x0, active_end=active_end, window=window)
    # Same prep as masked: shift-align when train used shift_loss_targets, and
    # ban MASK/PAD. Previously skipped here → U0+shift decode was unaligned
    # and MASK mass could enter ancestral draws.
    logits = self._prepare_masked_logits(
        model, raw, window=window, shift_mode=shift_mode)
    if logits.size(1) < seq_len:
      pad = torch.zeros(
          b, seq_len - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    if getattr(self.config.sampling, 'use_float64', False):
      p_x0 = p_x0.to(torch.float64)
      alpha_t = alpha_t.to(torch.float64)
      alpha_s = alpha_s.to(torch.float64)
    p_x0 = self._apply_x0_temperature(p_x0)

    # Hub DualCache conf remask analog: propose = argmax(p_x0) (T=0),
    # commit ≥thr + force-max, else keep Unif. Skip Duo posterior — it
    # would scribble samples onto undecided sites (Hub leaves MASK).
    if self.uniform_confidence_sticky:
      if self.p_nucleus < 1.0:
        p_x0 = self._apply_top_p(p_x0, self.p_nucleus)
      return self._apply_uniform_sticky(xt, p_x0, xt, window=window)

    if self.posterior_sampler == 'fast' and not getattr(
        self, '_greedy_decode', False):
      if self.p_nucleus < 1.0:
        p_x0 = self._apply_top_p(p_x0, self.p_nucleus)
      return self._sample_uniform_posterior_fast(
          p_x0, xt, alpha_s, alpha_t,
          v_eff=v_eff, noise_removal_step=noise_removal, model=model)

    # Naive path (materialize q_xs) — Duo / previous BlockSampler.
    alpha_ts = alpha_t / alpha_s.clamp(min=1e-8)
    xt_one_hot = F.one_hot(xt, v).to(p_x0.dtype)
    uniform = self._uniform_limiting(model, device=xt.device, dtype=p_x0.dtype)
    numerator = (
        (alpha_t * v_eff * p_x0 * xt_one_hot)
        + ((alpha_ts - alpha_t) * xt_one_hot)
        + ((alpha_s - alpha_t) * p_x0)
        + ((1 - alpha_ts) * (1 - alpha_s) * uniform)
    )
    denom = (
        alpha_t * v_eff * torch.gather(p_x0, -1, xt.unsqueeze(-1))
    ) + (1 - alpha_t)
    q_xs = numerator / denom.clamp(min=1e-12)
    q_xs = self._apply_top_p(q_xs, self.p_nucleus)
    if getattr(self, '_greedy_decode', False):
      return q_xs.argmax(dim=-1)
    return sample_categorical(q_xs)

  def _apply_uniform_sticky(
      self,
      xs: torch.Tensor,
      p_x0: torch.Tensor,
      xt: torch.Tensor,
      *,
      window: tuple[int, int] | None,
  ) -> torch.Tensor:
    """Hub DualCache conf remask analog on USDM (Unif undecided).

    Mirrors ``_masked_step`` confidence path at T=0: propose argmax,
    conf = peak ``p_x0``, commit ≥thr + force-max ≥1/row. Non-commits
    keep ``xt`` (Unif), matching Hub leaving MASK. ``xs`` is ignored
    (kept for call-site compat). Not DualCache K/V.
    """
    if not self.uniform_confidence_sticky:
      return xs
    b, seq_len = xt.shape
    frozen = self._block_sticky_frozen
    if frozen is None or frozen.shape != xt.shape:
      frozen = torch.zeros_like(xt, dtype=torch.bool)
      self._block_sticky_frozen = frozen

    active = torch.ones(b, seq_len, dtype=torch.bool, device=xt.device)
    if window is not None:
      w0, w1 = window
      active = torch.zeros_like(active)
      active[:, w0:w1] = True

    # Hub: start from current state (MASK/Unif); only write on commit.
    out = xt.clone()

    conf = p_x0.max(dim=-1).values
    hard = p_x0.argmax(dim=-1)
    cand = active & ~frozen
    thr_stick = torch.zeros_like(cand)
    # Ranking scores for force-max site: confidence, random, or L→R.
    order = str(getattr(self, 'uniform_commit_order', 'confidence'))
    if order == 'random' or self.uniform_commit_random:
      rank = torch.rand_like(conf)
    elif order == 'ltr':
      # Prefer leftmost candidate: low index = high rank.
      pos = torch.arange(seq_len, device=xt.device, dtype=conf.dtype)
      rank = -pos.unsqueeze(0).expand_as(conf)
    else:
      rank = conf
    if self.unmask_threshold is not None:
      # thr still uses true confidence (thr=1 stays nearly inert on D2).
      thr_stick = cand & (conf >= float(self.unmask_threshold))
    new_stick = thr_stick.clone()
    # Force-max among candidates, gated by sticky_min_conf (UCC).
    rank_c = rank.masked_fill(~cand, float('-inf'))
    max_idx = rank_c.argmax(dim=-1)
    rows = torch.arange(b, device=xt.device)
    still = cand.any(dim=-1)
    force_stick = torch.zeros_like(cand)
    if still.any():
      peak = conf[rows[still], max_idx[still]]
      allow = peak >= float(self.sticky_min_conf)
      if allow.any():
        sel_rows = rows[still][allow]
        sel_idx = max_idx[still][allow]
        force_stick[sel_rows, sel_idx] = True
    # Force-max only counts sites not already thr-committed this step.
    force_only = force_stick & cand & ~thr_stick
    new_stick = (thr_stick | force_stick) & cand
    n_thr = int(thr_stick.sum().item())
    n_force = int(force_only.sum().item())
    if n_thr:
      self._bump_nfe('n_thr_commits', n_thr)
    if n_force:
      self._bump_nfe('n_force_max_commits', n_force)
    self._bump_nfe('n_commits', n_thr + n_force)

    out = torch.where(new_stick, hard, out)
    self._block_sticky_frozen = frozen | new_stick
    return out

  def _uniform_commit_revise_low_conf(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      start: int,
      end: int,
      *,
      active_end: int,
  ) -> int:
    """BlockGen-inspired corrector: re-Unif frozen sites the model regrets.

    Uses token confidence ``p_x0[xt]`` (diffusion confidence). Low-conf
    commits are unfrozen and redrawn from Unif — then UCC re-commits.
    Returns number of sites revised.
    """
    if not self.uniform_commit_revise or self._block_sticky_frozen is None:
      return 0
    frozen = self._block_sticky_frozen
    if not bool(frozen[:, start:end].any()):
      return 0
    raw, shift_mode = self._logits(
        model, xt, x0, active_end=active_end, window=(start, end))
    logits = self._prepare_masked_logits(
        model, raw, window=(start, end), shift_mode=shift_mode)
    if logits.size(1) < xt.shape[1]:
      pad = torch.zeros(
          xt.shape[0], xt.shape[1] - logits.size(1), logits.size(-1),
          device=logits.device, dtype=logits.dtype)
      logits = torch.cat([logits, pad], dim=1)
    p_x0 = F.log_softmax(logits, dim=-1).exp()
    p_x0 = self._apply_x0_temperature(p_x0)
    tok_conf = p_x0.gather(-1, xt.unsqueeze(-1)).squeeze(-1)
    low = frozen[:, start:end] & (
        tok_conf[:, start:end] < float(self.uniform_commit_revise_tau))
    n = int(low.sum().item())
    if n == 0:
      return 0
    unif = self._uniform_noise(
        model, xt.shape, device=xt.device, dtype=xt.dtype)
    block = xt[:, start:end].clone()
    block = torch.where(low, unif[:, start:end], block)
    xt[:, start:end] = block
    x0[:, start:end] = block
    frozen[:, start:end] = frozen[:, start:end] & ~low
    self._block_sticky_frozen = frozen
    return n

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
    active_end = self._truncated_active_end(model, end, xt.shape[1])
    raw, _shift_mode = self._logits(
        model, xt, x0, active_end=active_end, window=(start, end))
    logits = self._scale_logits(raw[:, start:end])
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
    """BlockGen predictor–corrector: clean proposal → score → re-noise top-k.

    Absorb (masked): re-noise = write ``mask_id``. Uniform: re-noise = Unif(V).
    Paper TinyGSM pin uses ``ar_metric`` + ``nll`` (AR LL of the proposal).
    """
    block_len = end - start
    alpha_t = self._expand_alpha(model, t_scalar, xt.shape[1])
    t_prev = (t_scalar - dt).clamp(min=0.0)
    alpha_s = self._expand_alpha(model, t_prev, xt.shape[1])

    active_end = self._truncated_active_end(model, end, xt.shape[1])
    raw, shift_mode = self._logits(
        model, xt, x0, active_end=active_end, window=(start, end))
    # BlockGen scales logits by a single TEMP; we expose ARPC + x0 knobs.
    # Apply ARPC temperature on logits, then x0 temperature on p(x0).
    logits = self._scale_logits(raw)
    logits = self._prepare_masked_logits(
        model, logits, window=(start, end), shift_mode=shift_mode)
    p_x0 = F.log_softmax(logits[:, start:end], dim=-1).exp()
    p_x0 = self._apply_x0_temperature(p_x0)
    log_p = p_x0.clamp(min=1e-12).log()
    # Predictor: BlockGen samples the *posterior* at alpha_s=1
    # (noise_removal_step=True), not raw categorical p(x0). That keeps
    # high-conf xt with keep_xt_prob instead of always redrawing.
    ones = torch.ones_like(alpha_t[:, start:end])
    a_t = alpha_t[:, start:end].unsqueeze(-1)
    a_s = ones.unsqueeze(-1)
    if self.is_masked:
      # Absorbing: noise_removal ⇒ denoise every still-masked site.
      sampled_x0 = sample_categorical(p_x0)
      is_masked = xt[:, start:end] == model.mask_id
      proposal = torch.where(is_masked, sampled_x0, xt[:, start:end])
    else:
      proposal = self._sample_uniform_posterior_fast(
          p_x0, xt[:, start:end], a_s, a_t,
          v_eff=self._uniform_v_eff(model),
          noise_removal_step=True,
          model=model)
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
      noisy = self._uniform_noise(
          model, (xt.shape[0], idxs.shape[1]),
          device=xt.device, dtype=xt.dtype)
    block = xt[:, start:end].clone()
    block.scatter_(1, idxs, noisy)
    xt[:, start:end] = block
    x0[:, start:end] = block

  @staticmethod
  def _eos_stop_ready(
      xt: torch.Tensor,
      *,
      gen_start: int,
      end: int,
      eos_id: int,
      mask_id: int | None,
  ) -> bool:
    """Hub early-stop: every row has EOS with no MASK before it in gen span."""
    if end <= gen_start:
      return False
    span = xt[:, gen_start:end]
    has_eos = (span == eos_id).any(dim=-1)
    if not bool(has_eos.all()):
      return False
    if mask_id is None:
      return True
    # For each row: no MASK strictly before the first EOS.
    b = span.shape[0]
    for i in range(b):
      row = span[i]
      eos_pos = (row == eos_id).nonzero(as_tuple=False)
      if eos_pos.numel() == 0:
        return False
      first = int(eos_pos[0].item())
      if (row[:first] == mask_id).any():
        return False
    return True

  def _gen_start(self, prefix_len: int, inject_bos: bool) -> int:
    return prefix_len if prefix_len > 0 else (1 if inject_bos else 0)

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
    # Hub AR block-bridge seeds the first token of the next attention block
    # (generation_functions.py ~83-85 / ~130-140). Preserve non-MASK tokens
    # already present in this window so `_init_block` does not wipe the seed.
    seeded = None
    if self.is_masked:
      seeded = x0[:, start:end].clone()
      xt[:, start:end] = model.mask_id
    else:
      xt[:, start:end] = self._uniform_noise(
          model, (xt.shape[0], end - start),
          device=xt.device, dtype=xt.dtype)
    # When inject_bos=False (lm-eval continuation), do not force BOS at pos 0.
    if self._ignore_bos(model) and start == 0 and inject_bos:
      bos = model.tokenizer.bos_token_id
      if bos is not None:
        xt[:, 0] = bos
        x0[:, 0] = bos
    xt[:, :start] = committed
    x0[:, :start] = committed
    if seeded is not None:
      keep = seeded != model.mask_id
      if keep.any():
        xt[:, start:end] = torch.where(keep, seeded, xt[:, start:end])
    x0[:, start:end] = xt[:, start:end]
    if self.use_arpc and self.arpc_use_prefix_fill:
      xt, x0 = self._arpc_prefix_fill(model, xt, x0, start, end)
    return xt, x0

  def _ar_block_bridge(
      self,
      model,
      xt: torch.Tensor,
      x0: torch.Tensor,
      end: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One AR step after a fully unmasked attention block (Hub ~81-85).

    Hub forwards the completed block, takes ``logits[:, -1].argmax``, and
    appends that token (seeds the next block). Fixed-length: write into
    position ``end`` when it is still MASK. Uses *unshifted* logits so that
    with ``shift_loss_targets`` position ``end-1`` predicts token ``end``.
    """
    n = xt.shape[1]
    if end < 1 or end >= n:
      return xt, x0
    # Dense / hierarchical full forward — not DualCache replace.
    self._dual_cache = None
    bs = getattr(model, 'block_size', None)
    if self.single_stream_decode and hasattr(model.backbone, 'block_eval_logits'):
      logits = model.backbone.block_eval_logits(
          xt, active_len=end, block_size=bs)
    elif self.hierarchical_kv:
      logits = model.backbone_logits(xt, x0, active_len=end)
    else:
      logits = model.backbone_logits(xt, x0)
    if logits.size(1) < end:
      raise RuntimeError(
          f'AR block-bridge expected >= {end} logits, got {logits.size(1)}')
    logits = logits.clone()
    if self.ban_mask_pad_logits:
      if getattr(model, 'mask_id', None) is not None:
        neg = float(getattr(model, 'neg_infinity', -1e6))
        logits[..., model.mask_id] = neg
      pad_id = getattr(getattr(model, 'tokenizer', None), 'pad_token_id', None)
      if pad_id is not None:
        logits[..., int(pad_id)] = float('-inf')
    next_tok = logits[:, end - 1, :].argmax(dim=-1)
    is_mask = xt[:, end] == model.mask_id
    if is_mask.any():
      xt[is_mask, end] = next_tok[is_mask]
      x0[is_mask, end] = next_tok[is_mask]
    return xt, x0

  def _emit_step_hook(
      self,
      model,
      xt: torch.Tensor,
      *,
      start: int,
      end: int,
      active_end: int,
      before_window: torch.Tensor | None,
      t_scalar: torch.Tensor | None,
      step_dt: float | None,
      local_step: int,
      phase: str,
  ) -> None:
    """Fire ``step_hook`` with a CPU-friendly decode event (batch 0)."""
    hook = self.step_hook
    if hook is None:
      return
    row = xt[0].detach()
    mask_id = getattr(model, 'mask_id', None)
    window = row[start:end]
    before = None if before_window is None else before_window[0].detach()
    if mask_id is not None and self.is_masked:
      still_masked = (window == mask_id).nonzero(as_tuple=False).view(-1)
      still_masked = (still_masked + start).tolist()
      if before is not None:
        newly = ((before == mask_id) & (window != mask_id)).nonzero(
            as_tuple=False).view(-1)
        newly = (newly + start).tolist()
      else:
        newly = []
      changed = newly
    else:
      still_masked = []
      if before is not None:
        changed = (before != window).nonzero(as_tuple=False).view(-1)
        changed = (changed + start).tolist()
        newly = changed
      else:
        newly = []
        changed = []
    committed = list(range(0, start))
    if mask_id is not None and self.is_masked:
      in_win = (window != mask_id).nonzero(as_tuple=False).view(-1)
      committed.extend((in_win + start).tolist())
    else:
      committed.extend(range(start, end))
    t_val = None
    if t_scalar is not None and t_scalar.numel() > 0:
      t_val = float(t_scalar[0].detach().cpu())
    event = {
        'global_step': int(self._decode_trace_step),
        'local_step': int(local_step),
        'phase': phase,
        'window_start': int(start),
        'window_end': int(end),
        'active_end': int(active_end),
        't': t_val,
        'dt': None if step_dt is None else float(step_dt),
        'mode': str(self.mode),
        'token_ids': row.detach().cpu().tolist(),
        'still_masked_positions': still_masked,
        'newly_committed_positions': newly,
        'changed_positions': changed,
        'committed_positions': committed,
        'active_positions': newly if newly else list(range(start, end)),
    }
    self._decode_trace_step += 1
    hook(event)

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

    # Do NOT reset DualCache here — Hub keeps ``block_past_key_values`` across
    # small blocks in the same attention block (generation_functions.py ~71,
    # ~101-108). ``generate`` clears on new attention block; ``_logits``
    # refreshes when the first small-block token is still MASK.
    window = (start, end)
    # Prefill / hierarchical forward span = attention-block end (Hub), while
    # commit / unmask stay on the sub-window ``(start, end)``.
    active_end = self._truncated_active_end(model, end, xt.shape[1])
    self._maybe_refresh_dual_cache(
        model, xt, window_start=start, active_end=active_end)
    # Reset sticky freeze for this sub-window (prefix stays frozen via
    # ``_restore``; only active sites accumulate commits).
    if self.uniform_confidence_sticky and not self.is_masked:
      self._block_sticky_frozen = torch.zeros(
          xt.shape, dtype=torch.bool, device=xt.device)
      if start > 0:
        self._block_sticky_frozen[:, :start] = True
    else:
      self._block_sticky_frozen = None

    # Tokens already filled by ARPC prefix fill stay protected across reverse
    # steps (and in blockgen guided corruption).
    block_prefix_len = 0
    if self.use_arpc and self.arpc_use_prefix_fill:
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

    def _apply(
        t_scalar: torch.Tensor,
        step_dt: float | None,
        *,
        local_step: int = 0,
        phase: str = 'ancestral',
    ) -> None:
      _restore()
      x0[:, start:end] = xt[:, start:end]
      before = None
      if self.track_revisions and not self.is_masked:
        before = xt[:, start:end].clone()
      before_hook = (
          xt[:, start:end].clone() if self.step_hook is not None else None)
      xt_new = step_fn(
          model, xt, x0, t_scalar, step_dt,
          active_end=active_end, window=window)
      if before is not None and self._revision_stats is not None:
        chunk = xt_new[:, start:end]
        changed = (chunk != before).sum().item()
        self._revision_stats['token_changes'] += float(changed)
        self._revision_stats['token_slots'] += float(chunk.numel())
        self._revision_stats['steps_tracked'] += 1.0
      xt[:, start:end] = xt_new[:, start:end]
      _restore()
      x0[:, start:end] = xt[:, start:end]
      self._emit_step_hook(
          model, xt,
          start=start, end=end, active_end=active_end,
          before_window=before_hook, t_scalar=t_scalar, step_dt=step_dt,
          local_step=local_step, phase=phase)

    use_blockgen = self.use_arpc and self.arpc_mode == 'blockgen'
    if use_blockgen and self.is_masked and self.unmask_threshold is not None:
      raise ValueError(
          'masked BlockGen ARPC requires sampling.unmask_threshold=null '
          '(confidence unmask loop bypasses the ARPC predictor–corrector)')

    # Fast-dLLM confidence decode: iterate until the block has no masks
    # (Hub generate while-loop), not a fixed α-schedule of length ``steps``.
    # Bound = remaining mask count (codex-fixes): each pass must make progress.
    if self.is_masked and self.unmask_threshold is not None:
      t_full = torch.ones(xt.shape[0], device=xt.device)
      max_iters = max(int((xt[:, start:end] == model.mask_id).sum().item()), 1)
      max_iters = max(max_iters, end - start)
      for conf_i in range(max_iters):
        if not (xt[:, start:end] == model.mask_id).any():
          break
        before = int((xt[:, start:end] == model.mask_id).sum().item())
        _apply(t_full, None, local_step=conf_i, phase='confidence')
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

    # UCC: Hub DualCache conf-until loop (Unif undecided). Optional revise
    # is off by default. DualCache K/V remains MASK-only.
    if (self.uniform_confidence_sticky and not self.is_masked
        and self.unmask_threshold is not None):
      t_full = torch.ones(xt.shape[0], device=xt.device)
      # Batch-aware budget (mirrors masked confidence path). Old bound
      # ``end-start`` under-counted when B>1 or sticky_min_conf blocked
      # force-max on weak rows while other rows still "progressed".
      def _ucc_unfrozen_count() -> int:
        frozen = self._block_sticky_frozen
        if frozen is None:
          return max((end - start) * xt.shape[0], 1)
        return int((~frozen[:, start:end]).sum().item())

      def _ucc_until_frozen(phase_prefix: str, budget: int) -> None:
        budget = max(int(budget), _ucc_unfrozen_count(), end - start, 1)

        def _flush_force_max(local_step: int, tag: str) -> None:
          """Drop sticky_min_conf so force-max can always commit ≥1/row."""
          old_min = self.sticky_min_conf
          self.sticky_min_conf = 0.0
          try:
            _apply(
                t_full, None, local_step=local_step,
                phase=f'{phase_prefix}_{tag}')
          finally:
            self.sticky_min_conf = old_min

        for conf_i in range(budget):
          frozen = self._block_sticky_frozen
          if frozen is not None and bool(frozen[:, start:end].all()):
            return
          before = _ucc_unfrozen_count()
          _apply(
              t_full, None, local_step=conf_i,
              phase=f'{phase_prefix}_{conf_i}')
          after = _ucc_unfrozen_count()
          if after >= before:
            # Stall: usually sticky_min_conf gated force-max on diffuse p_x0.
            # Flush once and continue (do NOT return — flush is one commit
            # per row; a B×W window may need many flushes).
            if self.sticky_min_conf > 0:
              _flush_force_max(conf_i, 'flush')
              continue
            raise RuntimeError(
                'uniform confidence commit failed to make progress '
                f'(unfrozen {before} → {after} in [{start},{end}))')
        frozen = self._block_sticky_frozen
        if frozen is not None and bool(frozen[:, start:end].all()):
          return
        # Final guarantee: sticky_min_conf=0 until every site freezes.
        remaining = _ucc_unfrozen_count()
        for conf_i in range(max(remaining + 2, 1)):
          if bool(self._block_sticky_frozen[:, start:end].all()):
            return
          _flush_force_max(conf_i, 'final_flush')
        if not bool(self._block_sticky_frozen[:, start:end].all()):
          raise RuntimeError(
              'uniform confidence commit exhausted its bound '
              f'in [{start},{end})')

      _ucc_until_frozen('ucc', _ucc_unfrozen_count())
      if self.uniform_commit_revise:
        n_rev = self._uniform_commit_revise_low_conf(
            model, xt, x0, start, end, active_end=active_end)
        if n_rev > 0:
          _restore()
          x0[:, start:end] = xt[:, start:end]
          # Re-commit revised sites (budget = revised count + slack).
          _ucc_until_frozen('ucc_revise', max(n_rev + 2, 4))
      _restore()
      x0[:, start:end] = xt[:, start:end]
      return xt, x0

    for i in range(num_steps):
      t = timesteps[i].expand(xt.shape[0])
      is_last = (i == num_steps - 1)
      # BlockGen: last of N uses α_s=1 at α_t=timesteps[N-1] (still mid-noise
      # for N=32). We previously ran N ancestral steps then a near-noop final
      # at α_t≈eps — never matching BlockGen's noise-removal commit.
      is_guided = (
          use_blockgen
          and not is_last
          and i >= self.arpc_warmup_steps
          and (i - self.arpc_warmup_steps) % self.arpc_guide_every == 0)
      if is_guided:
        _restore()
        x0[:, start:end] = xt[:, start:end]
        before_hook = (
            xt[:, start:end].clone() if self.step_hook is not None else None)
        self._arpc_guided_step(
            model, xt, x0, start, end, t, dt,
            block_prefix_len=block_prefix_len)
        _restore()
        x0[:, start:end] = xt[:, start:end]
        self._emit_step_hook(
            model, xt,
            start=start, end=end, active_end=active_end,
            before_window=before_hook, t_scalar=t, step_dt=dt,
            local_step=i, phase='arpc_guided')
      else:
        _apply(
            t, None if is_last else dt,
            local_step=i,
            phase='final' if is_last else 'ancestral')

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
    # Uniform reverse: argmax(q_xs) preferentially keeps xt (prior noise) when
    # p_x0 is diffuse — locks multilingual soup into later blocks. UCC sticky
    # never samples q_xs (Hub argmax on p_x0); allow greedy there. Else
    # auto-disable (matches baseline / ancestral profiles).
    if (self._greedy_decode and not self.is_masked
        and not self.uniform_confidence_sticky):
      logger.warning(
          'sampling.greedy=true is invalid for uniform reverse '
          f'(forward_process_name={self.forward_process_name!r}): '
          'argmax(q_xs) locks prior noise. Forcing greedy=false '
          '(ancestral). Pass greedy=false explicitly to silence.')
      self._greedy_decode = False
    # Honest branch: thr without sticky is conf-remask on MASK only.
    # On uniform it is a dead pin (falls through to fixed-N ancestral).
    # Remask twin for Unif is uniform_commit / uniform_dual (UCC).
    if (not self.is_masked
        and self.unmask_threshold is not None
        and not self.uniform_confidence_sticky):
      logger.warning(
          'uniform decode: unmask_threshold=%s with '
          'uniform_confidence_sticky=false → ancestral (thr ignored). '
          'Masked thr runs confidence-until; Unif remask twin is '
          'uniform_commit/UCC. Prefer hierarchical_ancestral or '
          'uniform_commit for honest PROTOCOL.',
          self.unmask_threshold)
    # Re-assert truncation even if a profile pinned hierarchical_kv false
    # after __init__ (legacy baseline footgun). Escape: allow_full_seq_decode.
    _open_loop = (
        not self.allow_full_seq_decode
        and not self.use_block_cache
        and self.unmask_threshold is None
        and not self.uniform_confidence_sticky)
    if (self.forward_process_name in ('masked', 'uniform', 'hybrid')
        and _open_loop
        and not self.hierarchical_kv):
      logger.warning(
          '%s generate(): forcing hierarchical_kv '
          '(refuse full-seq dual open-loop packing).',
          self.forward_process_name)
      self.hierarchical_kv = True
    self._maybe_warn_arpc_mixture(model)
    if self.track_revisions:
      self.reset_revision_stats()
    self._decode_trace_step = 0
    self._reset_nfe_stats()

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
      # Hub: ``block_past_key_values = None`` at each attention block (~71).
      self._dual_cache = None
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
      early_stop = False
      for w_start, w_end in windows:
        xt, x0 = self._init_block(
            model, xt, x0, w_start, w_end, inject_bos=inject_bos)
        xt, x0 = self._denoise_block(
            model, xt, x0, w_start, w_end, num_steps, eps,
            inject_bos=inject_bos)
        if self.use_arpc and self.arpc_mode == 'simplified':
          xt = self._arpc_correct_block(model, xt, x0, w_start, w_end)
          x0[:, w_start:w_end] = xt[:, w_start:w_end]
        # Hub mid-block early stop after a small-block commit (~722-725).
        if self.stop_on_eos:
          tok = getattr(model, 'tokenizer', None)
          eos_id = getattr(tok, 'eos_token_id', None) if tok is not None else None
          if eos_id is not None and self._eos_stop_ready(
              xt,
              gen_start=self._gen_start(prefix_len, inject_bos),
              end=w_end,
              eos_id=int(eos_id),
              mask_id=(
                  getattr(model, 'mask_id', None) if self.is_masked else None),
          ):
            early_stop = True
            break
      x0[:, :end] = xt[:, :end]
      if prefix_len > 0:
        xt[:, :prefix_len] = prefix_ids[:, :prefix_len].to(xt.device)
        x0[:, :prefix_len] = xt[:, :prefix_len]

      # Hub AR block-bridge after the attention block is fully unmasked (~81-85).
      if (not early_stop
          and self._ar_block_bridge_enabled(model)
          and not (xt[:, denoise_start:end] == model.mask_id).any()):
        xt, x0 = self._ar_block_bridge(model, xt, x0, end)

      if early_stop:
        break
      # Early-stop at attention-block boundary (same EOS/MASK rule).
      if self.stop_on_eos:
        tok = getattr(model, 'tokenizer', None)
        eos_id = getattr(tok, 'eos_token_id', None) if tok is not None else None
        if eos_id is not None and self._eos_stop_ready(
            xt,
            gen_start=self._gen_start(prefix_len, inject_bos),
            end=end,
            eos_id=int(eos_id),
            mask_id=(
                getattr(model, 'mask_id', None) if self.is_masked else None),
        ):
          break

    generated_start = self._gen_start(prefix_len, inject_bos)
    if self.pad_after_eos:
      tok = getattr(model, 'tokenizer', None)
      xt = self._pad_after_eos(
          xt,
          start=generated_start,
          eos_id=getattr(tok, 'eos_token_id', None),
          pad_id=getattr(tok, 'pad_token_id', None),
      )
    self.last_nfe_stats = dict(self._nfe_stats)
    return xt


__all__ = ['BlockSampler']
