"""Unified block diffusion trainer (masked | uniform | hybrid forward processes).

Switch ``algo.forward_process_name`` between masked (absorbing), uniform
(BlockGen), and hybrid (mask + Unif mix). Does **not** use BD3LM or BlockDiT.
"""

from __future__ import annotations

import math

import hydra.utils
import omegaconf
import torch
import torch.nn.functional as F

from ..forward_process.block_masked import (
    BlockMaskedForwardProcess,
    complementary_pair_from_mask,
    sample_block_timesteps,
)
from ..forward_process.block_uniform import BlockUniformForwardProcess
from ..forward_process.block_hybrid import BlockHybridForwardProcess
from ..forward_process.utils import sample_uniform_excluding_mask
from ..noise_schedules import LogLinear
from .base import Loss, TrainerBase, ensure_mask_token
from .block_geometry import (
    parse_block_weights,
    sample_log_block_size,
    sizes_from_weights,
    validate_mixture_vs_weights,
)
from ..contracts.special_tokens import (
    SpecialTokenIds,
    assert_same_mask_id,
    ensure_special_tokens,
)
from ..losses.block_elbo import (
    masked_block_nll_per_token,
    masked_plain_ce_per_token,
    subs_log_probs,
    uniform_block_nll_per_token,
)


# Layer 3 — standing symmetry audit (see tests/test_loss_symmetry.py).
# Every training special-case that touches `_masked_loss` / `nll` must appear
# here with an explicit uniform decision. Adding a masked-only hook without
# updating this dict must fail CI.
LOSS_SPECIAL_CASE_POLICY: dict[str, str] = {
    # Fast-dLLM / Unifusion-style next-token shift: AR logit at i-1 scores
    # clean token at i. Allowed on masked (Fast-dLLM) and uniform (Unifusion
    # port). Hybrid requires explicit algo.shift_on_hybrid=true (explorative).
    'shift_loss_targets': 'masked_and_uniform',
    # Fast-dLLM paired m/~m views; N/A for uniform replacement FP.
    'complementary_masks': 'masked_only',
    # Fast-dLLM p_mask=(1-eps)t+eps; ELBO uses α_eff=1-p (see _elbo_schedule_weights).
    'mask_schedule': 'masked_only',
    # Fast-dLLM unweighted CE on mask sites; denom MUST be mask-count
    # (Hub ForCausalLMLoss ignore_index), never valid_tokens.sum().
    'loss_weighting': 'masked_only',
    # SUBS log-probs / mask-site NLL — absorbing parameterization.
    'subs_log_probs': 'masked_only',
    # Applied in nll() for both corruptions after the per-corruption loss.
    'ignore_bos': 'shared',
    # valid_tokens trim to T-1 when shift shortens masked *or* uniform loss.
    'valid_tokens_shift_trim': 'when_shift',
    # BlockGen pure-noise / CE-at-size specials apply to both arms.
    'pure_noise_block_sizes': 'shared',
    'pure_noise_mode': 'shared',
    'x0_causal': 'shared',
    'loss_type_special_cases': 'shared',
    'block_weights': 'shared',
    # NLD joint AR — corruption-agnostic conversion recipe (axis D).
    'joint_ar_alpha': 'shared',
    'causal_clean_stream': 'shared',
    # Hybrid FP mix rate — hybrid arm only.
    'hybrid_p_uniform': 'hybrid_only',
    # Unifusion-style intra-block attention anneal (default off).
    'intra_block_attn_anneal_steps': 'shared',
    # Optional hybrid_p ramp (mask→uniform curriculum on hybrid arm).
    'kernel_anneal_steps': 'hybrid_only',
}


class BlockTrainer(TrainerBase):
  """Single Lightning trainer for Qwen block diffusion (masked | uniform)."""

  def __init__(self, config, tokenizer):
    self.token_ids: SpecialTokenIds = ensure_special_tokens(tokenizer)
    self.mask_id, vocab_size = self.token_ids.mask_id, self.token_ids.vocab_size
    # Keep ensure_mask_token path warm (same values) for callers/tests.
    mid, vs = ensure_mask_token(tokenizer)
    assert_same_mask_id(self.mask_id, mid, where='ensure_mask_token')
    if vs != vocab_size:
      raise AssertionError(
          f'L0 vocab_size mismatch: special_tokens={vocab_size} '
          f'ensure_mask_token={vs}')
    # Absorbing (masked) needs SUBS. Uniform / hybrid keep the yaml value
    # (BlockGen uniform = ``mean``). Never silently rewrite uniform→subs.
    omegaconf.OmegaConf.set_struct(config.algo, False)
    fp_name = str(
        getattr(config.algo, 'forward_process_name', 'masked') or 'masked'
    ).lower()
    if fp_name == 'masked':
      config.algo.parameterization = 'subs'
    elif getattr(config.algo, 'parameterization', None) in (None, '', 'null'):
      config.algo.parameterization = 'mean' if fp_name == 'uniform' else 'subs'
    omegaconf.OmegaConf.set_struct(config.algo, True)
    # Label only on the block path: loss + BlockSampler always treat logits as
    # an x0 predictor → Duo/USDM posterior. ``mean`` vs ``subs`` does not
    # change ancestral reverse (base TrainerBase ``mean`` path is unused).
    _param = str(getattr(config.algo, 'parameterization', '') or '')
    if fp_name in ('uniform', 'hybrid') and _param not in ('mean', 'subs', ''):
      raise ValueError(
          f'algo.parameterization={_param!r} unsupported for {fp_name}')
    if fp_name in ('uniform', 'hybrid') and _param == 'subs':
      import logging as _logging
      _logging.getLogger(__name__).warning(
          'algo.parameterization=subs on %s is a LABEL only — block path '
          'always uses Duo/USDM uniform ELBO / x0→posterior (not SUBS). '
          'Prefer parameterization=mean for clarity.',
          fp_name)

    super().__init__(config, tokenizer, vocab_size=vocab_size)

    self.block_size = int(getattr(config, 'block_size', config.model.length))
    self.num_tokens = int(config.model.length)
    self.forward_process_name = getattr(
        config.algo, 'forward_process_name', 'masked')

    if not isinstance(self.noise, LogLinear):
      raise ValueError('BlockTrainer requires LogLinear noise schedule')

    self.shift_loss_targets = bool(
        getattr(config.algo, 'shift_loss_targets', False))
    self.complementary_masks = bool(
        getattr(config.algo, 'complementary_masks', False))
    # Hub modeling.py cats m/~m on the batch dim (fused 2B). ``sequential``
    # is a memory fallback that runs two B forwards; grads match fused.
    _cb = str(
        getattr(config.algo, 'complementary_batching', 'fused') or 'fused'
    ).lower()
    if _cb not in ('fused', 'sequential'):
      raise ValueError(
          f'algo.complementary_batching must be fused|sequential, got {_cb!r}')
    self.complementary_batching = _cb
    self.mask_schedule = str(
        getattr(config.algo, 'mask_schedule', 'alpha') or 'alpha')
    self.loss_weighting = str(
        getattr(config.algo, 'loss_weighting', 'elbo') or 'elbo')
    # Hub train: gen_mask overwrites batch attention (pads stay visible).
    # When True, do not pass padding into SDPA — structural block-diff only.
    self.hub_struct_attn_only = bool(
        getattr(config.algo, 'hub_struct_attn_only', False))
    # Match Hub/BlockGen *decode* packing at train time: forward xt only under
    # eval_block_diff_mask (no concat(xt,x0)). Default off = Fast-dLLM dual
    # stream. Prefer on for uniform when decode uses single_stream_decode.
    self.single_stream_train = bool(
        getattr(config.algo, 'single_stream_train', False))
    # conversion = Unif(V\E) Instruct-safe; blockgen = literal Unif(V) ablation.
    from ..forward_process.utils import normalize_uniform_simplex_mode
    self.uniform_simplex_mode = normalize_uniform_simplex_mode(
        getattr(config.algo, 'uniform_simplex_mode', 'conversion'))
    mixture = getattr(config.algo, 'block_size_mixture', None) or []
    self.block_size_mixture = [int(x) for x in mixture]
    stratified = getattr(config.algo, 'stratified_gamma', None)
    self.stratified_gamma = (
        float(stratified) if stratified is not None else None)

    self.block_weights = parse_block_weights(
        getattr(config.algo, 'block_weights', None))
    self.block_size_per_gpu = getattr(
        config.algo, 'block_size_per_gpu', None)
    if self.block_size_per_gpu in (None, '', 'null'):
      self.block_size_per_gpu = None
    pure = getattr(config.algo, 'pure_noise_block_sizes', None) or []
    self.pure_noise_block_sizes = {int(x) for x in pure}
    _pnm = str(getattr(config.algo, 'pure_noise_mode', 'soft') or 'soft').lower()
    if _pnm not in ('soft', 'hard'):
      raise ValueError(
          f'algo.pure_noise_mode must be soft|hard, got {_pnm!r}')
    self.pure_noise_mode = _pnm
    # Fast-dLLM default: block-causal clean stream. BlockGen multi-size: True.
    self.x0_causal = bool(getattr(config.algo, 'x0_causal', False))
    # Thread into Qwen dual-stream train mask (Fast-dLLM default False).
    bb = getattr(self, 'backbone', None)
    if bb is not None and not isinstance(bb, type):
      try:
        bb.x0_causal = self.x0_causal
      except (AttributeError, TypeError):
        # Unit tests may stub backbone as object() (no __dict__).
        pass
    self.loss_per_block_size = self._parse_loss_special_cases(
        getattr(config.algo, 'loss_type_special_cases', None))

    self.joint_ar_alpha = float(getattr(config.algo, 'joint_ar_alpha', 0.0) or 0.0)
    self.causal_clean_stream = bool(
        getattr(config.algo, 'causal_clean_stream', False))
    if (self.single_stream_train and self.joint_ar_alpha > 0
        and not self.causal_clean_stream):
      raise ValueError(
          'algo.single_stream_train=true cannot provide dual clean-stream '
          'logits for joint_ar; set causal_clean_stream=true or joint_ar_alpha=0')
    self.hybrid_p_uniform = float(
        getattr(config.algo, 'hybrid_p_uniform', 0.1) or 0.0)
    self.hybrid_decode = str(
        getattr(config.algo, 'hybrid_decode', 'masked') or 'masked')
    # Target hybrid_p after kernel anneal (defaults to configured hybrid_p).
    self._hybrid_p_uniform_target = self.hybrid_p_uniform
    # Unifusion-style: 0 = disabled (always full bi within block).
    self.intra_block_attn_anneal_steps = int(
        getattr(config.algo, 'intra_block_attn_anneal_steps', 0) or 0)
    # Hybrid-only: ramp p_uniform 0 → target over these steps (0 = off).
    self.kernel_anneal_steps = int(
        getattr(config.algo, 'kernel_anneal_steps', 0) or 0)
    if self.kernel_anneal_steps > 0:
      # Start at pure mask; ramp toward target during training_step.
      self.hybrid_p_uniform = 0.0
    # Explicit opt-in: allow shift_loss_targets on hybrid (explorative).
    self.shift_on_hybrid = bool(
        getattr(config.algo, 'shift_on_hybrid', False))
    # Clean-stream logits from the last dual forward (joint AR, non-causal).
    self._pending_clean_logits: torch.Tensor | None = None
    # plain_ce: Hub ForCausalLMLoss averages over labels != -100 (mask sites
    # only). Buffer mask indicators across fused/sequential complementary
    # forwards so _loss can use the same denom (not valid_tokens.sum()).
    self._plain_ce_mask_buf: list[torch.Tensor] | None = None
    self._plain_ce_token_count: torch.Tensor | None = None

    self._block_size_generator: torch.Generator | None = None
    self._u_rv_state: dict = {'u_rv': None}

    self._init_forward_process()
    self._validate_configuration()
    self._logged_init_metrics = False

  @staticmethod
  def _parse_loss_special_cases(raw) -> dict[int, str]:
    """BlockGen ``loss_type_special_cases``: flat [size, type, ...] or mapping."""
    if raw is None:
      return {}
    if isinstance(raw, (omegaconf.DictConfig, omegaconf.ListConfig)):
      raw = omegaconf.OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, dict):
      out = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
      if len(raw) % 2 != 0:
        raise ValueError(
            'algo.loss_type_special_cases list must be '
            '[size, type, size, type, ...]')
      out = {int(raw[i]): str(raw[i + 1]) for i in range(0, len(raw), 2)}
    else:
      raise TypeError(
          f'algo.loss_type_special_cases must be null|list|dict, got {type(raw)}')
    allowed = {'elbo', 'ce', 'ce-noisy'}
    for k, v in out.items():
      if v not in allowed:
        raise ValueError(
            f'loss_type_special_cases[{k}]={v!r} not in {allowed}')
    return out

  def backbone_logits(self, xt: torch.Tensor, x0: torch.Tensor,
                      *, block_size: int | None = None,
                      active_len: int | None = None) -> torch.Tensor:
    return self._backbone_logits(
        xt, x0, block_size=block_size or self.block_size,
        active_len=active_len)

  def prior_sample(self, *batch_dims):
    size = (
        batch_dims[0]
        if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list))
        else batch_dims)
    if self.forward_process_name == 'uniform':
      # conversion: Unif(V\E) + ban EOS from prior. blockgen: literal Unif(V).
      from ..forward_process.utils import (
          resolve_uniform_noise_redraw_exclude_ids,
          sample_uniform_excluding_mask,
      )
      exclude = resolve_uniform_noise_redraw_exclude_ids(
          self.tokenizer, mask_id=self.mask_id, vocab_size=self.vocab_size,
          mode=self.uniform_simplex_mode)
      mid = None if self.uniform_simplex_mode == 'blockgen' else self.mask_id
      return sample_uniform_excluding_mask(
          tuple(size),
          vocab_size=self.vocab_size,
          mask_id=mid,
          device=self.device,
          dtype=torch.int64,
          exclude_ids=exclude,
      )
    # masked + hybrid: start from all MASK
    return torch.full(
        tuple(size), self.mask_id, dtype=torch.int64, device=self.device)

  def on_train_start(self) -> None:
    super().on_train_start()
    if self._logged_init_metrics:
      return
    if getattr(self, 'global_rank', 0) not in (0, None):
      return
    from ..training.init import (
        compute_ar_block_init_metrics,
        log_ar_block_init_metrics,
    )
    metrics = compute_ar_block_init_metrics(self.backbone)
    log_ar_block_init_metrics(self, metrics)
    self._logged_init_metrics = True

  def _init_forward_process(self):
    fp_cfg = getattr(self.config.algo, 'forward_process', None)
    if fp_cfg is not None and hasattr(fp_cfg, '_target_'):
      # Hydra may instantiate without hybrid_p_uniform; pass when hybrid.
      kwargs = {}
      target = str(getattr(fp_cfg, '_target_', ''))
      if 'block_hybrid' in target or self.forward_process_name == 'hybrid':
        kwargs['p_uniform'] = self.hybrid_p_uniform
      if 'block_uniform' in target or self.forward_process_name == 'uniform':
        kwargs['simplex_mode'] = self.uniform_simplex_mode
      try:
        fp = hydra.utils.instantiate(
            fp_cfg, tokenizer=self.tokenizer, schedule=self.noise, **kwargs)
      except TypeError:
        fp = hydra.utils.instantiate(
            fp_cfg, tokenizer=self.tokenizer, schedule=self.noise)
        if isinstance(fp, BlockHybridForwardProcess):
          fp.p_uniform = self.hybrid_p_uniform
        if isinstance(fp, BlockUniformForwardProcess):
          fp.simplex_mode = self.uniform_simplex_mode
          from ..forward_process.utils import resolve_uniform_exclude_ids
          fp.exclude_ids = resolve_uniform_exclude_ids(
              self.tokenizer, mask_id=fp.mask_id, vocab_size=fp.vocab_size,
              mode=self.uniform_simplex_mode)
    elif self.forward_process_name == 'uniform':
      fp = BlockUniformForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_uniform',
          simplex_mode=self.uniform_simplex_mode)
    elif self.forward_process_name == 'hybrid':
      fp = BlockHybridForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_hybrid',
          p_uniform=self.hybrid_p_uniform)
    else:
      fp = BlockMaskedForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_masked')
    self._forward_process = fp
    # L0: FP mask id must match trainer (masked / hybrid path).
    fp_mask = getattr(fp, 'mask_id', None)
    if fp_mask is not None:
      assert_same_mask_id(
          self.mask_id, int(fp_mask), where='forward_process.mask_id')

  def _active_train_sizes(self) -> set[int]:
    sizes = {self.block_size}
    sizes.update(self.block_size_mixture)
    if self.block_weights is not None:
      sizes.update(sizes_from_weights(self.block_weights))
    return sizes

  def _validate_configuration(self):
    if self.time_conditioning:
      raise ValueError('BlockTrainer expects algo.time_conditioning=False')
    if self.forward_process_name not in ('masked', 'uniform', 'hybrid'):
      raise ValueError(f'Unknown forward_process_name={self.forward_process_name}')
    if self.num_tokens % self.block_size != 0:
      raise ValueError('model.length must be divisible by block_size')
    for bs in self.block_size_mixture:
      if self.num_tokens % bs != 0:
        raise ValueError(
            f'model.length must be divisible by block_size_mixture entry {bs}')
    validate_mixture_vs_weights(self.block_size_mixture, self.block_weights)
    if self.block_weights is not None:
      for bs in sizes_from_weights(self.block_weights):
        if self.num_tokens % bs != 0:
          raise ValueError(
              f'model.length must be divisible by block_weights size {bs}')
      if self.block_size_per_gpu is None:
        self.block_size_per_gpu = 'same'
      if self.block_size_per_gpu not in ('same', 'random', 'u-stratified'):
        raise ValueError(
            f'algo.block_size_per_gpu={self.block_size_per_gpu!r} invalid')
    elif self.block_size_per_gpu is not None:
      raise ValueError(
          'algo.block_size_per_gpu requires algo.block_weights '
          '(do not confuse with algo.stratified_gamma)')

    active = self._active_train_sizes()
    if not self.pure_noise_block_sizes.issubset(active):
      raise ValueError(
          f'algo.pure_noise_block_sizes={sorted(self.pure_noise_block_sizes)} '
          f'must be ⊆ active train sizes {sorted(active)}')
    if not set(self.loss_per_block_size).issubset(active):
      raise ValueError(
          f'algo.loss_type_special_cases keys '
          f'{sorted(self.loss_per_block_size)} must be ⊆ {sorted(active)}')

    if self.mask_schedule not in ('alpha', 'fast_dllm'):
      raise ValueError(
          f'algo.mask_schedule={self.mask_schedule!r} '
          f'(expected alpha|fast_dllm)')
    if self.loss_weighting not in ('elbo', 'plain_ce'):
      raise ValueError(
          f'algo.loss_weighting={self.loss_weighting!r} '
          f'(expected elbo|plain_ce)')

    if self.joint_ar_alpha < 0:
      raise ValueError(
          f'algo.joint_ar_alpha must be >= 0, got {self.joint_ar_alpha}')
    if self.causal_clean_stream and self.joint_ar_alpha <= 0:
      raise ValueError(
          'algo.causal_clean_stream=true requires algo.joint_ar_alpha > 0')
    if not (0.0 <= self.hybrid_p_uniform <= 1.0):
      raise ValueError(
          f'algo.hybrid_p_uniform must be in [0, 1], got {self.hybrid_p_uniform}')
    if self.forward_process_name == 'hybrid' and self.hybrid_decode != 'masked':
      raise ValueError(
          'algo.hybrid_decode must be \"masked\" for B4v1 '
          f'(DESIGN_LOCKS); got {self.hybrid_decode!r}')
    if self.intra_block_attn_anneal_steps < 0:
      raise ValueError(
          'algo.intra_block_attn_anneal_steps must be >= 0, '
          f'got {self.intra_block_attn_anneal_steps}')
    if self.kernel_anneal_steps < 0:
      raise ValueError(
          f'algo.kernel_anneal_steps must be >= 0, got {self.kernel_anneal_steps}')
    if self.kernel_anneal_steps > 0 and self.forward_process_name != 'hybrid':
      raise ValueError(
          'algo.kernel_anneal_steps > 0 requires forward_process_name=hybrid '
          '(mask→uniform p ramp); refuse on '
          f'{self.forward_process_name}')
    if self.shift_on_hybrid and self.forward_process_name != 'hybrid':
      raise ValueError(
          'algo.shift_on_hybrid=true is only meaningful on the hybrid arm')

    # Uniform / hybrid must not silently inherit masked-only hooks.
    if self.forward_process_name in ('uniform', 'hybrid'):
      shift_pol = LOSS_SPECIAL_CASE_POLICY.get('shift_loss_targets')
      if self.shift_loss_targets:
        # Unifusion-style port: shift allowed on uniform; hybrid needs opt-in.
        if self.forward_process_name == 'hybrid':
          if not self.shift_on_hybrid:
            raise ValueError(
                'algo.shift_loss_targets=true on hybrid requires '
                'algo.shift_on_hybrid=true (explorative; default refuse)')
        elif shift_pol == 'masked_only':
          raise ValueError(
              'algo.shift_loss_targets=true is not allowed on '
              f'{self.forward_process_name} '
              f'(LOSS_SPECIAL_CASE_POLICY={shift_pol!r})')
      if (self.complementary_masks
          and LOSS_SPECIAL_CASE_POLICY.get('complementary_masks')
          == 'masked_only'):
        raise ValueError(
            'algo.complementary_masks=true is masked-only '
            f'(LOSS_SPECIAL_CASE_POLICY); refuse on {self.forward_process_name}')
      if (self.loss_weighting == 'plain_ce'
          and LOSS_SPECIAL_CASE_POLICY.get('loss_weighting', 'masked_only')
          == 'masked_only'):
        raise ValueError(
            'algo.loss_weighting=plain_ce is masked-only '
            f'(LOSS_SPECIAL_CASE_POLICY); refuse on {self.forward_process_name}')
      if self.mask_schedule != 'alpha':
        raise ValueError(
            f'algo.mask_schedule!=alpha is masked-only; refuse on '
            f'{self.forward_process_name}')

    # Decode ARPC sanity (train-time check so bad Slurm wiring fails early).
    # BlockGen ARPC applies to absorb (masked remask) and uniform (redraw);
    # see blockgen_{absorb,uniform}_ar_then_arpc*.sh.
    sampling = getattr(self.config, 'sampling', None)
    if sampling is not None and bool(getattr(sampling, 'use_arpc', False)):
      if self.forward_process_name not in ('uniform', 'masked', 'hybrid'):
        raise ValueError(
            'sampling.use_arpc=true requires forward_process_name in '
            f'{{uniform, masked, hybrid}}; got {self.forward_process_name!r}')
      if 1 not in active:
        # Decode-only ARPC on fixed-size ckpts is weak but allowed (sampler
        # warns). Hard-fail only blocked mistaken eval submits; size-1 gate
        # now lives in submit_family_eval (EVAL_HAS_ARPC_SIZE1).
        import logging as _logging
        _logging.getLogger(__name__).warning(
            'sampling.use_arpc=true but block size 1 is not in '
            'algo.block_size_mixture / block_weights (BlockGen ARPC '
            'prerequisite). AR verify (L\'=1) is untrained on this ckpt.')
      mode = str(getattr(sampling, 'arpc_mode', 'simplified') or 'simplified')
      if mode not in ('simplified', 'blockgen'):
        raise ValueError(
            f'sampling.arpc_mode={mode!r} not in (simplified|blockgen)')

  def _sample_training_block_size(
      self, current_accumulation_step: int | None = None) -> int:
    if self.block_weights is not None:
      world_size = 1
      global_rank = 0
      accum = 1
      seed = int(getattr(self.config, 'seed', 0) or 0)
      if getattr(self, '_trainer', None) is not None:
        world_size = int(getattr(self.trainer, 'world_size', 1) or 1)
        global_rank = int(getattr(self.trainer, 'global_rank', 0) or 0)
        accum = int(getattr(self.trainer, 'accumulate_grad_batches', 1) or 1)
      log_bs, self._block_size_generator = sample_log_block_size(
          self.block_weights,
          mode=self.block_size_per_gpu or 'same',
          world_size=world_size,
          global_rank=global_rank,
          accumulate_grad_batches=accum,
          current_accumulation_step=current_accumulation_step,
          seed=seed,
          generator=self._block_size_generator,
          u_rv_state=self._u_rv_state,
      )
      return 2 ** log_bs
    if not self.block_size_mixture:
      return self.block_size
    idx = int(torch.randint(
        0, len(self.block_size_mixture), (1,), device=self.device).item())
    return self.block_size_mixture[idx]

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def _corrupt(self, x0: torch.Tensor, t: torch.Tensor,
               *, block_size: int,
               return_move_mask: bool = False,
               corruption_mask: torch.Tensor | None = None
               ) -> torch.Tensor | tuple:
    if isinstance(self._forward_process, BlockMaskedForwardProcess):
      out = self._forward_process(
          x0, t, block_size=block_size,
          complementary=False,
          return_move_mask=return_move_mask,
          mask_schedule=self.mask_schedule)
      if return_move_mask:
        xt, _, move_mask = out
      else:
        xt, _ = out
        move_mask = None
    else:
      # Uniform / hybrid: request Bernoulli move mask when available so
      # probes can separate keeps from 1/V_eff collisions.
      fp = self._forward_process
      if return_move_mask:
        try:
          out = fp(
              x0, t, block_size=block_size, return_move_mask=True)
          if isinstance(out, tuple) and len(out) == 2:
            xt, move_mask = out
          else:
            xt, move_mask = out, None
        except TypeError:
          xt = fp(x0, t, block_size=block_size)
          move_mask = None
      else:
        xt = fp(x0, t, block_size=block_size)
        move_mask = None
    if corruption_mask is not None:
      supervised = corruption_mask.bool()
      xt = torch.where(supervised, xt, x0)
      if move_mask is not None:
        move_mask = move_mask & supervised
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
      if move_mask is not None:
        move_mask = move_mask.clone()
        move_mask[:, 0] = False
    if return_move_mask:
      return xt, move_mask
    return xt

  def _use_pure_noise(self, train_mode: bool, block_size: int) -> bool:
    """BlockGen: train at listed sizes, or always size-1 at eval."""
    if (not train_mode) and int(block_size) == 1:
      return True
    return int(block_size) in self.pure_noise_block_sizes

  def _hard_pure_noise_xt(
      self,
      x0: torch.Tensor,
      *,
      corruption_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """BlockGen ``q_xt(..., use_pure_noise=True)``: hard prior fill.

    Masked → all MASK on supervised sites. Uniform/hybrid → same simplex as
    ``prior_sample`` / ARPC redraw (EOS banned under conversion mode).
    """
    if isinstance(self._forward_process, BlockMaskedForwardProcess):
      noise = torch.full_like(x0, int(self.mask_id))
    else:
      from ..forward_process.utils import (
          resolve_uniform_noise_redraw_exclude_ids,
      )
      vocab = int(getattr(self, 'vocab_size', self.vocab_size))
      mode = str(getattr(self, 'uniform_simplex_mode', 'conversion'))
      mid = None if mode == 'blockgen' else self.mask_id
      exclude = resolve_uniform_noise_redraw_exclude_ids(
          self.tokenizer, mask_id=self.mask_id, vocab_size=vocab, mode=mode)
      noise = sample_uniform_excluding_mask(
          x0.shape,
          vocab_size=vocab,
          mask_id=mid,
          device=x0.device,
          dtype=x0.dtype,
          exclude_ids=exclude,
      )
    if corruption_mask is not None:
      xt = torch.where(corruption_mask.bool(), noise, x0)
    else:
      xt = noise
    if self.ignore_bos:
      xt = xt.clone()
      xt[:, 0] = x0[:, 0]
    return xt

  def _backbone_logits(
      self, xt: torch.Tensor, x0: torch.Tensor,
      *, block_size: int | None = None,
      return_clean: bool = False,
      active_len: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if active_len is not None:
      xt = xt[:, :active_len]
      x0 = x0[:, :active_len]
      if attention_mask is not None:
        attention_mask = attention_mask[:, :active_len]
    if self.single_stream_train:
      if return_clean:
        raise RuntimeError(
            'single_stream_train has no clean half; use causal_clean_stream '
            'for joint AR')
      train_fn = getattr(self.backbone, 'block_train_logits', None)
      if train_fn is None:
        raise RuntimeError(
            'algo.single_stream_train requires backbone.block_train_logits')
      # Full seq as active window (train NLL is over all supervised sites).
      a = int(xt.shape[1])
      return train_fn(
          xt, active_len=a, block_size=block_size,
          attention_mask=attention_mask)
    x_in = torch.cat([xt, x0], dim=-1)
    return self.backbone(
        x_in, sigma=None, block_size=block_size, return_both=return_clean,
        active_len=active_len, attention_mask=attention_mask)

  def _ar_ce_from_logits(
      self, logits: torch.Tensor, x0: torch.Tensor,
      valid_tokens: torch.Tensor,
  ) -> torch.Tensor:
    """Token-mean next-token CE (ArSft-style); returns scalar."""
    shift_logits = logits[:, :-1, :]
    shift_labels = x0[:, 1:]
    shift_valid = valid_tokens[:, 1:].to(shift_logits.dtype)
    if self.ignore_bos:
      shift_valid = shift_valid.clone()
      shift_valid[:, 0] = 0
    nll = -shift_logits.log_softmax(-1).gather(
        -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    denom = shift_valid.sum().clamp(min=1)
    return (nll * shift_valid).sum() / denom

  def _ar_ce_mean(self, x0: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
    """AR term: causal HF or pending clean-stream logits."""
    if self.causal_clean_stream:
      causal_fn = getattr(self.backbone, 'causal_train_logits', None)
      if causal_fn is None:
        raise RuntimeError(
            'causal_clean_stream requires backbone.causal_train_logits')
      logits = causal_fn(x0)
      return self._ar_ce_from_logits(logits, x0, valid_tokens)
    if self._pending_clean_logits is None:
      raise RuntimeError(
          'joint_ar_alpha>0 without causal_clean_stream requires clean-stream '
          'logits from the dual block forward')
    # Complementary doubles batch on diff; AR uses first view's clean half only.
    clean = self._pending_clean_logits
    if clean.size(0) != x0.size(0):
      clean = clean[: x0.size(0)]
    return self._ar_ce_from_logits(clean, x0, valid_tokens)

  def _elbo_schedule_weights(self, t: torch.Tensor):
    """Return ``(alpha_t, dalpha_t)`` consistent with the forward mask rate.

    ``mask_schedule=alpha`` (default): LogLinear ``α=1-(1-ε)t``.
    ``mask_schedule=fast_dllm``: FP uses ``p=(1-ε)t+ε``; ELBO weights must use
    ``α_eff=1-p`` (same ``α'=-(1-ε)``) so training matches the corruption rate.
    """
    if (self.mask_schedule == 'fast_dllm'
        and self.forward_process_name == 'masked'):
      eps = float(self.noise.eps)
      p_mask = (1.0 - eps) * t + eps
      alpha_t = (1.0 - p_mask).to(dtype=torch.float32)
      dalpha_t = -(1.0 - eps) * torch.ones_like(t, dtype=torch.float32)
      return alpha_t, dalpha_t
    return self.noise.alpha_t(t), self.noise.alpha_prime_t(t)

  def _record_plain_ce_mask(self, mask_positions: torch.Tensor) -> None:
    """Accumulate mask-site indicators for Hub-faithful plain_ce averaging."""
    buf = self._plain_ce_mask_buf
    if buf is not None:
      buf.append(mask_positions.detach())

  def _masked_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    # Shift must mirror Diffusion.nll (base.py): raw next-token CE on
    # mask positions only. Applying SUBS *then* shifting scores x0[i+1]
    # under a one-hot on xt[i] for unmasked i → ~neg_infinity NLL and
    # trainer/loss ~1e6 (Track 2 jobs 138068 / 138098).
    weight = getattr(self, 'loss_weighting', 'elbo')
    if self.shift_loss_targets:
      logits = logits[:, :-1]
      x0 = x0[:, 1:]
      xt = xt[:, 1:]
      alpha_t = alpha_t[:, 1:]
      dalpha_t = dalpha_t[:, 1:]
      ce = -logits.log_softmax(-1).gather(
          -1, x0.unsqueeze(-1)).squeeze(-1)
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      # plain_ce must return *positive* CE (minimize-oriented), matching
      # masked_plain_ce_per_token below. Returning -ce here inverted the
      # Fast-dLLM objective (C2 job 1660576: trainer/loss -6 → -730, logit
      # explosion on a single attractor token). ELBO still uses -ce so that
      # (dalpha/(1-alpha)) * (-ce) stays minimize-oriented (dalpha < 0).
      if weight == 'plain_ce':
        self._record_plain_ce_mask(mask_positions)
        return mask_positions * ce
      masked_neg_ce = mask_positions * (-ce)
      weighting = dalpha_t / (1.0 - alpha_t)
      return weighting * masked_neg_ce
    if weight == 'plain_ce':
      mask_positions = (xt == self.mask_id).to(
          dtype=logits.dtype if logits.is_floating_point() else torch.float32)
      self._record_plain_ce_mask(mask_positions)
      return masked_plain_ce_per_token(logits, x0, xt, self.mask_id)
    log_probs = subs_log_probs(logits, xt, self.mask_id, self.neg_infinity)
    return masked_block_nll_per_token(log_probs, x0, alpha_t, dalpha_t)

  def _apply_ignore_bos_mask(
      self,
      loss: torch.Tensor,
      valid_tokens: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero BOS in the valid mask; only zero loss[:, 0] when loss still has BOS."""
    if not self.ignore_bos:
      return loss, valid_tokens
    valid_tokens = valid_tokens.clone()
    valid_tokens[:, 0] = 0
    # shift_loss_targets drops BOS from the loss grid; index 0 is the first real token.
    if not self.shift_loss_targets:
      loss = loss.clone()
      loss[:, 0] = 0
    return loss, valid_tokens

  def _uniform_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    """DUO/UDLM closed-form uniform NLL (explicit ``x0==xt`` handled inside).

    Intentional non-applications vs ``_masked_loss`` (see
    ``LOSS_SPECIAL_CASE_POLICY``):
    - ``subs_log_probs`` / mask-site CE: absorbing parameterization only.
    - ``complementary_masks``: applied in ``nll`` for masked FP only.

    ``shift_loss_targets`` (Unifusion / Fast-dLLM AR alignment): when on,
    logits at position i score clean token i+1 — same index shift as masked.

    MASK contract (Qwen conversion): BlockGen/Duo have no MASK on the uniform
    arm. We always keep MASK for the shared masked arm, so the predictive
    simplex is ``V \\ {MASK}`` — ban MASK logits and pass ``V_eff = V-1``
    (same as ``_hybrid_loss`` uniform branch).
    """
    from ..forward_process.utils import (
        resolve_uniform_exclude_ids,
        uniform_simplex_size,
    )
    if self.shift_loss_targets:
      logits = logits[:, :-1]
      x0 = x0[:, 1:]
      xt = xt[:, 1:]
      alpha_t = alpha_t[:, 1:]
      dalpha_t = dalpha_t[:, 1:]
    exclude = resolve_uniform_exclude_ids(
        self.tokenizer, mask_id=self.mask_id, vocab_size=self.vocab_size,
        mode=self.uniform_simplex_mode)
    logits_u = logits
    if exclude:
      logits_u = logits.clone()
      for mid in exclude:
        if 0 <= int(mid) < int(self.vocab_size):
          logits_u[..., int(mid)] = self.neg_infinity
    log_probs = F.log_softmax(logits_u, dim=-1)
    # Explicit exclude set (possibly empty = full V for blockgen mode).
    v_eff = uniform_simplex_size(
        self.vocab_size, None, exclude_ids=exclude)
    return uniform_block_nll_per_token(
        log_probs, xt, x0, alpha_t, dalpha_t, v_eff,
        mask_id=self.mask_id if exclude else None, exclude_ids=exclude)

  def _hybrid_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    """Hybrid ELBO surrogate matching ``BlockHybridForwardProcess``.

    FP replaces with MASK or Unif(V\\{MASK}). We score:
    - MASK sites with the absorbing (masked) ELBO
    - Unif-replaced sites with DUO/uniform ELBO using ``V-1`` (MASK
      excluded from the predictive simplex, matching the FP)
    - Clean sites contribute 0 (noise-site scoring, same spirit as SUBS)

    This is not a full GIDD transition ELBO; it is the block-path hybrid
    used for B4.

    ``shift_loss_targets`` (explorative via ``shift_on_hybrid``): both arms
    share one T−1 grid — logits[i] scores clean token i+1 — before site
    routing. Do not call ``_masked_loss`` with shift still on (it would
    shorten only the masked arm and crash on the uniform term).
    """
    from ..forward_process.utils import (
        resolve_uniform_exclude_ids,
        uniform_simplex_size,
    )
    if self.shift_loss_targets:
      # Shared Unifusion-style shift (same index surgery as _masked_loss /
      # _uniform_loss). Masked arm uses CE path under shift (SUBS+shift is
      # illegal — see _masked_loss). Uniform arm uses Duo ELBO on the same
      # shortened tensors.
      logits = logits[:, :-1]
      x0 = x0[:, 1:]
      xt = xt[:, 1:]
      alpha_t = alpha_t[:, 1:]
      dalpha_t = dalpha_t[:, 1:]
      weight = getattr(self, 'loss_weighting', 'elbo')
      ce = -logits.log_softmax(-1).gather(
          -1, x0.unsqueeze(-1)).squeeze(-1)
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      if weight == 'plain_ce':
        self._record_plain_ce_mask(mask_positions)
        masked_term = mask_positions * ce
      else:
        masked_term = (dalpha_t / (1.0 - alpha_t)) * (
            mask_positions * (-ce))
    else:
      masked_term = self._masked_loss(logits, xt, x0, alpha_t, dalpha_t)
    # Exclude MASK from the uniform predictive simplex (FP never draws MASK
    # on the uniform branch). Tensors are already shift-aligned above.
    logits_u = logits.clone()
    exclude = resolve_uniform_exclude_ids(
        self.tokenizer, mask_id=self.mask_id, vocab_size=self.vocab_size)
    for mid in exclude:
      if 0 <= int(mid) < int(self.vocab_size):
        logits_u[..., int(mid)] = self.neg_infinity
    log_probs_u = F.log_softmax(logits_u, dim=-1)
    uniform_term = uniform_block_nll_per_token(
        log_probs_u, xt, x0, alpha_t, dalpha_t,
        uniform_simplex_size(
            self.vocab_size, self.mask_id, exclude_ids=exclude),
        mask_id=self.mask_id, exclude_ids=exclude)
    is_mask = (xt == self.mask_id)
    is_unif = (xt != x0) & (~is_mask)
    if masked_term.shape != uniform_term.shape:
      raise RuntimeError(
          'hybrid loss shape mismatch after shift alignment: '
          f'masked={tuple(masked_term.shape)} '
          f'uniform={tuple(uniform_term.shape)}')
    return (
        is_mask.to(masked_term.dtype) * masked_term
        + is_unif.to(uniform_term.dtype) * uniform_term)

  def _ce_loss(self, logits, xt, x0, *, noisy_only: bool) -> torch.Tensor:
    """Unweighted token CE (BlockGen size-1 / special-case path)."""
    ce = -logits.log_softmax(-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    if noisy_only:
      if self.forward_process_name == 'masked':
        keep = (xt == self.mask_id).to(ce.dtype)
      elif self.forward_process_name == 'hybrid':
        keep = ((xt == self.mask_id) | (xt != x0)).to(ce.dtype)
      else:
        keep = (xt != x0).to(ce.dtype)
      ce = ce * keep
    return ce

  def _loss_for_block(
      self, logits, xt, x0, alpha_t, dalpha_t, *, block_size: int,
      train_mode: bool = True,
  ) -> torch.Tensor:
    # BlockGen eval rule (third_party/blockgen/algo.py ``_get_loss_type``):
    # size-1 at eval → CE (true next-token / AR NLL under the size-1 mask),
    # never the continuous-time ELBO. Training still honours
    # ``loss_type_special_cases`` (e.g. ce_at_1) and default ELBO.
    if (not train_mode) and block_size == 1:
      return self._ce_loss(logits, xt, x0, noisy_only=False)
    lt = self.loss_per_block_size.get(block_size, self.loss_type)
    if lt == 'ce':
      return self._ce_loss(logits, xt, x0, noisy_only=False)
    if lt == 'ce-noisy':
      return self._ce_loss(logits, xt, x0, noisy_only=True)
    # elbo
    if self.forward_process_name == 'uniform':
      return self._uniform_loss(logits, xt, x0, alpha_t, dalpha_t)
    if self.forward_process_name == 'hybrid':
      return self._hybrid_loss(logits, xt, x0, alpha_t, dalpha_t)
    return self._masked_loss(logits, xt, x0, alpha_t, dalpha_t)

  def nll(self, x0, valid_tokens, current_accumulation_step=None,
          train_mode=False, block_size: int | None = None,
          attention_mask: torch.Tensor | None = None):
    self._pending_clean_logits = None
    self._plain_ce_token_count = None
    self._plain_ce_mask_buf = (
        [] if self.loss_weighting == 'plain_ce' else None)
    bsz = x0.shape[0]
    bs = (
        block_size if block_size is not None
        else self._sample_training_block_size(current_accumulation_step))
    # Hub modeling.py: t ~ U(0,1), then p=(1-ε)t+ε. Our sample_block_timesteps
    # normally floors t to [sampling_eps,1]. Applying that *and* fast_dllm's
    # p-map double-counts ε (min p≈2ε). Use raw U(0,1) t only for fast_dllm.
    # Hub also uses i.i.d. block times (no antithetic); keep that for fast_dllm.
    t_eps = (
        0.0 if self.mask_schedule == 'fast_dllm' else self.sampling_eps)
    use_antithetic = (
        False if self.mask_schedule == 'fast_dllm'
        else self.antithetic_sampling)
    t = sample_block_timesteps(
        bsz, self.num_tokens, bs, self.device,
        sampling_eps=t_eps,
        antithetic=use_antithetic,
        stratified_gamma=self.stratified_gamma)

    # Pure-noise schedule: always force t=1 when the size qualifies.
    # soft (Fast-dLLM default): Bernoulli at α(1)≈ε.
    # hard (BlockGen opt-in): hard-fill MASK / Unif — skip complementary.
    use_pure = self._use_pure_noise(train_mode, bs)
    if use_pure:
      t = torch.ones_like(t)

    alpha_t, dalpha_t = self._elbo_schedule_weights(t)
    want_clean = (
        self.joint_ar_alpha > 0 and not self.causal_clean_stream)
    # valid_tokens = supervised sites (assistant labels != -100).
    # Hub modeling.py training overwrites attention with structural gen_mask
    # only — prompt *and* pad positions stay visible; labels=-100 drop CE.
    # Passing assistant-only masks as attention blocked the prompt (fixed).
    # hub_struct_attn_only=True goes further and matches Hub (no pad mask).
    supervised = valid_tokens
    if getattr(self, 'hub_struct_attn_only', False):
      attn = None
    else:
      attn = attention_mask if attention_mask is not None else valid_tokens

    hard_pure = use_pure and self.pure_noise_mode == 'hard'
    if hard_pure:
      xt = self._hard_pure_noise_xt(x0, corruption_mask=supervised)
      if want_clean:
        logits, clean = self._backbone_logits(
            xt, x0, block_size=bs, return_clean=True,
            attention_mask=attn)
        self._pending_clean_logits = clean
      else:
        logits = self._backbone_logits(
            xt, x0, block_size=bs, attention_mask=attn)
      loss = self._loss_for_block(
          logits, xt, x0, alpha_t, dalpha_t, block_size=bs,
          train_mode=train_mode)
    elif (self.complementary_masks
        and isinstance(self._forward_process, BlockMaskedForwardProcess)):
      # Fast-dLLM Hub: paired m/~m, then ``torch.cat(..., dim=0)`` → fused 2B.
      _, move_mask = self._corrupt(
          x0, t, block_size=bs, return_move_mask=True,
          corruption_mask=supervised)
      xt_a, xt_b = complementary_pair_from_mask(x0, move_mask, self.mask_id)
      if self.ignore_bos:
        xt_a[:, 0] = x0[:, 0]
        xt_b[:, 0] = x0[:, 0]
      if supervised is not None:
        keep = supervised.bool()
        xt_a = torch.where(keep, xt_a, x0)
        xt_b = torch.where(keep, xt_b, x0)
      if self.complementary_batching == 'sequential':
        # Memory fallback: two B forwards; same summed NLL as fused.
        if want_clean:
          out_a = self._backbone_logits(
              xt_a, x0, block_size=bs, return_clean=True,
              attention_mask=attn)
          logits_a, clean_a = out_a
          self._pending_clean_logits = clean_a
        else:
          logits_a = self._backbone_logits(
              xt_a, x0, block_size=bs, attention_mask=attn)
        loss_a = self._loss_for_block(
            logits_a, xt_a, x0, alpha_t, dalpha_t, block_size=bs,
            train_mode=train_mode)
        logits_b = self._backbone_logits(
            xt_b, x0, block_size=bs, attention_mask=attn)
        loss_b = self._loss_for_block(
            logits_b, xt_b, x0, alpha_t, dalpha_t, block_size=bs,
            train_mode=train_mode)
        loss = torch.cat([loss_a, loss_b], dim=0)
        valid_tokens = torch.cat([valid_tokens, valid_tokens], dim=0)
      else:
        xt_pair = torch.cat([xt_a, xt_b], dim=0)
        x0_pair = torch.cat([x0, x0], dim=0)
        attn_pair = (
            torch.cat([attn, attn], dim=0) if attn is not None else None)
        alpha_pair = torch.cat([alpha_t, alpha_t], dim=0)
        dalpha_pair = torch.cat([dalpha_t, dalpha_t], dim=0)
        if want_clean:
          out = self._backbone_logits(
              xt_pair, x0_pair, block_size=bs, return_clean=True,
              attention_mask=attn_pair)
          logits, clean = out
          self._pending_clean_logits = clean[: bsz]
        else:
          logits = self._backbone_logits(
              xt_pair, x0_pair, block_size=bs, attention_mask=attn_pair)
        loss = self._loss_for_block(
            logits, xt_pair, x0_pair, alpha_pair, dalpha_pair,
            block_size=bs, train_mode=train_mode)
        valid_tokens = torch.cat([valid_tokens, valid_tokens], dim=0)
    else:
      xt = self._corrupt(x0, t, block_size=bs, corruption_mask=supervised)
      if want_clean:
        logits, clean = self._backbone_logits(
            xt, x0, block_size=bs, return_clean=True,
            attention_mask=attn)
        self._pending_clean_logits = clean
      else:
        logits = self._backbone_logits(
            xt, x0, block_size=bs, attention_mask=attn)
      loss = self._loss_for_block(
          logits, xt, x0, alpha_t, dalpha_t, block_size=bs,
          train_mode=train_mode)

    # masked_block_nll_per_token / uniform_block_nll_per_token already return
    # minimize-oriented terms (MDLM-equivalent: coeff * log_p with coeff<0).
    # Do NOT negate — that inverts the ELBO and rewards worse predictions.
    # CE special-cases return positive CE (also minimize-oriented).
    loss, valid_tokens = self._apply_ignore_bos_mask(loss, valid_tokens)
    # Mirror base.py: shift_loss_targets shortens loss to T-1 in
    # _masked_loss / _uniform_loss; trim the pad mask so multiply / BPD
    # denom stay aligned.
    if (self.shift_loss_targets
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]
    if self._plain_ce_mask_buf is not None:
      # Hub modeling.py sets labels=-100 on clean sites, then
      # ForCausalLMLoss averages only over remaining (mask) labels. With
      # complementary, that is mask sites across both views — NOT 2B×T
      # valid tokens. Using valid.sum() here halves the gradient vs Hub.
      if not self._plain_ce_mask_buf:
        raise RuntimeError(
            'plain_ce requested but no mask indicators were recorded')
      pcm = (
          self._plain_ce_mask_buf[0]
          if len(self._plain_ce_mask_buf) == 1
          else torch.cat(self._plain_ce_mask_buf, dim=0))
      if pcm.shape != loss.shape:
        raise RuntimeError(
            f'plain_ce mask shape {tuple(pcm.shape)} vs loss '
            f'{tuple(loss.shape)}')
      self._plain_ce_token_count = (
          pcm * valid_tokens.to(dtype=pcm.dtype)).sum()
      self._plain_ce_mask_buf = None
    return loss * valid_tokens

  def _loss(self, x0, valid_tokens, current_accumulation_step=None,
            train_mode=False, attention_mask: torch.Tensor | None = None):
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
    attn = attention_mask
    if attn is not None:
      # Keep attn aligned with any _process_model_input length changes.
      if attn.size(-1) != input_tokens.size(-1):
        raise RuntimeError(
            f'attention_mask length {attn.size(-1)} vs input '
            f'{input_tokens.size(-1)}')
    # Keep pre-complementary / pre-shift-trim mask for the AR term (full T).
    ar_valid = valid_tokens
    nlls = self.nll(
        input_tokens, valid_tokens,
        current_accumulation_step=current_accumulation_step,
        train_mode=train_mode,
        attention_mask=attn)
    # Metrics expect (nll_sum, num_tokens) where num_tokens is the total
    # count used as the aggregation weight. Using a per-position tensor
    # here would broadcast the scalar weight across positions and mis-scale
    # val/bpd (and train/bpd as a side effect).
    # Same shift trim as nll() / base Diffusion._loss so num_tokens matches
    # the T-1 loss grid when shift_loss_targets is on.
    if (self.shift_loss_targets
        and valid_tokens.size(-1) == nlls.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]
    # Complementary doubles the batch; valid_tokens already doubled in nll,
    # but the caller-passed mask must match nlls batch for num_tokens.
    if valid_tokens.size(0) != nlls.size(0):
      if nlls.size(0) == 2 * valid_tokens.size(0):
        valid_tokens = torch.cat([valid_tokens, valid_tokens], dim=0)
      else:
        raise RuntimeError(
            f'nlls batch {nlls.size(0)} vs valid_tokens {valid_tokens.size(0)}')
    nll_sum = nlls.sum()
    # plain_ce: Hub mean over mask labels only (see nll / modeling.py).
    # ELBO keeps valid_tokens.sum() (per-position weighted terms).
    if self.loss_weighting == 'plain_ce':
      if self._plain_ce_token_count is None:
        raise RuntimeError(
            'plain_ce loss missing mask-site token count from nll()')
      num_tokens = self._plain_ce_token_count
    else:
      num_tokens = valid_tokens.sum()
    diff_nll = nll_sum / num_tokens.clamp(min=1)
    if self.joint_ar_alpha > 0:
      # AR once on clean stream (not doubled under complementary).
      ar_nll = self._ar_ce_mean(input_tokens, ar_valid)
      combined = ar_nll + self.joint_ar_alpha * diff_nll
      self._last_ar_nll = ar_nll.detach()
      self._last_diff_nll = diff_nll.detach()
      self._last_joint_nll = combined.detach()
      # nlls stay diffusion-only so epoch val/nll|bpd stay cross-cell
      # comparable; optimized objective is losses.loss / *_joint_nll.
      return Loss(loss=combined, nlls=nll_sum, num_tokens=num_tokens)
    self._last_ar_nll = None
    self._last_diff_nll = diff_nll.detach()
    self._last_joint_nll = None
    return Loss(loss=diff_nll, nlls=nll_sum, num_tokens=num_tokens)

  def _curriculum_progress(self, steps: int) -> float:
    """Linear [0,1] progress for anneal schedules (1.0 when disabled/done)."""
    if steps <= 0:
      return 1.0
    try:
      step = int(self.global_step)
    except Exception:
      step = 0
    return float(min(1.0, max(0.0, step / float(steps))))

  def _apply_unifusion_curricula(self) -> None:
    """Update intra-block attn open + hybrid_p (no-op when schedules are 0).

    Defaults keep legacy behaviour: open=1.0, hybrid_p fixed.
    """
    open_p = self._curriculum_progress(self.intra_block_attn_anneal_steps)
    backbone = getattr(self, 'backbone', None)
    if backbone is not None and hasattr(backbone, 'intra_block_attn_open'):
      backbone.intra_block_attn_open = open_p

    if (self.kernel_anneal_steps > 0
        and self.forward_process_name == 'hybrid'):
      p = (self._curriculum_progress(self.kernel_anneal_steps)
           * float(self._hybrid_p_uniform_target))
      self.hybrid_p_uniform = p
      fp = getattr(self, 'forward_process', None)
      if fp is not None and hasattr(fp, 'p_uniform'):
        fp.p_uniform = p

  def training_step(self, batch, batch_idx):
    self._apply_unifusion_curricula()
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    valid_tokens = self._batch_valid_tokens(batch)
    losses = self._loss(
        batch['input_ids'], valid_tokens,
        current_accumulation_step=current_accumulation_step, train_mode=True,
        attention_mask=batch['attention_mask'])
    self.metrics.update_train(losses.nlls, losses.num_tokens)
    # Step-level train metrics: with max_steps + huge SFT epochs,
    # on_train_epoch_end (train/nll|bpd|ppl) almost never fires.
    # sync_dist=False: avoid NCCL barriers every log_every_n_steps; rank0
    # WandB is enough for train curves.
    # train/nll|bpd|ppl = diffusion ELBO (matches metrics / val).
    # trainer/loss = optimized objective (joint when joint_ar_alpha>0).
    obj = losses.loss.detach()
    diff = (
        self._last_diff_nll if self._last_diff_nll is not None
        else obj)
    self.log('trainer/loss', obj, on_step=True, on_epoch=False,
             sync_dist=False, prog_bar=True)
    self.log('train/nll', diff, on_step=True, on_epoch=False, sync_dist=False)
    self.log('train/bpd', diff / math.log(2), on_step=True, on_epoch=False,
             sync_dist=False)
    self.log('train/ppl', torch.exp(diff), on_step=True, on_epoch=False,
             sync_dist=False)
    if getattr(self, '_last_ar_nll', None) is not None:
      self.log('train/ar_nll', self._last_ar_nll, on_step=True, on_epoch=False,
               sync_dist=False)
      self.log('train/diff_nll', self._last_diff_nll, on_step=True,
               on_epoch=False, sync_dist=False)
      self.log('train/joint_nll', self._last_joint_nll, on_step=True,
               on_epoch=False, sync_dist=False)
    return losses.loss

  def validation_step(self, batch, batch_idx):
    del batch_idx
    valid_tokens = self._batch_valid_tokens(batch)
    losses = self._loss(
        batch['input_ids'], valid_tokens,
        attention_mask=batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.num_tokens)
    if getattr(self, '_last_ar_nll', None) is not None:
      # Epoch-agg of C5 terms (val/nll from metrics stays diffusion-only).
      self.log('val/ar_nll', self._last_ar_nll, on_step=False, on_epoch=True,
               sync_dist=True)
      self.log('val/diff_nll', self._last_diff_nll, on_step=False, on_epoch=True,
               sync_dist=True)
      self.log('val/joint_nll', self._last_joint_nll, on_step=False,
               on_epoch=True, sync_dist=True)
    if bool(getattr(self.config.eval, 't_bucketed_nll', False)):
      self._log_t_bucketed_nll(
          batch['input_ids'], valid_tokens,
          attention_mask=batch['attention_mask'])
    return losses.loss

  @staticmethod
  def _batch_valid_tokens(batch):
    """Return assistant-label mask for SFT, or the usual padding mask."""
    labels = batch.get('labels')
    if labels is None:
      return batch['attention_mask']
    return (
        labels.ne(-100) & batch['attention_mask'].bool()
    ).to(batch['attention_mask'].dtype)

  @torch.no_grad()
  def _log_t_bucketed_nll(
      self, x0: torch.Tensor, valid_tokens: torch.Tensor,
      attention_mask: torch.Tensor | None = None):
    """Layer-3 diagnostic: NLL vs corruption level (AR-init cliff detector).

    Logs ``val/{nll,bpd,ppl}_alpha_{lo}_{hi}`` for fixed α bands.
    Aggregate val/nll alone can hide a high-t cliff on ar2block_uniform.
    """
    x0, valid_tokens = self._process_model_input(x0, valid_tokens)
    attn = attention_mask if attention_mask is not None else valid_tokens
    if getattr(self, 'hub_struct_attn_only', False):
      attn = None
    bsz, seq = x0.shape
    bs = self.block_size
    # α bands: high α = low corruption. Use midpoints via LogLinear inverse.
    bands = ((0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95))
    eps = float(self.noise.eps)
    # Complementary doubling would distort band curves; probe single-view.
    saved_comp = self.complementary_masks
    self.complementary_masks = False
    try:
      for lo, hi in bands:
        alpha_mid = 0.5 * (lo + hi)
        # Invert so band α matches ELBO α (LogLinear or fast_dllm α_eff).
        if (self.mask_schedule == 'fast_dllm'
            and self.forward_process_name == 'masked'):
          # α_eff = (1-ε)(1-t)  →  t = 1 - α_eff/(1-ε)
          t_val = 1.0 - alpha_mid / max(1.0 - eps, 1e-8)
          t_val = float(min(1.0, max(0.0, t_val)))
        else:
          # alpha = 1 - (1-eps)*t  →  t = (1-alpha)/(1-eps)
          t_val = (1.0 - alpha_mid) / max(1.0 - eps, 1e-8)
        t = torch.full((bsz, seq), t_val, device=self.device, dtype=torch.float32)
        # Constant within each block (match training geometry).
        n_blocks = seq // bs
        for bi in range(n_blocks):
          sl = slice(bi * bs, (bi + 1) * bs)
          t[:, sl] = t[:, bi * bs: bi * bs + 1]
        alpha_t, dalpha_t = self._elbo_schedule_weights(t)
        xt = self._corrupt(x0, t, block_size=bs, corruption_mask=valid_tokens)
        logits = self._backbone_logits(
            xt, x0, block_size=bs, attention_mask=attn)
        loss = self._loss_for_block(
            logits, xt, x0, alpha_t, dalpha_t, block_size=bs)
        vt = valid_tokens
        loss, vt = self._apply_ignore_bos_mask(loss, valid_tokens)
        # Mirror nll(): shift shortens masked loss to T-1 — trim pad mask
        # (Track 2 crash 138103: 511 vs 512 in loss * vt).
        if (self.shift_loss_targets
            and vt.size(-1) == loss.size(-1) + 1):
          vt = vt[:, 1:]
        weighted = loss * vt
        if self.loss_weighting == 'plain_ce':
          # Match train Hub-CE mean (mask sites), not all valid — otherwise
          # t-bucket curves under-scale vs trainer/loss under plain_ce.
          xt_m = xt[:, 1:] if (
              self.shift_loss_targets and xt.size(-1) == vt.size(-1) + 1
          ) else xt
          if xt_m.size(-1) != vt.size(-1):
            xt_m = xt_m[:, : vt.size(-1)]
          denom = (
              (xt_m == self.mask_id).to(weighted.dtype) * vt
          ).sum().clamp(min=1)
        else:
          denom = vt.sum().clamp(min=1)
        nll = weighted.sum() / denom
        band = f'{lo:.2f}_{hi:.2f}'.replace('.', 'p')
        # NLL plus BPD/PPL (same transforms as aggregate val/{bpd,ppl}).
        self.log(f'val/nll_alpha_{band}', nll, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log(f'val/bpd_alpha_{band}', nll / torch.log(torch.tensor(2.0, device=nll.device)),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'val/ppl_alpha_{band}', torch.exp(nll), on_step=False,
                 on_epoch=True, sync_dist=True)
    finally:
      self.complementary_masks = saved_comp


__all__ = ['BlockTrainer', 'LOSS_SPECIAL_CASE_POLICY']
