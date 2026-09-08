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
    # Fast-dLLM shift: masked-only; Unif(V) DUO has no AR next-token shift.
    'shift_loss_targets': 'masked_only',
    # Fast-dLLM paired m/~m views; N/A for uniform replacement FP.
    'complementary_masks': 'masked_only',
    # Fast-dLLM p_mask=(1-eps)t+eps; ELBO uses α_eff=1-p (see _elbo_schedule_weights).
    'mask_schedule': 'masked_only',
    'loss_weighting': 'masked_only',
    # SUBS log-probs / mask-site NLL — absorbing parameterization.
    'subs_log_probs': 'masked_only',
    # Applied in nll() for both corruptions after the per-corruption loss.
    'ignore_bos': 'shared',
    # valid_tokens trim to T-1 only when shift shortens masked loss.
    'valid_tokens_shift_trim': 'masked_only_when_shift',
    # BlockGen pure-noise / CE-at-size specials apply to both arms.
    'pure_noise_block_sizes': 'shared',
    'loss_type_special_cases': 'shared',
    'block_weights': 'shared',
    # NLD joint AR — corruption-agnostic conversion recipe (axis D).
    'joint_ar_alpha': 'shared',
    'causal_clean_stream': 'shared',
    # Hybrid FP mix rate — hybrid arm only.
    'hybrid_p_uniform': 'hybrid_only',
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
    omegaconf.OmegaConf.set_struct(config.algo, False)
    config.algo.parameterization = 'subs'
    omegaconf.OmegaConf.set_struct(config.algo, True)

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
    self.loss_per_block_size = self._parse_loss_special_cases(
        getattr(config.algo, 'loss_type_special_cases', None))

    self.joint_ar_alpha = float(getattr(config.algo, 'joint_ar_alpha', 0.0) or 0.0)
    self.causal_clean_stream = bool(
        getattr(config.algo, 'causal_clean_stream', False))
    self.hybrid_p_uniform = float(
        getattr(config.algo, 'hybrid_p_uniform', 0.1) or 0.0)
    self.hybrid_decode = str(
        getattr(config.algo, 'hybrid_decode', 'masked') or 'masked')
    # Clean-stream logits from the last dual forward (joint AR, non-causal).
    self._pending_clean_logits: torch.Tensor | None = None

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
      return torch.randint(
          0, self.vocab_size, tuple(size),
          dtype=torch.int64, device=self.device)
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
      try:
        fp = hydra.utils.instantiate(
            fp_cfg, tokenizer=self.tokenizer, schedule=self.noise, **kwargs)
      except TypeError:
        fp = hydra.utils.instantiate(
            fp_cfg, tokenizer=self.tokenizer, schedule=self.noise)
        if isinstance(fp, BlockHybridForwardProcess):
          fp.p_uniform = self.hybrid_p_uniform
    elif self.forward_process_name == 'uniform':
      fp = BlockUniformForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_uniform')
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

    # Uniform / hybrid must not silently inherit masked-only hooks.
    if self.forward_process_name in ('uniform', 'hybrid'):
      if (self.shift_loss_targets
          and LOSS_SPECIAL_CASE_POLICY.get('shift_loss_targets')
          == 'masked_only'):
        raise ValueError(
            'algo.shift_loss_targets=true is masked-only '
            f'(LOSS_SPECIAL_CASE_POLICY); refuse on {self.forward_process_name}')
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
    sampling = getattr(self.config, 'sampling', None)
    if sampling is not None and bool(getattr(sampling, 'use_arpc', False)):
      if self.forward_process_name != 'uniform':
        raise ValueError(
            'sampling.use_arpc=true is uniform-oriented; refuse on '
            f'{self.forward_process_name} (BlockGen ARPC).')
      if 1 not in active:
        raise ValueError(
            'sampling.use_arpc=true requires block size 1 in '
            'algo.block_size_mixture or algo.block_weights '
            '(BlockGen ARPC prerequisite)')
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
      xt = self._forward_process(x0, t, block_size=block_size)
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
        return mask_positions * ce
      masked_neg_ce = mask_positions * (-ce)
      weighting = dalpha_t / (1.0 - alpha_t)
      return weighting * masked_neg_ce
    if weight == 'plain_ce':
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
    - ``shift_loss_targets``: masked-only (Fast-dLLM AR alignment).
    - ``subs_log_probs`` / mask-site CE: absorbing parameterization only.
    - ``complementary_masks``: applied in ``nll`` for masked FP only.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return uniform_block_nll_per_token(
        log_probs, xt, x0, alpha_t, dalpha_t, self.vocab_size)

  def _hybrid_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    """Hybrid ELBO surrogate matching ``BlockHybridForwardProcess``.

    FP replaces with MASK or Unif(V\\{MASK}). We score:
    - MASK sites with the absorbing (masked) ELBO
    - Unif-replaced sites with DUO/uniform ELBO using ``V-1`` (MASK
      excluded from the predictive simplex, matching the FP)
    - Clean sites contribute 0 (noise-site scoring, same spirit as SUBS)

    This is not a full GIDD transition ELBO; it is the block-path hybrid
    used for B4.
    """
    masked_term = self._masked_loss(logits, xt, x0, alpha_t, dalpha_t)
    # Exclude MASK from the uniform predictive simplex (FP never draws MASK
    # on the uniform branch).
    logits_u = logits.clone()
    logits_u[..., self.mask_id] = self.neg_infinity
    log_probs_u = F.log_softmax(logits_u, dim=-1)
    uniform_term = uniform_block_nll_per_token(
        log_probs_u, xt, x0, alpha_t, dalpha_t, self.vocab_size - 1)
    is_mask = (xt == self.mask_id)
    is_unif = (xt != x0) & (~is_mask)
    if masked_term.shape != uniform_term.shape:
      raise RuntimeError(
          'hybrid loss shape mismatch (shift on hybrid should be refused)')
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
  ) -> torch.Tensor:
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
          train_mode=False, block_size: int | None = None):
    del train_mode
    self._pending_clean_logits = None
    bsz = x0.shape[0]
    bs = (
        block_size if block_size is not None
        else self._sample_training_block_size(current_accumulation_step))
    t = sample_block_timesteps(
        bsz, self.num_tokens, bs, self.device,
        sampling_eps=self.sampling_eps,
        antithetic=self.antithetic_sampling,
        stratified_gamma=self.stratified_gamma)

    if bs in self.pure_noise_block_sizes:
      t = torch.ones_like(t)

    alpha_t, dalpha_t = self._elbo_schedule_weights(t)
    want_clean = (
        self.joint_ar_alpha > 0 and not self.causal_clean_stream)
    pad_mask = valid_tokens

    if (self.complementary_masks
        and isinstance(self._forward_process, BlockMaskedForwardProcess)):
      # Fast-dLLM Hub: paired m/~m, then ``torch.cat(..., dim=0)`` → fused 2B.
      _, move_mask = self._corrupt(
          x0, t, block_size=bs, return_move_mask=True)
      xt_a, xt_b = complementary_pair_from_mask(x0, move_mask, self.mask_id)
      if self.ignore_bos:
        xt_a[:, 0] = x0[:, 0]
        xt_b[:, 0] = x0[:, 0]
      if pad_mask is not None:
        supervised = pad_mask.bool()
        xt_a = torch.where(supervised, xt_a, x0)
        xt_b = torch.where(supervised, xt_b, x0)
      if self.complementary_batching == 'sequential':
        # Memory fallback: two B forwards; same summed NLL as fused.
        if want_clean:
          out_a = self._backbone_logits(
              xt_a, x0, block_size=bs, return_clean=True,
              attention_mask=pad_mask)
          logits_a, clean_a = out_a
          self._pending_clean_logits = clean_a
        else:
          logits_a = self._backbone_logits(
              xt_a, x0, block_size=bs, attention_mask=pad_mask)
        loss_a = self._loss_for_block(
            logits_a, xt_a, x0, alpha_t, dalpha_t, block_size=bs)
        logits_b = self._backbone_logits(
            xt_b, x0, block_size=bs, attention_mask=pad_mask)
        loss_b = self._loss_for_block(
            logits_b, xt_b, x0, alpha_t, dalpha_t, block_size=bs)
        loss = torch.cat([loss_a, loss_b], dim=0)
      else:
        xt_pair = torch.cat([xt_a, xt_b], dim=0)
        x0_pair = torch.cat([x0, x0], dim=0)
        pad_pair = (
            torch.cat([pad_mask, pad_mask], dim=0)
            if pad_mask is not None else None)
        alpha_pair = torch.cat([alpha_t, alpha_t], dim=0)
        dalpha_pair = torch.cat([dalpha_t, dalpha_t], dim=0)
        if want_clean:
          logits_pair, clean_pair = self._backbone_logits(
              xt_pair, x0_pair, block_size=bs, return_clean=True,
              attention_mask=pad_pair)
          self._pending_clean_logits = clean_pair[:bsz]
        else:
          logits_pair = self._backbone_logits(
              xt_pair, x0_pair, block_size=bs, attention_mask=pad_pair)
        loss = self._loss_for_block(
            logits_pair, xt_pair, x0_pair, alpha_pair, dalpha_pair,
            block_size=bs)
      valid_tokens = torch.cat([valid_tokens, valid_tokens], dim=0)
    else:
      xt = self._corrupt(
          x0, t, block_size=bs, corruption_mask=pad_mask)
      if want_clean:
        logits, clean = self._backbone_logits(
            xt, x0, block_size=bs, return_clean=True,
            attention_mask=pad_mask)
        self._pending_clean_logits = clean
      else:
        logits = self._backbone_logits(
            xt, x0, block_size=bs, attention_mask=pad_mask)
      loss = self._loss_for_block(
          logits, xt, x0, alpha_t, dalpha_t, block_size=bs)

    # masked_block_nll_per_token / uniform_block_nll_per_token already return
    # minimize-oriented terms (MDLM-equivalent: coeff * log_p with coeff<0).
    # Do NOT negate — that inverts the ELBO and rewards worse predictions.
    # CE special-cases return positive CE (also minimize-oriented).
    loss, valid_tokens = self._apply_ignore_bos_mask(loss, valid_tokens)
    # Mirror base.py: shift_loss_targets shortens loss to T-1 in _masked_loss;
    # trim the pad mask so multiply / BPD denom stay aligned.
    if (self.shift_loss_targets
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]
    return loss * valid_tokens

  def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
    # Keep pre-complementary / pre-shift-trim mask for the AR term (full T).
    ar_valid = valid_tokens
    nlls = self.nll(
        input_tokens, valid_tokens,
        current_accumulation_step=current_accumulation_step,
        train_mode=train_mode)
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

  def training_step(self, batch, batch_idx):
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    valid_tokens = self._batch_valid_tokens(batch)
    losses = self._loss(
        batch['input_ids'], valid_tokens,
        current_accumulation_step=current_accumulation_step, train_mode=True)
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
    losses = self._loss(batch['input_ids'], valid_tokens)
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
      self._log_t_bucketed_nll(batch['input_ids'], valid_tokens)
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
  def _log_t_bucketed_nll(self, x0: torch.Tensor, valid_tokens: torch.Tensor):
    """Layer-3 diagnostic: NLL vs corruption level (AR-init cliff detector).

    Logs ``val/{nll,bpd,ppl}_alpha_{lo}_{hi}`` for fixed α bands.
    Aggregate val/nll alone can hide a high-t cliff on ar2block_uniform.
    """
    x0, valid_tokens = self._process_model_input(x0, valid_tokens)
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
        xt = self._corrupt(x0, t, block_size=bs)
        logits = self._backbone_logits(
            xt, x0, block_size=bs, attention_mask=valid_tokens)
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
        nll = weighted.sum() / vt.sum().clamp(min=1)
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
