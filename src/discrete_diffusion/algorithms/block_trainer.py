"""Unified block diffusion trainer (masked + uniform forward processes).

Switch ``algo.forward_process`` between masked (absorbing) and uniform (BlockGen).
Does **not** use BD3LM or BlockDiT.
"""

from __future__ import annotations

import math

import hydra.utils
import omegaconf
import torch
import torch.nn.functional as F

from ..forward_process.block_masked import (
    BlockMaskedForwardProcess,
    sample_block_timesteps,
)
from ..forward_process.block_uniform import BlockUniformForwardProcess
from ..noise_schedules import LogLinear
from .base import Loss, TrainerBase, ensure_mask_token
from ..contracts.special_tokens import (
    SpecialTokenIds,
    assert_same_mask_id,
    ensure_special_tokens,
)
from ..losses.block_elbo import (
    masked_block_nll_per_token,
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
    # Absorbing polarity flip; N/A for uniform replacement FP.
    'complementary_masks': 'masked_only',
    # SUBS log-probs / mask-site NLL — absorbing parameterization.
    'subs_log_probs': 'masked_only',
    # Applied in nll() for both corruptions after the per-corruption loss.
    'ignore_bos': 'shared',
    # valid_tokens trim to T-1 only when shift shortens masked loss.
    'valid_tokens_shift_trim': 'masked_only_when_shift',
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

    self.shift_loss_targets = getattr(config.algo, 'shift_loss_targets', False)
    self.complementary_masks = bool(
        getattr(config.algo, 'complementary_masks', False))
    mixture = getattr(config.algo, 'block_size_mixture', None) or []
    self.block_size_mixture = [int(x) for x in mixture]
    stratified = getattr(config.algo, 'stratified_gamma', None)
    self.stratified_gamma = (
        float(stratified) if stratified is not None else None)
    self._init_forward_process()
    self._validate_configuration()
    self._logged_init_metrics = False

  def backbone_logits(self, xt: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return self._backbone_logits(xt, x0, block_size=self.block_size)

  def prior_sample(self, *batch_dims):
    size = (
        batch_dims[0]
        if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list))
        else batch_dims)
    if self.forward_process_name == 'uniform':
      return torch.randint(
          0, self.vocab_size, tuple(size),
          dtype=torch.int64, device=self.device)
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
      fp = hydra.utils.instantiate(
          fp_cfg, tokenizer=self.tokenizer, schedule=self.noise)
    elif self.forward_process_name == 'uniform':
      fp = BlockUniformForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_uniform')
    else:
      fp = BlockMaskedForwardProcess(
          tokenizer=self.tokenizer, schedule=self.noise, name='block_masked')
    self._forward_process = fp
    # L0: FP mask id must match trainer (masked path).
    fp_mask = getattr(fp, 'mask_id', None)
    if fp_mask is not None:
      assert_same_mask_id(
          self.mask_id, int(fp_mask), where='forward_process.mask_id')

  def _validate_configuration(self):
    if self.time_conditioning:
      raise ValueError('BlockTrainer expects algo.time_conditioning=False')
    if self.forward_process_name not in ('masked', 'uniform'):
      raise ValueError(f'Unknown forward_process_name={self.forward_process_name}')
    if self.num_tokens % self.block_size != 0:
      raise ValueError('model.length must be divisible by block_size')
    for bs in self.block_size_mixture:
      if self.num_tokens % bs != 0:
        raise ValueError(
            f'model.length must be divisible by block_size_mixture entry {bs}')
    # Uniform must not silently inherit masked-only hooks (symmetry policy).
    if self.forward_process_name == 'uniform':
      if (self.shift_loss_targets
          and LOSS_SPECIAL_CASE_POLICY.get('shift_loss_targets')
          == 'masked_only'):
        raise ValueError(
            'algo.shift_loss_targets=true is masked-only '
            '(LOSS_SPECIAL_CASE_POLICY); refuse on uniform arm')
      if (self.complementary_masks
          and LOSS_SPECIAL_CASE_POLICY.get('complementary_masks')
          == 'masked_only'):
        raise ValueError(
            'algo.complementary_masks=true is masked-only '
            '(LOSS_SPECIAL_CASE_POLICY); refuse on uniform arm')

  def _sample_training_block_size(self) -> int:
    if not self.block_size_mixture:
      return self.block_size
    idx = int(torch.randint(
        0, len(self.block_size_mixture), (1,), device=self.device).item())
    return self.block_size_mixture[idx]

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def _corrupt(self, x0: torch.Tensor, t: torch.Tensor,
               *, block_size: int) -> torch.Tensor:
    if isinstance(self._forward_process, BlockMaskedForwardProcess):
      xt, _ = self._forward_process(
          x0, t, block_size=block_size,
          complementary=self.complementary_masks)
    else:
      xt = self._forward_process(x0, t, block_size=block_size)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    return xt

  def _backbone_logits(self, xt: torch.Tensor, x0: torch.Tensor,
                       *, block_size: int | None = None) -> torch.Tensor:
    x_in = torch.cat([xt, x0], dim=-1)
    return self.backbone(x_in, sigma=None, block_size=block_size)

  def _masked_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    # Shift must mirror Diffusion.nll (base.py): raw next-token CE on
    # mask positions only. Applying SUBS *then* shifting scores x0[i+1]
    # under a one-hot on xt[i] for unmasked i → ~neg_infinity NLL and
    # trainer/loss ~1e6 (Track 2 jobs 138068 / 138098).
    if self.shift_loss_targets:
      logits = logits[:, :-1]
      x0 = x0[:, 1:]
      xt = xt[:, 1:]
      alpha_t = alpha_t[:, 1:]
      dalpha_t = dalpha_t[:, 1:]
      ce = -logits.log_softmax(-1).gather(
          -1, x0.unsqueeze(-1)).squeeze(-1)
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      masked_neg_ce = mask_positions * (-ce)
      weighting = dalpha_t / (1.0 - alpha_t)
      return weighting * masked_neg_ce
    log_probs = subs_log_probs(logits, xt, self.mask_id, self.neg_infinity)
    return masked_block_nll_per_token(log_probs, x0, alpha_t, dalpha_t)

  def _uniform_loss(self, logits, xt, x0, alpha_t, dalpha_t):
    """DUO/UDLM closed-form uniform NLL (explicit ``x0==xt`` handled inside).

    Intentional non-applications vs ``_masked_loss`` (see
    ``LOSS_SPECIAL_CASE_POLICY``):
    - ``shift_loss_targets``: masked-only (Fast-dLLM AR alignment).
    - ``subs_log_probs`` / mask-site CE: absorbing parameterization only.
    - ``complementary_masks``: applied in ``_corrupt`` for masked FP only.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return uniform_block_nll_per_token(
        log_probs, xt, x0, alpha_t, dalpha_t, self.vocab_size)

  def nll(self, x0, valid_tokens, current_accumulation_step=None,
          train_mode=False, block_size: int | None = None):
    del train_mode, current_accumulation_step
    bsz = x0.shape[0]
    bs = block_size if block_size is not None else self._sample_training_block_size()
    t = sample_block_timesteps(
        bsz, self.num_tokens, bs, self.device,
        sampling_eps=self.sampling_eps,
        antithetic=self.antithetic_sampling,
        stratified_gamma=self.stratified_gamma)

    dalpha_t = self.noise.alpha_prime_t(t)
    alpha_t = self.noise.alpha_t(t)

    xt = self._corrupt(x0, t, block_size=bs)
    logits = self._backbone_logits(xt, x0, block_size=bs)

    if self.forward_process_name == 'uniform':
      loss = self._uniform_loss(logits, xt, x0, alpha_t, dalpha_t)
    else:
      loss = self._masked_loss(logits, xt, x0, alpha_t, dalpha_t)

    # masked_block_nll_per_token / uniform_block_nll_per_token already return
    # minimize-oriented terms (MDLM-equivalent: coeff * log_p with coeff<0).
    # Do NOT negate — that inverts the ELBO and rewards worse predictions.
    if self.ignore_bos:
      loss[:, 0] = 0
      valid_tokens = valid_tokens.clone()
      valid_tokens[:, 0] = 0
    # Mirror base.py: shift_loss_targets shortens loss to T-1 in _masked_loss;
    # trim the pad mask so multiply / BPD denom stay aligned.
    if (self.shift_loss_targets
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]
    return loss * valid_tokens

  def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
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
    nll_sum = nlls.sum()
    num_tokens = valid_tokens.sum()
    token_nll = nll_sum / num_tokens.clamp(min=1)
    return Loss(loss=token_nll, nlls=nll_sum, num_tokens=num_tokens)

  def training_step(self, batch, batch_idx):
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    losses = self._loss(
        batch['input_ids'], batch['attention_mask'],
        current_accumulation_step=current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.num_tokens)
    # Step-level train metrics: with max_steps + huge SFT epochs,
    # on_train_epoch_end (train/nll|bpd|ppl) almost never fires.
    # sync_dist=False: avoid NCCL barriers every log_every_n_steps; rank0
    # WandB is enough for train curves.
    nll = losses.loss.detach()
    self.log('trainer/loss', nll, on_step=True, on_epoch=False,
             sync_dist=False, prog_bar=True)
    self.log('train/nll', nll, on_step=True, on_epoch=False, sync_dist=False)
    self.log('train/bpd', nll / math.log(2), on_step=True, on_epoch=False,
             sync_dist=False)
    self.log('train/ppl', torch.exp(nll), on_step=True, on_epoch=False,
             sync_dist=False)
    return losses.loss

  def validation_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'], batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.num_tokens)
    if bool(getattr(self.config.eval, 't_bucketed_nll', False)):
      self._log_t_bucketed_nll(batch['input_ids'], batch['attention_mask'])
    return losses.loss

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
    for lo, hi in bands:
      alpha_mid = 0.5 * (lo + hi)
      # alpha = 1 - (1-eps)*t  →  t = (1-alpha)/(1-eps)
      t_val = (1.0 - alpha_mid) / max(1.0 - eps, 1e-8)
      t = torch.full((bsz, seq), t_val, device=self.device, dtype=torch.float32)
      # Constant within each block (match training geometry).
      n_blocks = seq // bs
      for bi in range(n_blocks):
        sl = slice(bi * bs, (bi + 1) * bs)
        t[:, sl] = t[:, bi * bs: bi * bs + 1]
      alpha_t = self.noise.alpha_t(t)
      dalpha_t = self.noise.alpha_prime_t(t)
      xt = self._corrupt(x0, t, block_size=bs)
      logits = self._backbone_logits(xt, x0, block_size=bs)
      if self.forward_process_name == 'uniform':
        loss = self._uniform_loss(logits, xt, x0, alpha_t, dalpha_t)
      else:
        loss = self._masked_loss(logits, xt, x0, alpha_t, dalpha_t)
      vt = valid_tokens
      if self.ignore_bos:
        loss = loss.clone()
        loss[:, 0] = 0
        vt = valid_tokens.clone()
        vt[:, 0] = 0
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


__all__ = ['BlockTrainer']
