"""Matched AR SFT trainer (causal next-token CE on Qwen).

Paper cell C3: same data / steps / LR / batch as ``block_qwen`` AR→block arms,
but standard causal LM loss — no ``concat(xt, x0)``, no block corruption.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import Loss, TrainerBase, ensure_mask_token
from ..contracts.special_tokens import (
    SpecialTokenIds,
    assert_same_mask_id,
    ensure_special_tokens,
)


class ArSftTrainer(TrainerBase):
  """Causal next-token SFT on a Qwen backbone in ``forward_mode=causal``."""

  def __init__(self, config, tokenizer):
    self.token_ids: SpecialTokenIds = ensure_special_tokens(tokenizer)
    self.mask_id, vocab_size = self.token_ids.mask_id, self.token_ids.vocab_size
    mid, vs = ensure_mask_token(tokenizer)
    assert_same_mask_id(self.mask_id, mid, where='ensure_mask_token')
    if vs != vocab_size:
      raise AssertionError(
          f'vocab_size mismatch: special_tokens={vocab_size} '
          f'ensure_mask_token={vs}')
    # Force causal forward before backbone instantiate in TrainerBase.
    import omegaconf
    omegaconf.OmegaConf.set_struct(config.model, False)
    config.model.forward_mode = 'causal'
    omegaconf.OmegaConf.set_struct(config.model, True)

    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.ignore_bos = bool(getattr(config.algo, 'ignore_bos', True))
    self._validate_configuration()

  def _validate_configuration(self):
    if self.time_conditioning:
      raise ValueError('ArSftTrainer expects algo.time_conditioning=False')
    mode = getattr(self.backbone, 'forward_mode', None)
    if mode != 'causal':
      raise ValueError(
          f'ArSftTrainer requires model.forward_mode=causal, got {mode}')

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def nll(self, x0, valid_tokens, current_accumulation_step=None,
          train_mode=False):
    del current_accumulation_step, train_mode
    # Causal HF logits at position i predict token i+1.
    logits = self.backbone(x0, sigma=None)
    shift_logits = logits[:, :-1, :]
    shift_labels = x0[:, 1:]
    shift_valid = valid_tokens[:, 1:].to(shift_logits.dtype)
    if self.ignore_bos:
      # Do not train on predicting the first content token from BOS only if
      # position 0 is marked invalid; still zero the BOS→tok1 term when
      # ignore_bos mirrors block arms.
      shift_valid = shift_valid.clone()
      shift_valid[:, 0] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return nll * shift_valid

  def _loss(self, x0, valid_tokens, current_accumulation_step=None,
            train_mode=False):
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
    nlls = self.nll(
        input_tokens, valid_tokens,
        current_accumulation_step=current_accumulation_step,
        train_mode=train_mode)
    nll_sum = nlls.sum()
    vt = valid_tokens[:, 1:].clone()
    if self.ignore_bos:
      vt[:, 0] = 0
    num_tokens = vt.sum()
    token_nll = nll_sum / num_tokens.clamp(min=1)
    return Loss(loss=token_nll, nlls=nll_sum, num_tokens=num_tokens)

  def training_step(self, batch, batch_idx):
    current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
    losses = self._loss(
        batch['input_ids'], batch['attention_mask'],
        current_accumulation_step=current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.num_tokens)
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
    return losses.loss

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=None):
    del num_steps, eps
    # Free-gen from BOS via HF generate (matched AR baseline).
    bos = self.tokenizer.bos_token_id
    if bos is None:
      bos = self.tokenizer.eos_token_id
    input_ids = torch.full(
        (num_samples, 1), bos, dtype=torch.long, device=self.device)
    max_new = int(getattr(self.config.sampling, 'max_new_tokens',
                          self.num_tokens - 1))
    max_new = min(max_new, self.num_tokens - 1)
    out = self.backbone.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        top_p=float(getattr(self.config.sampling, 'p_nucleus', 0.9)),
        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
    )
    # Pad / trim to model.length for gen-PPL harness compatibility.
    T = self.num_tokens
    if out.shape[1] < T:
      pad = torch.full(
          (num_samples, T - out.shape[1]),
          self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
          dtype=out.dtype, device=out.device)
      out = torch.cat([out, pad], dim=1)
    return out[:, :T]


__all__ = ['ArSftTrainer']
