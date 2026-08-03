"""Qwen2 backbone for block-diffusion training."""

from __future__ import annotations

import omegaconf
import torch
import torch.nn as nn

from .attention import block_diff_attention_mask
from .config import ForwardMode, QwenBlockConfig


class QwenBlockForCausalLM(nn.Module):
  """HF Qwen2 weights + block-diffusion attention.

  block_diff training: input ``[B, 2n]`` = concat(xt, x0) → logits ``[B, n, V]``.
  causal mode: standard HF forward for AR parity checks (G1).
  """

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    model_cfg = config.model

    self.qwen_cfg = QwenBlockConfig(
        hub_id=getattr(model_cfg, 'hub_id', 'Qwen/Qwen2.5-1.5B-Instruct'),
        block_size=int(getattr(config, 'block_size', 16)),
        length=int(getattr(model_cfg, 'length', 128)),
        forward_mode=getattr(model_cfg, 'forward_mode', 'block_diff'),
        attn_implementation=getattr(model_cfg, 'attn_implementation', 'eager'),
    )
    self.n_tokens = self.qwen_cfg.length
    self.block_size = self.qwen_cfg.block_size
    self.forward_mode: ForwardMode = self.qwen_cfg.forward_mode

    from transformers import AutoConfig, AutoModelForCausalLM
    hf_config = AutoConfig.from_pretrained(self.qwen_cfg.hub_id)
    hf_config._attn_implementation = self.qwen_cfg.attn_implementation
    load_pretrained = bool(getattr(model_cfg, 'load_pretrained', True))
    if load_pretrained:
      self.model = AutoModelForCausalLM.from_pretrained(
          self.qwen_cfg.hub_id, config=hf_config)
    else:
      # Pipeline 2 (pure block diffusion): same architecture, random init.
      self.model = AutoModelForCausalLM.from_config(hf_config)

    if getattr(model_cfg, 'gradient_checkpointing', False):
      self.model.gradient_checkpointing_enable()

    if vocab_size > 0 and vocab_size != self.model.config.vocab_size:
      self.model.resize_token_embeddings(vocab_size)

  def compare_hf_keys(self, other_state_dict: dict):
    own = set(self.model.state_dict().keys())
    other = set(other_state_dict.keys())
    matched = own & other
    return len(matched) / max(len(own), 1), sorted(own - other), sorted(other - own)

  def forward(self, indices, sigma=None, sample_mode=False, store_kv=False,
              block_size: int | None = None):
    del sigma, sample_mode, store_kv
    if self.forward_mode == 'causal':
      return self.model(input_ids=indices, use_cache=False).logits

    n = self.n_tokens
    if indices.shape[1] != 2 * n:
      raise ValueError(f'Expected seq len {2 * n}, got {indices.shape[1]}')

    bs = int(block_size) if block_size is not None else self.block_size
    dtype = next(self.model.parameters()).dtype
    with block_diff_attention_mask(self.model, n, bs, indices.device, dtype):
      out = self.model(input_ids=indices, use_cache=False)
    return out.logits[:, :n, :]

  @torch.no_grad()
  def causal_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
    prev = self.forward_mode
    self.forward_mode = 'causal'
    try:
      return self.forward(input_ids)
    finally:
      self.forward_mode = prev

  @torch.no_grad()
  def block_diff_logits(self, xt_x0: torch.Tensor) -> torch.Tensor:
    prev = self.forward_mode
    self.forward_mode = 'block_diff'
    try:
      return self.forward(xt_x0)
    finally:
      self.forward_mode = prev
