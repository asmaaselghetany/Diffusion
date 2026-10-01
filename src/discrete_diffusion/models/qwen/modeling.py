"""Qwen2 backbone for block-diffusion training."""

from __future__ import annotations

import omegaconf
import torch
import torch.nn as nn

from ...contracts.attention_hook import assert_block_attention_hook_compatible
from .attention import (
    block_diff_attention_mask,
    block_generation_attention_mask,
    eval_block_causal_attention_mask,
)
from .config import ForwardMode, QwenBlockConfig


def shared_block_position_ids(
    n: int,
    device: torch.device | str,
    batch_size: int = 1,
) -> torch.Tensor:
  """RoPE ids for ``concat(xt, x0)`` of length ``2n``.

  Fast-dLLM and BlockGen apply the *same* ``0..n-1`` positions to both
  halves so previous-block ``x0`` is nearby in RoPE space. Stock HF
  defaults to ``0..2n-1``, which puts clean context at offset ``n``.
  """
  pos = torch.arange(n, device=device)
  return pos.repeat(2).unsqueeze(0).expand(batch_size, 2 * n).contiguous()


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

    # Layer 2: fail at construction if transformers lacks the hook point
    # (jobs 137328/137329 class — do not wait for mid-train NCCL abort).
    assert_block_attention_hook_compatible(self.model)

    if getattr(model_cfg, 'gradient_checkpointing', False):
      self.model.gradient_checkpointing_enable()

    if vocab_size > 0:
      cur = int(self.model.config.vocab_size)
      # Match requested V exactly. New trains pass Hub-padded 151936 (no-op
      # after from_pretrained). Eval load of old ckpts passes 151666 so we
      # shrink to the checkpoint embed table.
      if vocab_size != cur:
        self.model.resize_token_embeddings(vocab_size)

    # Unifusion-style intra-block anneal progress (1.0 = full bi, default).
    # BlockTrainer sets this each step when algo.intra_block_attn_anneal_steps>0.
    self.intra_block_attn_open: float = 1.0
    # BlockGen multi-size: token-causal clean stream. Default Fast-dLLM block-causal.
    self.x0_causal: bool = False

  def compare_hf_keys(self, other_state_dict: dict):
    own = set(self.model.state_dict().keys())
    other = set(other_state_dict.keys())
    matched = own & other
    return len(matched) / max(len(own), 1), sorted(own - other), sorted(other - own)

  def forward(self, indices, sigma=None, sample_mode=False, store_kv=False,
              block_size: int | None = None, return_both: bool = False,
              active_len: int | None = None, attention_mask: torch.Tensor | None = None):
    del sigma, sample_mode, store_kv
    if self.forward_mode == 'causal':
      if return_both:
        raise ValueError('return_both is only valid in block_diff mode')
      return self.model(input_ids=indices, use_cache=False).logits

    n = int(active_len) if active_len is not None else self.n_tokens
    if active_len is not None and not (1 <= n <= self.n_tokens):
      raise ValueError(
          f'active_len={active_len} must be in [1, n_tokens={self.n_tokens}]')
    if indices.shape[1] != 2 * n:
      raise ValueError(f'Expected seq len {2 * n}, got {indices.shape[1]}')

    bs = int(block_size) if block_size is not None else self.block_size
    dtype = next(self.model.parameters()).dtype
    position_ids = shared_block_position_ids(
        n, indices.device, batch_size=indices.shape[0])
    with block_diff_attention_mask(
        self.model, n, bs, indices.device, dtype,
        padding_mask=attention_mask,
        intra_block_open=float(
            getattr(self, 'intra_block_attn_open', 1.0)),
        x0_causal=bool(getattr(self, 'x0_causal', False))):
      # Qwen2ForCausalLM projects every hidden state to the full vocabulary
      # before returning logits. Block diffusion only consumes the first half,
      # so projecting all 2n positions wastes lm-head compute and a large FP32
      # logits buffer that is immediately sliced away. Run the decoder over the
      # complete concat(xt, x0) sequence, but apply lm_head only to the hidden
      # states whose logits enter the objective (and optionally the clean half).
      decoder_out = self.model.model(
          input_ids=indices,
          position_ids=position_ids,
          use_cache=False,
          attention_mask=attention_mask,
          return_dict=True,
      )
      hidden = decoder_out[0]
      xt_logits = self.model.lm_head(hidden[:, :n, :]).float()
      if return_both:
        clean_logits = self.model.lm_head(hidden[:, n:, :]).float()
        return xt_logits, clean_logits
      return xt_logits

  def causal_train_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Trainable causal next-token logits (grads enabled)."""
    prev = self.forward_mode
    self.forward_mode = 'causal'
    try:
      return self.forward(input_ids)
    finally:
      self.forward_mode = prev

  @torch.no_grad()
  def causal_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.causal_train_logits(input_ids)

  @torch.no_grad()
  def causal_next_with_cache(
      self,
      input_ids: torch.Tensor,
      past_key_values=None,
  ):
    """Causal decode step with HF KV cache (ARPC / hierarchical AR path).

    Returns ``(next_logits [B, V], past_key_values)``.
    """
    prev = self.forward_mode
    self.forward_mode = 'causal'
    try:
      if past_key_values is None:
        out = self.model(
            input_ids=input_ids, use_cache=True, past_key_values=None)
      else:
        out = self.model(
            input_ids=input_ids[:, -1:],
            use_cache=True,
            past_key_values=past_key_values)
      return out.logits[:, -1, :], out.past_key_values
    finally:
      self.forward_mode = prev

  def block_train_logits(
      self,
      xt: torch.Tensor,
      *,
      active_len: int | None = None,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Trainable single-stream block-causal logits (Hub ``eval_block_diff_mask``).

    Same graph as ``block_eval_logits`` but with gradients enabled so train
    packing can match single-stream decode (``algo.single_stream_train``).
    """
    a = int(active_len) if active_len is not None else xt.shape[1]
    if not (1 <= a <= xt.shape[1]):
      raise ValueError(f'active_len={a} out of range for seq {xt.shape[1]}')
    bs = int(block_size) if block_size is not None else self.block_size
    ids = xt[:, :a]
    attn = attention_mask[:, :a] if attention_mask is not None else None
    dtype = next(self.model.parameters()).dtype
    position_ids = torch.arange(a, device=xt.device).unsqueeze(0).expand(
        xt.shape[0], a).contiguous()
    with eval_block_causal_attention_mask(
        self.model, a, bs, xt.device, dtype, cache_seq_len=0,
        padding_mask=attn):
      out = self.model.model(
          input_ids=ids,
          position_ids=position_ids,
          attention_mask=attn,
          use_cache=False,
          return_dict=True,
      )
      return self.model.lm_head(out[0]).float()

  @torch.no_grad()
  def block_gen_logits(
      self,
      prefix_ids: torch.Tensor,
      xt_block: torch.Tensor,
      *,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
      x0_causal: bool = False,
  ) -> torch.Tensor:
    """BlockGen generate packing: ``[clean_prefix | noisy_block]`` → block logits.

    Matches ``third_party/blockgen`` ``forward_generate`` attention structure
    (``block_generation_mask``). Absolute RoPE ids = sequence positions
    ``0..ctx+xt-1`` so packed layout stays aligned with token indices.
    Returns ``[B, xt_len, V]``.
    """
    if prefix_ids.dim() != 2 or xt_block.dim() != 2:
      raise ValueError('prefix_ids and xt_block must be [B, L]')
    if prefix_ids.shape[0] != xt_block.shape[0]:
      raise ValueError(
          f'batch mismatch prefix={prefix_ids.shape[0]} xt={xt_block.shape[0]}')
    ctx_len = int(prefix_ids.shape[1])
    xt_len = int(xt_block.shape[1])
    if xt_len < 1:
      raise ValueError('xt_block must be non-empty')
    bs = int(block_size) if block_size is not None else self.block_size
    ids = torch.cat([prefix_ids, xt_block], dim=1)
    pos = torch.arange(ctx_len + xt_len, device=ids.device)
    position_ids = pos.unsqueeze(0).expand(ids.shape[0], -1).contiguous()
    dtype = next(self.model.parameters()).dtype
    with block_generation_attention_mask(
        self.model, ctx_len, xt_len, bs, ids.device, dtype,
        padding_mask=attention_mask, x0_causal=x0_causal):
      out = self.model.model(
          input_ids=ids,
          position_ids=position_ids,
          attention_mask=attention_mask,
          use_cache=False,
          return_dict=True,
      )
      return self.model.lm_head(out[0][:, ctx_len:, :]).float()

  @torch.no_grad()
  def block_eval_logits(
      self,
      xt: torch.Tensor,
      *,
      active_len: int | None = None,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Hub-style single-stream block-causal decode logits (no dual concat).

    Forwards ``xt[:, :A]`` under ``eval_block_diff_mask``. Matches Hub generate
    when committed prefix has ``xt == x0`` (quality-equivalent to train graph).
    """
    return self.block_train_logits(
        xt, active_len=active_len, block_size=block_size,
        attention_mask=attention_mask)

  @torch.no_grad()
  def block_eval_prefill(
      self,
      xt: torch.Tensor,
      *,
      active_len: int,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ):
    """Single-stream DualCache prefill (Hub ``use_block_cache`` shape)."""
    from .dual_cache import single_stream_prefill
    bs = int(block_size) if block_size is not None else self.block_size
    attn = attention_mask[:, :active_len] if attention_mask is not None else None
    return single_stream_prefill(
        self.model, xt[:, :active_len], active_len=active_len, block_size=bs,
        attention_mask=attn)

  @torch.no_grad()
  def block_eval_replace(
      self,
      xt: torch.Tensor,
      *,
      active_len: int,
      window: tuple[int, int],
      cache,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Single-stream DualCache ``replace_position`` splice."""
    from .dual_cache import single_stream_replace
    bs = int(block_size) if block_size is not None else self.block_size
    attn = attention_mask[:, :active_len] if attention_mask is not None else None
    return single_stream_replace(
        self.model, xt, active_len=active_len, window=window,
        block_size=bs, cache=cache, attention_mask=attn)

  @torch.no_grad()
  def block_diff_prefill(
      self,
      xt: torch.Tensor,
      x0: torch.Tensor,
      *,
      active_len: int,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ):
    """DualCache prefill: truncated dual-stream forward + store per-layer K/V."""
    from .dual_cache import dual_stream_prefill
    bs = int(block_size) if block_size is not None else self.block_size
    xt_a = xt[:, :active_len]
    x0_a = x0[:, :active_len]
    x_in = torch.cat([xt_a, x0_a], dim=-1)
    attn = attention_mask[:, :active_len] if attention_mask is not None else None
    position_ids = shared_block_position_ids(
        active_len, xt.device, batch_size=xt.shape[0])
    return dual_stream_prefill(
        self.model, x_in, active_len=active_len, block_size=bs,
        position_ids=position_ids, attention_mask=attn)

  @torch.no_grad()
  def block_diff_replace(
      self,
      xt: torch.Tensor,
      x0: torch.Tensor,
      *,
      active_len: int,
      window: tuple[int, int],
      cache,
      block_size: int | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """DualCache replace_position splice for an active window."""
    from .dual_cache import dual_stream_replace
    bs = int(block_size) if block_size is not None else self.block_size
    attn = attention_mask[:, :active_len] if attention_mask is not None else None
    return dual_stream_replace(
        self.model, xt, x0, active_len=active_len, window=window,
        block_size=bs, cache=cache, attention_mask=attn)

  @torch.no_grad()
  def block_diff_logits(self, xt_x0: torch.Tensor) -> torch.Tensor:
    prev = self.forward_mode
    self.forward_mode = 'block_diff'
    try:
      return self.forward(xt_x0)
    finally:
      self.forward_mode = prev
