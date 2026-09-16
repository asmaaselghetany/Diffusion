"""Conversion baseline hygiene: chat template, Hub vocab keep, Nemotron defaults."""

from __future__ import annotations

from types import SimpleNamespace

from discrete_diffusion.contracts.special_tokens import ensure_special_tokens
from discrete_diffusion.data.conversion_baseline import (
    CONVERSION_SYSTEM_PROMPT,
    FAST_DLLM_SFT_CHAT_TEMPLATE,
    NEMOTRON_DEFAULT_SPLITS,
    NEMOTRON_PREPROCESSING_VERSION,
    apply_checkpoint_embed_vocab_size,
    apply_conversion_chat_template,
    checkpoint_embed_vocab_size,
    conversion_prefix_text,
)
from discrete_diffusion.data.loaders import _finalize_tokenizer
from discrete_diffusion.evaluations.decode_profiles import (
    conversion_prefix_text as decode_prefix_text,
)
from discrete_diffusion.models.qwen.modeling import QwenBlockForCausalLM


def test_nemotron_defaults_include_math_and_code(monkeypatch):
  monkeypatch.delenv('NEMOTRON_SFT_SPLITS', raising=False)
  monkeypatch.delenv('NEMOTRON_SFT_MAX_PER_SPLIT', raising=False)
  assert 'math' in NEMOTRON_DEFAULT_SPLITS
  assert 'code' in NEMOTRON_DEFAULT_SPLITS
  assert NEMOTRON_PREPROCESSING_VERSION.startswith(
      'qwen-chat-block-aligned-v3')
  from discrete_diffusion.data.conversion_baseline import (
      NEMOTRON_DEFAULT_MAX_PER_SPLIT,
      conversion_baseline_manifest,
      nemotron_resolved_caps,
  )
  assert NEMOTRON_DEFAULT_MAX_PER_SPLIT['math'] == 1_000_000
  assert NEMOTRON_DEFAULT_MAX_PER_SPLIT['code'] == 500_000
  caps = dict(nemotron_resolved_caps())
  assert caps['math'] == 1_000_000
  assert caps['code'] == 500_000
  man = conversion_baseline_manifest()
  assert man['preprocessing_version'] == NEMOTRON_PREPROCESSING_VERSION
  assert man['packing'].startswith('block_aligned')
  assert man['max_per_split']['math'] == 1_000_000
  assert 'chat_template_hash' in man



def test_conversion_template_matches_train_system_not_alibaba():
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(
      'Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
  msgs = [{'role': 'user', 'content': 'What is 2+2?'}]
  stock = tok.apply_chat_template(
      msgs, add_generation_prompt=True, tokenize=False)
  ours = apply_conversion_chat_template(
      tok, msgs, add_generation_prompt=True, tokenize=False)
  assert CONVERSION_SYSTEM_PROMPT in ours
  assert 'Alibaba Cloud' not in ours
  assert 'Alibaba Cloud' in stock or stock != ours
  assert ours.endswith('<|im_start|>assistant\n')
  assert '<|im_start|>user\nWhat is 2+2?' in ours


def test_conversion_prefix_text_uses_short_system():
  text = conversion_prefix_text(None)
  assert CONVERSION_SYSTEM_PROMPT in text
  assert 'Alibaba Cloud' not in text
  assert decode_prefix_text(None) == text


def test_finalize_tokenizer_keeps_hub_vocab_and_installs_template():
  from transformers import AutoTokenizer, AutoConfig
  tok = AutoTokenizer.from_pretrained(
      'Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
  cfg = AutoConfig.from_pretrained(
      'Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
  _finalize_tokenizer(tok)
  ids = ensure_special_tokens(tok)
  assert ids.mask_id is not None
  assert ids.vocab_size == int(cfg.vocab_size)
  assert ids.vocab_size > len(tok)  # padded Hub table
  assert getattr(tok, 'chat_template', None) == FAST_DLLM_SFT_CHAT_TEMPLATE


def test_qwen_block_resizes_to_requested_vocab(monkeypatch):
  """Train keeps 151936 (no-op); eval load of old ckpt may shrink to 151666."""
  calls = []

  class _FakeModel:
    def __init__(self):
      self.config = SimpleNamespace(vocab_size=151936)

    def resize_token_embeddings(self, n):
      calls.append(int(n))
      self.config.vocab_size = int(n)

    def gradient_checkpointing_enable(self):
      return None

  class _AMC:
    @staticmethod
    def from_pretrained(*args, **kwargs):
      del args, kwargs
      return SimpleNamespace(vocab_size=151936, _attn_implementation=None)

  class _AML:
    @staticmethod
    def from_pretrained(*args, **kwargs):
      del args, kwargs
      return _FakeModel()

  import transformers
  monkeypatch.setattr(transformers, 'AutoConfig', _AMC)
  monkeypatch.setattr(transformers, 'AutoModelForCausalLM', _AML)
  monkeypatch.setattr(
      'discrete_diffusion.models.qwen.modeling.'
      'assert_block_attention_hook_compatible',
      lambda m: None)

  cfg = SimpleNamespace(
      block_size=32,
      model=SimpleNamespace(
          hub_id='Qwen/Qwen2.5-1.5B-Instruct',
          length=128,
          forward_mode='block_diff',
          attn_implementation='sdpa',
          load_pretrained=True,
          gradient_checkpointing=False,
      ),
  )
  QwenBlockForCausalLM(cfg, vocab_size=151936)
  assert calls == []
  QwenBlockForCausalLM(cfg, vocab_size=151666)
  assert calls == [151666]


def test_checkpoint_embed_vocab_helpers():
  import torch
  ckpt = {
      'state_dict': {
          'backbone.model.model.embed_tokens.weight': torch.zeros(151666, 8),
      }
  }
  assert checkpoint_embed_vocab_size(ckpt) == 151666
  tok = SimpleNamespace()
  apply_checkpoint_embed_vocab_size(tok, 151666)
  assert tok._effective_vocab_size == 151666
