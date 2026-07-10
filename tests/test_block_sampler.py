"""Tier 0: BlockSampler shape checks with a tiny mock backbone."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from discrete_diffusion.sampling.block_sampler import BlockSampler


class _TinyBackbone(nn.Module):
  def __init__(self, n: int, vocab: int):
    super().__init__()
    self.n_tokens = n
    self.block_size = 8
    self.embed = nn.Embedding(vocab, 16)
    self.head = nn.Linear(16, vocab)

  def forward(self, indices, sigma=None, sample_mode=False, store_kv=False):
    del sigma, sample_mode, store_kv
    h = self.embed(indices).mean(dim=1, keepdim=True).expand(-1, self.n_tokens, -1)
    return self.head(h)


class _MockBlockTrainer(nn.Module):
  def __init__(self, n=16, vocab=32, mode='masked'):
    super().__init__()
    self.num_tokens = n
    self.block_size = 8
    self.vocab_size = vocab
    self.mask_id = vocab - 1
    self.sampling_eps = 1e-3
    self.forward_process_name = mode
    self.backbone = _TinyBackbone(n, vocab)
    self.tokenizer = SimpleNamespace(bos_token_id=0)

    from discrete_diffusion.noise_schedules.log_linear import LogLinear
    self.noise = LogLinear(eps=1e-3)

  def backbone_logits(self, xt, x0):
    del x0
    return self.backbone(xt)

  def prior_sample(self, *batch_dims):
    size = batch_dims if len(batch_dims) > 1 else batch_dims[0]
    if self.forward_process_name == 'uniform':
      return torch.randint(0, self.vocab_size, size, dtype=torch.int64)
    return torch.full(size, self.mask_id, dtype=torch.int64)


def _config(mode: str):
  return SimpleNamespace(
      algo=SimpleNamespace(forward_process_name=mode),
      sampling=SimpleNamespace(steps=4, inject_bos=False, use_float64=False),
  )


def test_block_sampler_masked_shape():
  model = _MockBlockTrainer(mode='masked')
  sampler = BlockSampler(_config('masked'))
  out = sampler.generate(model, num_samples=2, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (2, 16)
  assert out.min() >= 0
  assert out.max() < model.vocab_size


def test_block_sampler_uniform_shape():
  model = _MockBlockTrainer(mode='uniform')
  sampler = BlockSampler(_config('uniform'))
  out = sampler.generate(model, num_samples=1, num_steps=4, eps=1e-3, inject_bos=False)
  assert out.shape == (1, 16)
