"""Chat-template training + Fast-dLLM decode gap closures."""

import torch

from discrete_diffusion.data.loaders import _nemotron_to_messages
from discrete_diffusion.sampling.block_sampler import BlockSampler
from tests.test_block_sampler import _MockBlockTrainer, _config


def test_nemotron_to_messages_roles():
  row = {
      'system_prompt': 'Be helpful.',
      'input': [{'role': 'user', 'content': 'Hi'}],
      'output': 'Hello!',
  }
  msgs = _nemotron_to_messages(row)
  assert msgs[0]['role'] == 'system'
  assert msgs[-1] == {'role': 'assistant', 'content': 'Hello!'}


def test_parallel_threshold_early_exit():
  model = _MockBlockTrainer(mode='masked', n=16)
  model.block_size = 8
  cfg = _config('masked')
  cfg.sampling.unmask_threshold = 0.0  # commit all masked immediately
  cfg.sampling.parallel_threshold_decode = True
  cfg.sampling.greedy = True
  sampler = BlockSampler(cfg)
  xt = model.prior_sample(1, 16)
  x0 = xt.clone()
  start, end = 0, 8
  xt, x0 = sampler._init_block(model, xt, x0, start, end)
  assert (xt[:, start:end] == model.mask_id).any()
  xt2, _ = sampler._denoise_block(
      model, xt.clone(), x0.clone(), start, end, num_steps=32, eps=1e-3)
  assert not (xt2[:, start:end] == model.mask_id).any()


def test_use_block_cache_requires_hierarchical_kv():
  cfg = _config('masked')
  cfg.sampling.use_block_cache = True
  cfg.sampling.hierarchical_kv = False
  try:
    BlockSampler(cfg)
    raised = False
  except ValueError as e:
    raised = True
    assert 'use_block_cache' in str(e)
  assert raised


def test_arpc_temperature_applied():
  model = _MockBlockTrainer(mode='uniform', n=16)
  cfg = _config('uniform')
  cfg.sampling.arpc_temperature = 2.0
  sampler = BlockSampler(cfg)
  logits = torch.ones(1, 4, model.vocab_size)
  scaled = sampler._scale_logits(logits)
  assert torch.allclose(scaled, logits / 2.0)
