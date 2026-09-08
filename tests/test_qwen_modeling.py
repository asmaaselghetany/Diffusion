"""CPU regressions for the Qwen block wrapper."""

import copy

import torch

from discrete_diffusion.models.qwen.attention import block_diff_attention_mask
from discrete_diffusion.models.qwen.modeling import (
    QwenBlockForCausalLM,
    shared_block_position_ids,
)


def _tiny_wrapper(*, n: int = 4, vocab_size: int = 32):
  from transformers import Qwen2Config, Qwen2ForCausalLM

  config = Qwen2Config(
      vocab_size=vocab_size,
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=2,
      num_key_value_heads=2,
      max_position_embeddings=2 * n,
  )
  causal_lm = Qwen2ForCausalLM(config)
  wrapper = QwenBlockForCausalLM.__new__(QwenBlockForCausalLM)
  torch.nn.Module.__init__(wrapper)
  wrapper.model = causal_lm
  wrapper.n_tokens = n
  wrapper.block_size = 2
  wrapper.forward_mode = 'block_diff'
  return wrapper


def _full_projection_reference(wrapper, indices):
  n = wrapper.n_tokens
  positions = shared_block_position_ids(
      n, indices.device, batch_size=indices.shape[0])
  dtype = next(wrapper.model.parameters()).dtype
  with block_diff_attention_mask(
      wrapper.model, n, wrapper.block_size, indices.device, dtype):
    return wrapper.model(
        input_ids=indices,
        position_ids=positions,
        use_cache=False,
    ).logits[:, :n, :]


def test_block_forward_projects_only_required_hidden_states():
  torch.manual_seed(7)
  wrapper = _tiny_wrapper()
  indices = torch.randint(0, 32, (2, 2 * wrapper.n_tokens))
  reference = _full_projection_reference(wrapper, indices)

  projected_lengths = []
  handle = wrapper.model.lm_head.register_forward_pre_hook(
      lambda _module, args: projected_lengths.append(args[0].shape[1]))
  try:
    actual = wrapper(indices)
  finally:
    handle.remove()

  assert projected_lengths == [wrapper.n_tokens]
  assert actual.shape == (2, wrapper.n_tokens, 32)
  assert actual.dtype == torch.float32
  torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_block_prefix_projection_preserves_gradients():
  torch.manual_seed(11)
  optimized = _tiny_wrapper()
  reference = copy.deepcopy(optimized)
  indices = torch.randint(0, 32, (2, 2 * optimized.n_tokens))

  optimized(indices).square().mean().backward()
  _full_projection_reference(reference, indices).square().mean().backward()

  for (name, parameter), (ref_name, ref_parameter) in zip(
      optimized.named_parameters(), reference.named_parameters()):
    assert name == ref_name
    torch.testing.assert_close(
        parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_causal_forward_still_delegates_to_hf_model():
  torch.manual_seed(13)
  wrapper = _tiny_wrapper()
  indices = torch.randint(0, 32, (2, wrapper.n_tokens))
  expected = wrapper.model(input_ids=indices, use_cache=False).logits

  wrapper.forward_mode = 'causal'
  actual = wrapper(indices)

  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
