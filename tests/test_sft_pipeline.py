"""Block-aligned SFT packing and assistant-only supervision."""

import torch

from discrete_diffusion.algorithms.block_trainer import BlockTrainer
from discrete_diffusion.data.processing import _group_block_aligned_sft


def test_sft_examples_are_padded_before_packing():
  result = _group_block_aligned_sft(
      {
          'input_ids': [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24, 25, 26]],
          'attention_mask': [[1] * 5, [1] * 7],
          'labels': [[-100, -100, 12, 13, 14], [-100, 21, 22, 23, 24, 25, 26]],
      },
      sequence_length=16,
      diffusion_block_size=4,
      mask_id=99,
  )

  assert result['input_ids'] == [[
      10, 11, 12, 13, 14, 99, 99, 99,
      20, 21, 22, 23, 24, 25, 26, 99,
  ]]
  assert result['attention_mask'] == [[
      1, 1, 1, 1, 1, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 0,
  ]]
  assert result['labels'][0][5:8] == [-100, -100, -100]
  assert result['labels'][0][15] == -100
  assert result['input_ids'][0][8] == 20


def test_sft_packing_keeps_supervised_tail_and_drops_prompt_only_rows():
  result = _group_block_aligned_sft(
      {
          'input_ids': [list(range(12))],
          'attention_mask': [[1] * 12],
          'labels': [[-100] * 8 + [8, 9, 10, 11]],
      },
      sequence_length=8,
      diffusion_block_size=4,
      mask_id=99,
  )

  assert result['input_ids'] == [[8, 9, 10, 11, 99, 99, 99, 99]]
  assert result['attention_mask'] == [[1, 1, 1, 1, 0, 0, 0, 0]]
  assert result['labels'] == [[8, 9, 10, 11, -100, -100, -100, -100]]


def test_sft_labels_control_training_and_corruption_mask():
  batch = {
      'attention_mask': torch.tensor([[1, 1, 1, 1, 0]]),
      'labels': torch.tensor([[-100, -100, 7, 8, -100]]),
  }
  valid = BlockTrainer._batch_valid_tokens(batch)
  assert torch.equal(valid, torch.tensor([[0, 0, 1, 1, 0]]))

  class _ReplaceEveryToken:
    def __call__(self, input_ids, _t, *, block_size):
      del block_size
      return torch.full_like(input_ids, 99)

  trainer = object.__new__(BlockTrainer)
  trainer._forward_process = _ReplaceEveryToken()
  trainer.ignore_bos = False
  x0 = torch.tensor([[10, 11, 12, 13, 14]])
  xt = trainer._corrupt(
      x0, torch.zeros_like(x0, dtype=torch.float32), block_size=5,
      corruption_mask=valid)
  assert torch.equal(xt, torch.tensor([[10, 11, 99, 99, 14]]))
