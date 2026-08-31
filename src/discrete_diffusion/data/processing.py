"""Text processing helpers used when grouping tokenized data."""

from __future__ import annotations

import itertools
import re

import torch

__all__ = [
    "wt_detokenizer",
    "ptb_detokenizer",
    "lm1b_detokenizer",
    "lambada_detokenizer",
    "scientific_papers_detokenizer",
    "_apply_detokenizer",
    "_pad_example_to_attention_blocks",
    "_group_texts",
]


def wt_detokenizer(string):
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
    x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


def _apply_detokenizer(detokenizer):
  def detok(text):
    for i, t in enumerate(text, 0):
      text[i] = detokenizer(t)
    return text
  return detok


def _pad_example_to_attention_blocks(
    token_ids: list[int],
    attention_block_size: int,
    pad_id: int,
) -> list[int]:
  """Pad one example so its length is a multiple of the diffusion block size."""
  if attention_block_size <= 1:
    return token_ids
  rem = len(token_ids) % attention_block_size
  if rem == 0:
    return token_ids
  return token_ids + [pad_id] * (attention_block_size - rem)


def _group_texts(
    examples,
    block_size,
    bos,
    eos,
    insert_special_tokens=True,
    attention_block_size: int = 1,
    pad_id: int = 0,
):
  padded_sequences = [
      _pad_example_to_attention_blocks(seq, attention_block_size, pad_id)
      for seq in examples['input_ids']
  ]
  concatenated_examples = list(itertools.chain(*padded_sequences))
  total_length = len(concatenated_examples)
  if insert_special_tokens:
    new_block_size = block_size - 2
  else:
    new_block_size = block_size
  total_length = (total_length // new_block_size) * new_block_size
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    if insert_special_tokens:
      chunk = (
          [bos]
          + concatenated_examples[i : i + new_block_size]
          + [eos])
    else:
      chunk = concatenated_examples[i : i + new_block_size]
    _values.append(chunk)
    mask = [0 if tid == pad_id else 1 for tid in chunk]
    _attn_masks.append(torch.tensor(mask, dtype=torch.long))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result
