"""Top-level loader API for discrete diffusion training."""

from __future__ import annotations

import functools
import os
from typing import Optional

import datasets
import tokenizers
import torch
import transformers

from .. import utils
from .datasets import (
    _collate_tensor_dict,
    generate_synthetic_dataset,
    get_lambada_test_dataset,
    get_text8_dataset,
)
from .processing import (
    _apply_detokenizer,
    _group_texts,
    lm1b_detokenizer,
    lambada_detokenizer,
    ptb_detokenizer,
    scientific_papers_detokenizer,
    wt_detokenizer,
)
from .tokenizers import SyntheticTokenizer, Text8Tokenizer
from .flex_chunking import chunk_documents

LOGGER = utils.get_logger(__name__)

# Llama-Nemotron post-training SFT (Fast-dLLM-style data). Full code+math is
# tens of millions of rows / ~100GB+; default to chat+safety+science and allow
# optional capped code/math via NEMOTRON_SFT_SPLITS / NEMOTRON_SFT_MAX_PER_SPLIT.
_NEMOTRON_HUB = 'nvidia/Llama-Nemotron-Post-Training-Dataset'
_NEMOTRON_CONFIG = 'SFT'
_NEMOTRON_DEFAULT_SPLITS = ('chat', 'safety', 'science')
_NEMOTRON_VALID_SIZE = 5000
_NEMOTRON_DEFAULT_MAX_PER_SPLIT = {
    'chat': None,
    'safety': None,
    'science': None,
    # Caps only apply when these splits are explicitly enabled.
    'code': 100_000,
    'math': 100_000,
}


def _default_num_proc() -> int:
  if hasattr(os, 'sched_getaffinity'):
    return len(os.sched_getaffinity(0))
  return os.cpu_count() or 1


def _nemotron_split_list() -> list[str]:
  raw = os.environ.get('NEMOTRON_SFT_SPLITS', '').strip()
  if not raw:
    return list(_NEMOTRON_DEFAULT_SPLITS)
  splits = [s.strip() for s in raw.split(',') if s.strip()]
  if not splits:
    raise ValueError('NEMOTRON_SFT_SPLITS is set but empty')
  return splits


def _nemotron_max_for_split(split: str) -> int | None:
  raw = os.environ.get('NEMOTRON_SFT_MAX_PER_SPLIT', '').strip()
  overrides: dict[str, int | None] = {}
  if raw:
    for part in raw.split(','):
      if not part.strip():
        continue
      if '=' not in part:
        raise ValueError(
            f'NEMOTRON_SFT_MAX_PER_SPLIT entries must be split=N, got {part!r}')
      key, val = part.split('=', 1)
      key = key.strip()
      val = val.strip().lower()
      overrides[key] = None if val in {'none', 'all', ''} else int(val)
  if split in overrides:
    return overrides[split]
  return _NEMOTRON_DEFAULT_MAX_PER_SPLIT.get(split)


def _nemotron_to_text(example: dict) -> dict:
  """Flatten Nemotron SFT chat rows into a single text field for wrapping."""
  parts: list[str] = []
  sys_p = (example.get('system_prompt') or '').strip()
  if sys_p:
    parts.append(f'### System:\n{sys_p}')

  msgs = example.get('input') or []
  user_bits: list[str] = []
  if isinstance(msgs, list):
    for m in msgs:
      if isinstance(m, dict):
        role = (m.get('role') or 'user').lower()
        content = (m.get('content') or '').strip()
        if not content:
          continue
        if role == 'system' and not sys_p:
          parts.insert(0, f'### System:\n{content}')
        else:
          user_bits.append(content)
      else:
        user_bits.append(str(m).strip())
  elif msgs:
    user_bits.append(str(msgs).strip())

  instruction = '\n'.join(b for b in user_bits if b).strip()
  output = (example.get('output') or '').strip()
  parts.append(f'### Instruction:\n{instruction}')
  parts.append(f'### Response:\n{output}')
  return {'text': '\n\n'.join(parts)}


def _load_nemotron_sft(
    *,
    cache_dir: str,
    num_proc: int,
    revision: Optional[str],
) -> datasets.Dataset:
  splits = _nemotron_split_list()
  pieces: list[datasets.Dataset] = []
  for split in splits:
    LOGGER.info('Loading Nemotron SFT split=%s from %s', split, _NEMOTRON_HUB)
    ds = datasets.load_dataset(
        _NEMOTRON_HUB,
        _NEMOTRON_CONFIG,
        split=split,
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )
    cap = _nemotron_max_for_split(split)
    if cap is not None and len(ds) > cap:
      LOGGER.info('Subsampling Nemotron split=%s: %s -> %s', split, len(ds), cap)
      ds = ds.shuffle(seed=0).select(range(cap))
    pieces.append(ds)

  if len(pieces) == 1:
    full = pieces[0]
  else:
    # Align columns across splits (safety/chat/science share the SFT schema).
    cols = set(pieces[0].column_names)
    for p in pieces[1:]:
      cols &= set(p.column_names)
    cols = sorted(cols)
    pieces = [p.remove_columns([c for c in p.column_names if c not in cols])
              for p in pieces]
    full = datasets.concatenate_datasets(pieces)

  full = full.shuffle(seed=0)
  mapped = full.map(
      _nemotron_to_text,
      remove_columns=full.column_names,
      num_proc=num_proc,
      desc='Nemotron SFT to text',
  )
  return mapped


__all__ = [
    "get_tokenizer",
    "get_dataset",
    "get_dataloaders",
]


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                insert_special_tokens=True,
                block_size=1024,
                num_proc=_default_num_proc(),
                streaming=False,
                revision: Optional[str] = None,
                min_length: int = 0,
                chunking: str = "none",
                attention_block_size: int = 1):
  chunking_mode = (chunking or "none").lower()
  if chunking_mode not in {"none", "double_newline"}:
    raise ValueError(f"Unsupported chunking mode: {chunking_mode}")
  if wrap and chunking_mode != "none":
    raise ValueError("Delimiter-based chunking only applies when wrap=False.")
  eos_tag = ""
  if not insert_eos:
    eos_tag += "_eosFalse"
  if not insert_special_tokens:
    eos_tag += "_specialFalse"
  min_len_tag = f"_min{min_length}" if (min_length and not wrap) else ""
  chunk_tag = "_flexchunk" if (not wrap and chunking_mode != "none") else ""
  align_tag = (
      f"_abl{attention_block_size}"
      if wrap and attention_block_size > 1 else "")
  if wrap:
    filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped{align_tag}{eos_tag}.dat"
  else:
    filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped{chunk_tag}{eos_tag}{min_len_tag}.dat"
  _path = os.path.join(cache_dir, filename)

  if utils.fsspec_exists(_path):
    LOGGER.info("Loading data from: %s", _path)
    return datasets.load_from_disk(_path).with_format("torch")
  LOGGER.info("Generating new data at: %s", _path)
  LOGGER.info("streaming=%s", streaming)

  crop_train = dataset_name == "text8-crop"
  if mode == "train" and crop_train:
    block_size *= 2

  if dataset_name == "wikitext103":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-103-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "wikitext2":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-2-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "ptb":
    dataset = datasets.load_dataset(
      "ptb_text_only",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "lambada":
    dataset = get_lambada_test_dataset()
  elif dataset_name == "text8":
    assert wrap
    assert revision is None
    dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
  elif dataset_name == "text8-crop":
    assert revision is None
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == "openwebtext-train":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[:-100000]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
  elif dataset_name == "openwebtext-valid":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[-100000:]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
  elif dataset_name in ("alpaca-train", "alpaca-valid"):
    _alpaca_valid_size = 2000
    _full = datasets.load_dataset(
        "yahma/alpaca-cleaned",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True)
    _n = len(_full)
    _split = max(_n - _alpaca_valid_size, 1)
    if dataset_name == "alpaca-train":
      dataset = _full.select(range(_split))
    else:
      dataset = _full.select(range(_split, _n))

    def _alpaca_to_text(example):
      inp = (example.get("input") or "").strip()
      if inp:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{example['output']}")
      else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}")
      return {"text": text}

    dataset = dataset.map(
        _alpaca_to_text,
        remove_columns=_full.column_names,
        num_proc=num_proc,
        desc="Alpaca to text")
  elif dataset_name in ("nemotron-sft-train", "nemotron-sft-valid"):
    _full = _load_nemotron_sft(
        cache_dir=cache_dir, num_proc=num_proc, revision=revision)
    _n = len(_full)
    _split = max(_n - _NEMOTRON_VALID_SIZE, 1)
    if dataset_name == "nemotron-sft-train":
      dataset = _full.select(range(_split))
    else:
      dataset = _full.select(range(_split, _n))
    LOGGER.info(
        'Nemotron SFT %s size=%s (full=%s, splits=%s)',
        dataset_name, len(dataset), _n, _nemotron_split_list())
  elif dataset_name == "scientific_papers_arxiv":
    dataset = datasets.load_dataset(
      "scientific_papers", "arxiv",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "scientific_papers_pubmed":
    dataset = datasets.load_dataset(
      "scientific_papers", "pubmed",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "ag_news":
    dataset = datasets.load_dataset(
      "ag_news",
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "synthetic":
    assert streaming
    assert wrap
    dataset = generate_synthetic_dataset(
      train_dataset_size=100000,
      validation_dataset_size=1024,
      seq_len=block_size,
      vocab_size=len(tokenizer),
    )
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming,
      trust_remote_code=True,
      revision=revision)

  if dataset_name in [
      "lambada", "openwebtext-train", "openwebtext-valid",
      "alpaca-train", "alpaca-valid",
      "nemotron-sft-train", "nemotron-sft-valid",
  ]:
    data = dataset
  else:
    data = dataset[mode]
    if dataset_name == "synthetic":
      return data

  if dataset_name.startswith("wikitext"):
    detokenizer = wt_detokenizer
  elif dataset_name == "lm1b":
    detokenizer = lm1b_detokenizer
  elif dataset_name == "ptb":
    detokenizer = ptb_detokenizer
  elif dataset_name == "lambada":
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith("scientific_papers"):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  EOS = tokenizer.eos_token_id
  BOS = tokenizer.bos_token_id

  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  use_chunking = chunking_mode != "none"
  if use_chunking:
    if chunking_mode == "double_newline":
      delimiter_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    else:
      delimiter_tokens = []
    if not delimiter_tokens:
      raise ValueError(
        "Tokenizer did not produce any tokens for the specified chunking delimiter.")
  else:
    delimiter_tokens = []

  def preprocess_and_tokenize(example):
    if dataset_name == "ptb":
      text = example["sentence"]
    elif "scientific_papers" in dataset_name:
      text = example["article"]
    else:
      text = example["text"]
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)
    if use_chunking:
      return chunk_documents(
        tokenizer,
        text,
        max_length=block_size,
        delimiter_tokens=delimiter_tokens,
        add_special_tokens=insert_special_tokens)
    if wrap:
      tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False)
      if insert_eos:
        tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
    else:
      tokens = tokenizer(
        text,
        max_length=block_size,
        padding="max_length",
        truncation=True,
        add_special_tokens=insert_special_tokens,
        return_attention_mask=True,
        return_token_type_ids=True)
    return tokens

  map_kwargs = {
    "batched": True,
  }
  if use_chunking:
    map_kwargs["remove_columns"] = ["text"]
  if not streaming:
    map_kwargs.update(
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Tokenizing")
  tokenized_dataset = data.map(
    preprocess_and_tokenize,
    **map_kwargs)
  if dataset_name == "ptb":
    tokenized_dataset = tokenized_dataset.remove_columns("sentence")
  elif "scientific_papers" in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["article", "abstract", "section_names"])
  elif dataset_name == "ag_news":
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["text", "label"])
  elif "text" in tokenized_dataset.column_names:
    tokenized_dataset = tokenized_dataset.remove_columns("text")

  if (not wrap) and min_length > 0 and (not streaming):
    def _has_min_length(example):
      mask = example.get("attention_mask", None)
      if mask is None:
        return True
      return sum(mask) >= min_length

    tokenized_dataset = tokenized_dataset.filter(
      _has_min_length,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Filtering min length")

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format("torch")

  group_texts = functools.partial(
    _group_texts,
    block_size=block_size,
    bos=BOS,
    eos=EOS,
    insert_special_tokens=insert_special_tokens,
    attention_block_size=attention_block_size,
    pad_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0))
  if streaming:
    chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Grouping")
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format("torch")
  return chunked_dataset


def _finalize_tokenizer(tokenizer):
  """Ensure special tokens exist (shared by get_tokenizer / load_tokenizer_by_name)."""
  if isinstance(tokenizer, (transformers.GPT2TokenizerFast,
                            transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = (
      tokenizers.processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id)))
  if tokenizer.bos_token is None:
    # Qwen/chat models have no BOS; prefer the trained <|im_start|> id
    # over aliasing EOS (which made free-gen start at <|im_end|>).
    vocab = tokenizer.get_vocab()
    if '<|im_start|>' in vocab:
      tokenizer.bos_token = '<|im_start|>'
    elif tokenizer.cls_token is not None:
      tokenizer.bos_token = tokenizer.cls_token
    elif tokenizer.eos_token is not None:
      tokenizer.bos_token = tokenizer.eos_token
    else:
      raise AttributeError(
        "Tokenizer must have a bos_token or "
        f"cls_token: {tokenizer}")
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        "Tokenizer must have a eos_token "
        f"or sep_token: {tokenizer}")
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  if getattr(tokenizer, 'mask_token', None) is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
  return tokenizer


def load_tokenizer_by_name(name_or_path: str):
  """Load a tokenizer by HF id or local alias (text8, synthetic, …)."""
  if name_or_path == "text8":
    tokenizer = Text8Tokenizer()
  elif name_or_path == "bert-base-uncased":
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
  elif name_or_path == "synthetic":
    tokenizer = SyntheticTokenizer(vocab_size=256)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True)
  return _finalize_tokenizer(tokenizer)


def get_tokenizer(config):
  return load_tokenizer_by_name(config.data.tokenizer_name_or_path)


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  # Prefer distributed world size (Slurm/torchrun DDP: 1 visible GPU per
  # process). Fall back to local CUDA count for single-process multi-GPU.
  num_gpus = 1
  for key in ('WORLD_SIZE', 'SLURM_NTASKS', 'SLURM_NPROCS'):
    raw = os.environ.get(key)
    if raw is not None and str(raw).strip().isdigit() and int(raw) > 0:
      num_gpus = int(raw)
      break
  else:
    num_gpus = torch.cuda.device_count()
  if torch.cuda.device_count() < 1:
    raise RuntimeError(
        'No CUDA devices visible. Launch training on a GPU node (e.g. via Slurm).')
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches)), (
      f'global_batch_size={config.loader.global_batch_size} != '
      f'batch_size({config.loader.batch_size}) * nodes('
      f'{config.trainer.num_nodes}) * world({num_gpus}) * accum('
      f'{config.trainer.accumulate_grad_batches})'
  )
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f"Train Batch Size {config.loader.batch_size} "
      f"not divisible by {num_gpus} gpus with accumulation "
      f"{config.trainer.accumulate_grad_batches}.")
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f"Eval Global Batch Size {config.loader.eval_global_batch_size} "
      f"not divisible by {num_gpus}.")
  default_chunking = config.data.get("chunking", "none")
  train_chunking = config.data.get("train_chunking", default_chunking)
  valid_chunking = config.data.get("valid_chunking", default_chunking)
  if skip_train:
    train_set = None
  else:
    train_min_length = config.data.get(
      "train_min_length", config.data.get("min_length", 0))
    attn_bs = int(getattr(config.algo, 'block_size', 1) or 1)
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode="train",
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      insert_special_tokens=getattr(
        config.data, "insert_train_special", True),
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("train_revision", None),
      min_length=train_min_length,
      chunking=train_chunking,
      attention_block_size=attn_bs)

  if config.data.valid in ["text8", "lm1b", "ag_news"]:
    validation_split = "test"
  else:
    validation_split = "validation"
  if skip_valid:
    valid_set = None
  else:
    valid_min_length = config.data.get(
      "valid_min_length", config.data.get("min_length", 0))
    attn_bs = int(getattr(config.algo, 'block_size', 1) or 1)
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      insert_eos=config.data.insert_valid_eos,
      insert_special_tokens=getattr(
        config.data, "insert_valid_special", True),
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("valid_revision", None),
      min_length=valid_min_length,
      chunking=valid_chunking,
      attention_block_size=attn_bs)

  use_synthetic_collate = (
      config.data.train == 'synthetic' or config.data.valid == 'synthetic')
  collate_fn = _collate_tensor_dict if use_synthetic_collate else None

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.num_workers > 0,
      collate_fn=collate_fn)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=config.loader.num_workers > 0,
      collate_fn=collate_fn)
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader
