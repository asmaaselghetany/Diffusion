"""Top-level loader API for discrete diffusion training."""

from __future__ import annotations

import functools
import os
import random
from typing import Optional

import datasets
import numpy as np
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
    _group_block_aligned_sft,
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

# Llama-Nemotron post-training SFT (Fast-dLLM-style data). Defaults + chat
# template live in conversion_baseline (shared masked + uniform hygiene).
from .conversion_baseline import (
    apply_conversion_chat_template,
    install_conversion_chat_template,
    maybe_keep_hub_vocab_size,
    nemotron_cache_fingerprint as _nemotron_cache_fingerprint,
)

_NEMOTRON_HUB = 'nvidia/Llama-Nemotron-Post-Training-Dataset'
_NEMOTRON_CONFIG = 'SFT'
_NEMOTRON_VALID_SIZE = 5000


def _default_num_proc() -> int:
  if hasattr(os, 'sched_getaffinity'):
    return len(os.sched_getaffinity(0))
  return os.cpu_count() or 1


def _nemotron_split_list() -> list[str]:
  from .conversion_baseline import nemotron_split_list
  return nemotron_split_list()


def _nemotron_max_for_split(split: str) -> int | None:
  from .conversion_baseline import nemotron_max_for_split
  return nemotron_max_for_split(split)


def _nemotron_to_messages(example: dict) -> list[dict[str, str]]:
  """Convert one Nemotron row into role-preserving Qwen chat messages."""
  messages: list[dict[str, str]] = []
  sys_p = (example.get('system_prompt') or '').strip()
  if sys_p:
    messages.append({'role': 'system', 'content': sys_p})

  msgs = example.get('input') or []
  if isinstance(msgs, list):
    for m in msgs:
      if isinstance(m, dict):
        role = (m.get('role') or 'user').lower()
        content = (m.get('content') or '').strip()
        if not content:
          continue
        if role not in {'system', 'user', 'assistant'}:
          raise ValueError(f'Unsupported Nemotron chat role: {role!r}')
        messages.append({'role': role, 'content': content})
      else:
        content = str(m).strip()
        if content:
          messages.append({'role': 'user', 'content': content})
  elif msgs:
    messages.append({'role': 'user', 'content': str(msgs).strip()})

  output = (example.get('output') or '').strip()
  messages.append({'role': 'assistant', 'content': output})
  return messages


def _tokenize_nemotron_sft_batch(examples: dict, tokenizer) -> dict:
  """Apply Qwen ChatML and retain labels only for assistant tokens."""
  size = len(examples['output'])
  result = {'input_ids': [], 'attention_mask': [], 'labels': []}
  for idx in range(size):
    row = {key: values[idx] for key, values in examples.items()}
    encoded = apply_conversion_chat_template(
        tokenizer,
        _nemotron_to_messages(row),
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    input_ids = list(encoded['input_ids'])
    assistant_mask = list(encoded['assistant_masks'])
    if len(input_ids) != len(assistant_mask):
      raise ValueError('Qwen assistant-token mask does not align with input_ids')
    if not any(assistant_mask):
      raise ValueError('Qwen chat template produced no supervised assistant tokens')
    result['input_ids'].append(input_ids)
    result['attention_mask'].append([1] * len(input_ids))
    result['labels'].append([
        token_id if is_assistant else -100
        for token_id, is_assistant in zip(input_ids, assistant_mask)
    ])
  return result


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
  return full


__all__ = [
    "get_tokenizer",
    "get_dataset",
    "get_dataloaders",
]



def _with_race_safe_cache(cache_dir, path, streaming, build):
  """Load, build under lock, or atomically publish a processed dataset cache."""
  from . import dataset_cache as dc

  lock_path = f'{path}.lockdir'
  cached = dc._load_complete_dataset_cache(path)
  if cached is not None:
    LOGGER.info('Loading data from: %s', path)
    return cached.with_format('torch')
  os.makedirs(cache_dir, exist_ok=True)
  with dc._exclusive_dataset_cache_lock(lock_path) as lock_token:
    cached = dc._load_complete_dataset_cache(path)
    if cached is not None:
      LOGGER.info('Loading data from: %s (after cache lock)', path)
      return cached.with_format('torch')
    dc._recover_legacy_flock_file(f'{path}.lock', path)
    if os.path.lexists(path):
      dc._recover_stale_incomplete_cache(path)
    LOGGER.info('Generating new data at: %s', path)
    built = build()
    if streaming:
      return built.with_format('torch')
    return dc._save_dataset_cache_atomically(
        built, path, lock_path, lock_token).with_format('torch')


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
                attention_block_size: int = 1,
                diffusion_block_size: Optional[int] = None):
  chunking_mode = (chunking or "none").lower()
  if chunking_mode not in {"none", "double_newline"}:
    raise ValueError(f"Unsupported chunking mode: {chunking_mode}")
  if wrap and chunking_mode != "none":
    raise ValueError("Delimiter-based chunking only applies when wrap=False.")
  if diffusion_block_size is None:
    diffusion_block_size = attention_block_size
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
  sft_tag = ""
  if dataset_name in ("nemotron-sft-train", "nemotron-sft-valid"):
    fingerprint = _nemotron_cache_fingerprint(
        tokenizer, revision, diffusion_block_size)
    sft_tag = f"_qwenchat_d{diffusion_block_size}_{fingerprint}"
  if wrap:
    filename = (
        f"{dataset_name}_{mode}_bs{block_size}_wrapped{align_tag}{sft_tag}"
        f"{eos_tag}.dat")
  else:
    filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped{chunk_tag}{eos_tag}{min_len_tag}.dat"
  _path = os.path.join(cache_dir, filename)

  def _build():
    nonlocal block_size
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
    elif dataset_name == "openwebtext-blockgen-train":
      # BlockGen OWT split (jdeschena/blockgen dataloader.py).
      dataset = datasets.load_dataset(
        "jdeschena/openwebtext",
        split="train[:-100000]",
        cache_dir=cache_dir,
        revision=revision,
        streaming=False,
        num_proc=num_proc)
    elif dataset_name == "openwebtext-blockgen-valid":
      dataset = datasets.load_dataset(
        "jdeschena/openwebtext",
        split="train[-100000:]",
        cache_dir=cache_dir,
        revision=revision,
        streaming=False,
        num_proc=num_proc)
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
        "openwebtext-blockgen-train", "openwebtext-blockgen-valid",
        "alpaca-train", "alpaca-valid",
        "nemotron-sft-train", "nemotron-sft-valid",
    ]:
      data = dataset
    else:
      data = dataset[mode]
      if dataset_name == "synthetic":
        return data

    if dataset_name in ("nemotron-sft-train", "nemotron-sft-valid"):
      if streaming:
        raise ValueError('Block-aligned Nemotron SFT does not support streaming')
      if block_size % diffusion_block_size != 0:
        raise ValueError(
            f'model.length={block_size} must be divisible by '
            f'diffusion_block_size={diffusion_block_size}')
      tokenized_dataset = data.map(
          functools.partial(_tokenize_nemotron_sft_batch, tokenizer=tokenizer),
          batched=True,
          remove_columns=data.column_names,
          num_proc=num_proc,
          load_from_cache_file=True,
          desc='Applying Qwen SFT chat template')
      group_sft = functools.partial(
          _group_block_aligned_sft,
          sequence_length=block_size,
          diffusion_block_size=diffusion_block_size,
          mask_id=tokenizer.mask_token_id)
      chunked_dataset = tokenized_dataset.map(
          group_sft,
          batched=True,
          num_proc=num_proc,
          load_from_cache_file=True,
          desc='Block-aligning and packing SFT')
      return chunked_dataset

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
      return tokenized_dataset

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
    return chunked_dataset



  return _with_race_safe_cache(cache_dir, _path, streaming, _build)

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
  # Hub-like: keep padded HF vocab (e.g. 151936) after MASK add.
  maybe_keep_hub_vocab_size(tokenizer)
  # Train↔eval: install conversion ChatML (short system prompt).
  install_conversion_chat_template(tokenizer)
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


def _training_world_size(config) -> int:
  """Total DDP processes (devices × nodes). Prefer config; fall back to env.

  Do NOT multiply ``WORLD_SIZE`` by ``trainer.num_nodes`` — under multi-node
  ``srun`` ``WORLD_SIZE`` is already the full world and that double-counts.
  """
  devices = int(getattr(config.trainer, 'devices', 1) or 1)
  nodes = int(getattr(config.trainer, 'num_nodes', 1) or 1)
  if devices * nodes > 1:
    return devices * nodes
  for key in ('WORLD_SIZE', 'SLURM_NTASKS', 'SLURM_NPROCS'):
    raw = os.environ.get(key)
    if raw is not None and str(raw).strip().isdigit() and int(raw) > 0:
      return int(raw)
  return max(torch.cuda.device_count(), 1)


def _dataloader_worker_init_fn(worker_id: int) -> None:
  """Seed numpy/random per worker (pairs with ``seed_everything(..., workers=True)``)."""
  del worker_id
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def _dataloader_generator(seed: int | None) -> torch.Generator | None:
  if seed is None:
    return None
  gen = torch.Generator()
  gen.manual_seed(int(seed) % (2**32))
  return gen


def _ensure_file_system_tensor_sharing() -> None:
  """Prefer file-backed tensor IPC over /dev/shm for DataLoader workers.

  Multi-node Jupiter paper jobs (8×4 GPUs × N workers) routinely hit
  ``Unexpected bus error … insufficient shared memory (shm)`` under the
  default ``file_descriptor`` strategy when /dev/shm is tight. Unifusion /
  Unif(V) ablations are not special — any workered loader can trip this.
  """
  try:
    torch.multiprocessing.set_sharing_strategy('file_system')
  except (RuntimeError, AttributeError, ValueError):
    pass


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  # Total DDP world size. Matches config.yaml accumulate_grad_batches resolver:
  #   GBS == batch_size * devices * num_nodes * accumulate_grad_batches
  num_gpus = _training_world_size(config)
  if int(getattr(config.loader, 'num_workers', 0) or 0) > 0:
    _ensure_file_system_tensor_sharing()
  if torch.cuda.device_count() < 1:
    raise RuntimeError(
        'No CUDA devices visible. Launch training on a GPU node (e.g. via Slurm).')
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * num_gpus
              * config.trainer.accumulate_grad_batches)), (
      f'global_batch_size={config.loader.global_batch_size} != '
      f'batch_size({config.loader.batch_size}) * world({num_gpus}) * accum('
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
      attention_block_size=attn_bs,
      diffusion_block_size=attn_bs)

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
      attention_block_size=attn_bs,
      diffusion_block_size=attn_bs)

  use_synthetic_collate = (
      config.data.train == 'synthetic' or config.data.valid == 'synthetic')
  collate_fn = _collate_tensor_dict if use_synthetic_collate else None
  base_seed = int(getattr(config, 'seed', 0) or 0)
  worker_init = (
      _dataloader_worker_init_fn if int(config.loader.num_workers) > 0 else None)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      generator=_dataloader_generator(base_seed),
      worker_init_fn=worker_init,
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
      generator = _dataloader_generator(int(valid_seed))
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      worker_init_fn=worker_init if shuffle_valid else None,
      persistent_workers=config.loader.num_workers > 0,
      collate_fn=collate_fn)
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader
