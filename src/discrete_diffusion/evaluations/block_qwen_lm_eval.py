# Copyright adaptations: Fast-dLLM v2 eval harness (Apache-2.0, NVlabs) reshaped
# to UNI-D² layout — load Lightning ``BlockTrainer`` checkpoints and
# ``BlockSampler`` instead of HuggingFace Fast_dLLM_v2 weights.
#
# Upstream reference:
#   https://github.com/NVlabs/Fast-dLLM/blob/main/v2/eval.py
#   https://github.com/NVlabs/Fast-dLLM/blob/main/v2/eval_script.sh

"""lm-evaluation-harness wrapper for Qwen ``BlockTrainer`` checkpoints.

Register model name ``block_qwen``. Task suite mirrors Fast-dLLM v2
(``eval_script.sh``): MMLU, GPQA, GSM8K, Minerva Math, IFEval, plus
optional HumanEval for code accuracy.

Also tracks decode **tok/s** during ``generate_until`` (our BlockSampler;
not Fast-dLLM hierarchical KV).

Example::

  PYTHONPATH=src python -m discrete_diffusion.evaluations.block_qwen_lm_eval \\
    --tasks gsm8k --batch_size 1 --num_fewshot 0 \\
    --model block_qwen --fewshot_as_multiturn --apply_chat_template \\
    --confirm_run_unsafe_code \\
    --model_args checkpoint_path=...,speed_metrics_path=./tok_s_lm_eval.json
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import hydra.utils
import numpy as np
import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from omegaconf import OmegaConf
from tqdm import tqdm

from discrete_diffusion.data import get_tokenizer


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def _load_block_trainer(checkpoint_path: str, device: torch.device):
  """Load a UNI-D² Lightning BlockTrainer ckpt (same path as generate_samples)."""
  path = Path(checkpoint_path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(f'checkpoint not found: {path}')
  ckpt = torch.load(path, map_location='cpu', weights_only=False)
  if 'hyper_parameters' not in ckpt or 'config' not in ckpt['hyper_parameters']:
    raise ValueError(f'{path} missing hyper_parameters.config')
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)
  tokenizer = get_tokenizer(config)
  algo_target = config.algo._target_
  algo_cls = hydra.utils.get_class(algo_target)
  model = algo_cls.load_from_checkpoint(
      str(path),
      config=config,
      tokenizer=tokenizer,
      map_location=device,
  )
  model.to(device)
  model.eval()
  return model, config, tokenizer


def _as_bool(v) -> bool:
  if isinstance(v, bool):
    return v
  if v is None:
    return False
  return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


@register_model('block_qwen')
class BlockQwenEvalHarness(LM):
  """Fast-dLLM-style lm-eval surface for UNI-D² block_qwen checkpoints."""

  def __init__(
      self,
      checkpoint_path: str | None = None,
      device: str = 'cuda',
      batch_size: int = 1,
      max_new_tokens: int = 512,
      num_steps: int | None = None,
      show_speed: bool = True,
      speed_metrics_path: str | None = None,
      seed: int = 0,
      **kwargs,
  ) -> None:
    super().__init__()
    del kwargs
    if not checkpoint_path:
      raise ValueError('model_args must include checkpoint_path=...')
    # lm-eval may pass batch_size via CLI *and* model_args; prefer explicit.
    set_seed(int(seed))
    self._device = torch.device(
        device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    self.model, self.config, self.tokenizer = _load_block_trainer(
        checkpoint_path, self._device)
    self.batch_size = int(batch_size)
    self.max_new_tokens = int(max_new_tokens)
    self.num_steps = (
        int(num_steps) if num_steps is not None
        else int(getattr(self.config.sampling, 'steps', 32)))
    self.show_speed = _as_bool(show_speed)
    self.speed_metrics_path = speed_metrics_path
    self.checkpoint_path = str(checkpoint_path)
    self.mask_id = int(self.model.mask_id)
    self.block_size = int(self.model.block_size)
    self.seq_len = int(self.model.num_tokens)
    self._rank = 0
    self._world_size = 1
    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    self._sampler = self.model._create_sampler()
    if self._sampler is None:
      raise RuntimeError('BlockTrainer has no configured BlockSampler')

  @property
  def device(self):
    return self._device

  @property
  def rank(self):
    return self._rank

  @property
  def world_size(self):
    return self._world_size

  @property
  def tokenizer_name(self):
    return str(getattr(self.config.data, 'tokenizer_name_or_path', 'qwen'))

  def apply_chat_template(self, chat_history, add_generation_prompt=True):
    return self.tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

  def loglikelihood_rolling(self, requests):
    raise NotImplementedError

  def _encode_pair(self, context: str, continuation: str):
    whole = self.tokenizer(context + continuation, add_special_tokens=False)[
        'input_ids']
    ctx = self.tokenizer(context, add_special_tokens=False)['input_ids']
    return ctx, whole[len(ctx):]

  @torch.no_grad()
  def get_loglikelihood(self, prefix, target) -> float:
    """One-shot masked CE on the continuation (Fast-dLLM-style approximation)."""
    seq = torch.tensor(prefix + target, dtype=torch.long, device=self._device)
    if seq.numel() > self.seq_len:
      return -1e8
    # Pad to full model length with masks / ignore.
    pad_len = self.seq_len - seq.numel()
    if pad_len > 0:
      pad = torch.full(
          (pad_len,), self.mask_id, dtype=torch.long, device=self._device)
      clean = torch.cat([seq, pad], dim=0)
    else:
      clean = seq
    xt = clean.clone()
    # Mask the continuation region (and pads).
    start = len(prefix)
    xt[start:] = self.mask_id
    xt = xt.unsqueeze(0)
    x0 = clean.unsqueeze(0)
    logits = self.model.backbone_logits(xt, x0)
    # Optional shift alignment used in Fast-dLLM eval.
    if bool(getattr(self.config.algo, 'shift_loss_targets', False)):
      logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    mask = torch.zeros_like(clean, dtype=torch.bool)
    mask[start:start + len(target)] = True
    if not mask.any():
      return 0.0
    loss = F.cross_entropy(
        logits[0, mask], clean[mask], reduction='sum')
    return float(-loss.item())

  def loglikelihood(self, requests):
    out = []
    for req in tqdm(requests, desc='block_qwen loglikelihood'):
      prefix, target = self._encode_pair(req.args[0], req.args[1])
      ll = self.get_loglikelihood(prefix, target)
      out.append((ll, 0.0))
      torch.cuda.empty_cache()
    return out

  @torch.no_grad()
  def _generate_batch(self, questions: list[str]) -> tuple[list[str], int, float]:
    encoded = [
        self.tokenizer(q, add_special_tokens=False, return_tensors='pt')[
            'input_ids'][0]
        for q in questions
    ]
    # Truncate prompts so at least max_new_tokens (or 1) remain.
    max_prefix = max(1, self.seq_len - max(self.max_new_tokens, 1))
    encoded = [ids[:max_prefix] for ids in encoded]
    max_len = max(int(ids.numel()) for ids in encoded)
    padded = []
    lengths = []
    for ids in encoded:
      lengths.append(int(ids.numel()))
      if ids.numel() < max_len:
        pad = torch.full(
            (max_len - ids.numel(),), self.mask_id, dtype=torch.long)
        ids = torch.cat([ids, pad], dim=0)
      padded.append(ids)
    prefix_batch = torch.stack(padded, dim=0).to(self._device)
    answers = []
    n_tokens = 0
    if self._device.type == 'cuda' and torch.cuda.is_available():
      torch.cuda.synchronize(self._device)
    t0 = time.perf_counter()
    for i, plen in enumerate(lengths):
      prefix = prefix_batch[i:i + 1, :plen]
      samples = self._sampler.generate(
          self.model,
          num_samples=1,
          num_steps=self.num_steps,
          eps=None,
          inject_bos=False,
          prefix_ids=prefix,
          max_new_tokens=self.max_new_tokens,
      )
      cont = samples[0, plen:plen + self.max_new_tokens]
      eos = self.tokenizer.eos_token_id
      if eos is not None:
        eos_hits = (cont == eos).nonzero(as_tuple=False)
        if eos_hits.numel() > 0:
          cont = cont[: int(eos_hits[0]) + 1]
      # Count non-mask continuation tokens for tok/s.
      n_tokens += int((cont != self.mask_id).sum().item())
      text = self.tokenizer.decode(cont, skip_special_tokens=True)
      answers.append(text)
    if self._device.type == 'cuda' and torch.cuda.is_available():
      torch.cuda.synchronize(self._device)
    elapsed = time.perf_counter() - t0
    return answers, n_tokens, elapsed

  def _write_speed_metrics(self) -> None:
    if not self.show_speed and not self.speed_metrics_path:
      return
    elapsed = max(self._speed_elapsed, 1e-9)
    tok_s = self._speed_tokens / elapsed
    metrics = {
        'checkpoint_path': self.checkpoint_path,
        'device': str(self._device),
        'source': 'lm_eval.generate_until',
        'tokens_generated': int(self._speed_tokens),
        'elapsed_s': float(self._speed_elapsed),
        'tok_s': float(tok_s),
        'num_steps': self.num_steps,
        'batch_size': self.batch_size,
        'note': (
            'UNI-D2 BlockSampler tok/s during task generation '
            '(no Fast-dLLM hierarchical KV / sub-block parallel).'),
    }
    print(
        f"[tok/s] tokens={metrics['tokens_generated']} "
        f"time={metrics['elapsed_s']:.2f}s tok/s={metrics['tok_s']:.2f}",
        flush=True)
    if self.speed_metrics_path:
      path = Path(self.speed_metrics_path)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
      print(f'Wrote speed metrics: {path}', flush=True)

  def generate_until(self, requests):
    output = [None] * len(requests)
    indexed = list(enumerate(requests))
    indexed.sort(key=lambda x: len(x[1].args[0]))

    batches: list[list[tuple[int, object]]] = []
    cur: list[tuple[int, object]] = []
    for item in indexed:
      cur.append(item)
      if len(cur) >= self.batch_size:
        batches.append(cur)
        cur = []
    if cur:
      batches.append(cur)

    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    for batch in tqdm(batches, desc='block_qwen generate_until'):
      questions = []
      for _, req in batch:
        q = req.args[0]
        if req.task_name.startswith('minerva_math'):
          q = q.replace(
              'Solution:',
              'Please reason step by step, and put your final answer '
              'within \\boxed{}.')
        elif req.task_name.startswith('gsm8k'):
          q = q.replace(
              'Answer:',
              'Please reason step by step, and put your final answer '
              'within \\boxed{}.')
        questions.append(q)
      answers, n_tok, elapsed = self._generate_batch(questions)
      self._speed_tokens += n_tok
      self._speed_elapsed += elapsed
      for (orig_idx, req), ans in zip(batch, answers):
        output[orig_idx] = ans
        print('=' * 20)
        print('question:', req.args[0][:200])
        print('answer:', ans[:500])
        print('=' * 20, end='\n\n')
      torch.cuda.empty_cache()

    self._write_speed_metrics()
    return output


if __name__ == '__main__':
  # Allow code-eval datasets (HumanEval) when users pass confirm flags.
  os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
  os.environ.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', 'true')
  cli_evaluate()
