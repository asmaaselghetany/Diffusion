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
from datetime import timedelta
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
from discrete_diffusion.evaluations.block_qwen_eval_utils import (
    encode_context_continuation,
    generation_request_args,
    humaneval_until_for_chat,
    prepare_code_completion,
    require_masked_likelihood,
    truncate_at_stops,
)
from discrete_diffusion.evaluations.code_eval_patches import (
    patch_code_eval_metric_cache,
    patch_humaneval_chat_predictions,
)
from discrete_diffusion.evaluations.checkpoint_utils import load_block_trainer_checkpoint

try:
  import accelerate
  from accelerate.utils import InitProcessGroupKwargs
except ImportError:  # pragma: no cover
  accelerate = None
  InitProcessGroupKwargs = None  # type: ignore[misc, assignment]


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def _load_block_trainer(checkpoint_path: str, device: torch.device):
  """Load a UNI-D² Lightning BlockTrainer ckpt (same path as generate_samples)."""
  return load_block_trainer_checkpoint(checkpoint_path, device)


from discrete_diffusion.evaluations.decode_profiles import LM_EVAL_DECODE_PROFILES
from discrete_diffusion.data.conversion_baseline import (
    apply_conversion_chat_template,
    eval_max_seq_len,
    install_conversion_chat_template,
)


def _as_bool(v) -> bool:
  if isinstance(v, bool):
    return v
  if v is None:
    return False
  return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


def _is_none_token(v) -> bool:
  if v is None:
    return True
  return str(v).strip().lower() in ('', 'none', 'null')


# Shared ancestral core (all pipelines) vs paper overlays.
_LM_EVAL_DECODE_PROFILES = LM_EVAL_DECODE_PROFILES


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
      threshold: float | None = None,
      unmask_threshold: float | None = None,
      greedy: bool | None = None,
      hierarchical_kv: bool | None = None,
      use_block_cache: bool | None = None,
      single_stream_decode: bool | None = None,
      decode_profile: str | None = None,
      **kwargs,
  ) -> None:
    super().__init__()
    del kwargs
    if not checkpoint_path:
      raise ValueError('model_args must include checkpoint_path=...')
    # lm-eval may pass batch_size via CLI *and* model_args; prefer explicit.
    set_seed(int(seed))
    # Fast-dLLM eval.py: Accelerate shards requests across GPUs/nodes.
    # BlockSampler generate is slow (~2 tok/s here); rank skew after
    # generate_until easily exceeds the default NCCL watchdog (600s).
    # lm-eval then calls accelerator.wait_for_everyone() and dies with
    # ALLREDUCE timeout even though generations largely finished.
    accelerator = None
    if accelerate is not None:
      kwargs_handlers = []
      if InitProcessGroupKwargs is not None:
        timeout_s = int(os.environ.get('LM_EVAL_DIST_TIMEOUT_SEC', str(6 * 3600)))
        kwargs_handlers.append(
            InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_s)))
      accelerator = accelerate.Accelerator(
          kwargs_handlers=kwargs_handlers or None)
      if accelerator.num_processes <= 1:
        accelerator = None
    if accelerator is not None:
      self._device = accelerator.device
      self._rank = int(accelerator.process_index)
      self._world_size = int(accelerator.num_processes)
      self.accelerator = accelerator
    else:
      self._device = torch.device(
          device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
      self._rank = 0
      self._world_size = 1
      self.accelerator = None
    self.model, self.config, self.tokenizer = _load_block_trainer(
        checkpoint_path, self._device)
    # Conversion baseline: train↔eval ChatML (not stock Alibaba system).
    install_conversion_chat_template(self.tokenizer)
    self.batch_size = int(batch_size)
    self.max_new_tokens = int(max_new_tokens)
    self.num_steps = (
        int(num_steps) if num_steps is not None
        else int(getattr(self.config.sampling, 'steps', 32)))
    self.show_speed = _as_bool(show_speed)
    self.speed_metrics_path = speed_metrics_path
    self.checkpoint_path = str(checkpoint_path)

    profile = str(decode_profile or 'baseline').strip().lower()
    if profile not in _LM_EVAL_DECODE_PROFILES:
      raise ValueError(
          f'decode_profile={decode_profile!r} not in '
          f'{sorted(_LM_EVAL_DECODE_PROFILES)}')
    prof = dict(_LM_EVAL_DECODE_PROFILES[profile])

    # Profile defaults, then explicit model_args win.
    if hierarchical_kv is None and 'hierarchical_kv' in prof:
      hierarchical_kv = prof['hierarchical_kv']
    if use_block_cache is None and 'use_block_cache' in prof:
      use_block_cache = prof['use_block_cache']
    if single_stream_decode is None and 'single_stream_decode' in prof:
      single_stream_decode = prof['single_stream_decode']
    if greedy is None and 'greedy' in prof:
      greedy = prof['greedy']

    # Fast-dLLM eval.py uses model_args threshold=… (confidence unmask).
    thr = unmask_threshold if unmask_threshold is not None else threshold
    clear_thr = bool(prof.get('clear_unmask_threshold', False))
    if not _is_none_token(thr):
      self._force_unmask_threshold = float(thr)
      clear_thr = False
    elif clear_thr or profile == 'baseline':
      self._force_unmask_threshold = None
      clear_thr = True
    else:
      self._force_unmask_threshold = None
    self._force_greedy = None if greedy is None else _as_bool(greedy)
    self.mask_id = int(self.model.mask_id)
    self.block_size = int(self.model.block_size)
    self.train_seq_len = int(self.model.num_tokens)
    self.seq_len = self.train_seq_len
    self._eval_max_seq_len = eval_max_seq_len()
    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    self._sampler = self.model._create_sampler()
    self._decode_pins = {
        'hierarchical_kv': None if hierarchical_kv is None else _as_bool(
            hierarchical_kv),
        'use_block_cache': None if use_block_cache is None else _as_bool(
            use_block_cache),
        'single_stream_decode': (
            None if single_stream_decode is None
            else _as_bool(single_stream_decode)),
    }
    if self._sampler is not None:
      # Strip / apply paper overlays from the active profile.
      if 'use_arpc' in prof and hasattr(self._sampler, 'use_arpc'):
        self._sampler.use_arpc = bool(prof['use_arpc'])
      if 'sub_block_size' in prof and hasattr(self._sampler, 'sub_block_size'):
        self._sampler.sub_block_size = prof['sub_block_size']
      if 'ban_mask_pad_logits' in prof and hasattr(
          self._sampler, 'ban_mask_pad_logits'):
        self._sampler.ban_mask_pad_logits = bool(prof['ban_mask_pad_logits'])
      if clear_thr:
        self._sampler.unmask_threshold = None
      elif self._force_unmask_threshold is not None:
        self._sampler.unmask_threshold = self._force_unmask_threshold
      for attr, val in self._decode_pins.items():
        if val is not None:
          setattr(self._sampler, attr, val)
      # DualCache path needs hierarchical progressive windows.
      if (bool(getattr(self._sampler, 'use_block_cache', False))
          and not bool(getattr(self._sampler, 'hierarchical_kv', False))):
        self._sampler.hierarchical_kv = True
      if self._rank == 0:
        print(
            f'[decode_profile={profile}] '
            f'unmask_threshold={getattr(self._sampler, "unmask_threshold", None)} '
            f'hierarchical_kv={getattr(self._sampler, "hierarchical_kv", None)} '
            f'use_block_cache={getattr(self._sampler, "use_block_cache", None)} '
            f'single_stream={getattr(self._sampler, "single_stream_decode", None)} '
            f'use_arpc={getattr(self._sampler, "use_arpc", None)}',
            flush=True)
    self._is_causal_ar = self._sampler is None
    if self._is_causal_ar:
      mode = getattr(self.model.backbone, 'forward_mode', None)
      if mode != 'causal':
        raise RuntimeError(
            'Checkpoint has no BlockSampler and is not causal AR (C3).')

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
    return apply_conversion_chat_template(
        self.tokenizer,
        chat_history,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

  def loglikelihood_rolling(self, requests):
    raise NotImplementedError

  def _encode_pair(self, context: str, continuation: str):
    return encode_context_continuation(self.tokenizer, context, continuation)

  @torch.no_grad()
  def get_loglikelihood(self, prefix, target) -> float:
    """Hub ``eval.py`` one-token masked CE on the single-stream eval path.

    Matches Fast-dLLM v2 likelihood as closely as our stack allows:
    - mask only the first continuation token
    - pad to the next ``block_size`` multiple (not full train ``seq_len``)
    - write EOS on the first pad site
    - ``block_eval_logits`` (single-stream block-causal), not dual-stream clean ``x0``
    - always shift logits the Hub way
    - CE only on MASK sites (pad labels ``-100``)

    Pad-to-2048 dual-stream LL was an OOD meter (MASK ocean + clean answer
    stream) and is not comparable to Hub MMLU.
    """
    require_masked_likelihood(self.model.forward_process_name)
    if not target:
      return 0.0
    seq = torch.tensor(prefix + target, dtype=torch.long, device=self._device)
    if seq.numel() > self.seq_len:
      return -1e8
    prompt_len = len(prefix)
    content_len = int(seq.numel())
    bd = max(int(self.block_size), 1)
    # Hub: ``bd_size - (L % bd_size)`` — when L is already a multiple this
    # is ``bd_size`` (always pads 1..bd tokens), not zero.
    pad_len = bd - (content_len % bd)
    xt = seq.clone()
    xt[prompt_len] = self.mask_id
    pad = torch.full(
        (pad_len,), self.mask_id, dtype=torch.long, device=self._device)
    xt = torch.cat([xt, pad], dim=0)
    # Hub: first pad site is EOS (not MASK), so it is not scored.
    eos_id = getattr(self.tokenizer, 'eos_token_id', None)
    if eos_id is None:
      raise ValueError('tokenizer.eos_token_id required for Hub-matched LL')
    xt[content_len] = int(eos_id)
    labels = torch.cat([
        seq,
        torch.full((pad_len,), -100, dtype=torch.long, device=self._device),
    ], dim=0)
    backbone = getattr(self.model, 'backbone', None)
    eval_logits = getattr(backbone, 'block_eval_logits', None)
    if eval_logits is None:
      raise RuntimeError(
          'Hub-matched LL requires backbone.block_eval_logits '
          '(single-stream block-causal)')
    logits = eval_logits(
        xt.unsqueeze(0), active_len=int(xt.numel()), block_size=bd)
    # Hub always shifts for likelihood.
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    mask_indices = xt == self.mask_id
    if not mask_indices.any():
      return 0.0
    loss = F.cross_entropy(
        logits[0, mask_indices],
        labels[mask_indices],
        reduction='sum',
        ignore_index=-100)
    return float(-loss.item())

  def loglikelihood(self, requests):
    out = []
    for req in tqdm(requests, desc='block_qwen loglikelihood'):
      prefix, target = self._encode_pair(req.args[0], req.args[1])
      ll = self.get_loglikelihood(prefix, target)
      out.append((ll, 0.0))
      torch.cuda.empty_cache()
    return out

  def _parse_gen_kwargs(self, req) -> dict:
    if len(req.args) >= 2 and isinstance(req.args[1], dict):
      return req.args[1]
    return {}

  @staticmethod
  def _normalize_gen_kwargs(gen_kwargs: dict, default_max: int):
    max_gen = int(gen_kwargs.get('max_gen_toks', default_max))
    until = gen_kwargs.get('until', [])
    if until is None:
      until = []
    elif isinstance(until, str):
      until = [until]
    greedy = not bool(gen_kwargs.get('do_sample', False))
    return max_gen, list(until), greedy

  @staticmethod
  def _truncate_at_stop(text: str, until: list[str]) -> str:
    return truncate_at_stops(text, until)

  @torch.no_grad()
  def _generate_batch(
      self,
      questions: list[str],
      *,
      max_new_tokens: int | None = None,
      until: list[str] | None = None,
      greedy: bool = True,
  ) -> tuple[list[str], int, float]:
    max_new = int(max_new_tokens or self.max_new_tokens)
    until = until or []
    encoded = [
        self.tokenizer(q, add_special_tokens=False, return_tensors='pt')[
            'input_ids'][0]
        for q in questions
    ]
    # Hub eval.py: max_new=2048 with ~32k context. Train length is often 2048.
    # Extend the generation buffer (RoPE already supports long positions on
    # Qwen) up to EVAL_MAX_SEQ_LEN so we do not wipe prompts or clamp max_new
    # for typical GSM8K/IFEval prefixes. Only left-truncate if the prompt
    # itself exceeds eval_cap-1.
    #
    # Keep BlockTrainer.num_tokens and backbone.n_tokens in lockstep, then
    # restore after the batch so loglikelihood / later tasks do not inherit a
    # ratcheted buffer (dual-stream forward bounds active_len by n_tokens).
    bs = max(1, int(self.block_size))
    eval_cap = max(int(self.train_seq_len), int(self._eval_max_seq_len))
    eval_cap = (eval_cap // bs) * bs
    hard_cap = max(1, eval_cap - 1)
    encoded = [
        ids[-hard_cap:] if ids.numel() > hard_cap else ids
        for ids in encoded
    ]
    longest = max(int(ids.numel()) for ids in encoded)
    needed = longest + max_new
    needed = ((needed + bs - 1) // bs) * bs
    seq_len = max(int(self.train_seq_len), min(needed, eval_cap))
    prev_num_tokens = int(self.model.num_tokens)
    backbone = getattr(self.model, 'backbone', None)
    prev_n_tokens = (
        int(getattr(backbone, 'n_tokens', prev_num_tokens))
        if backbone is not None else prev_num_tokens)
    extended = seq_len != prev_num_tokens
    if extended:
      print(
          f'[block_qwen_lm_eval] extend seq_len '
          f'{prev_num_tokens}→{seq_len} '
          f'(train={self.train_seq_len}, eval_cap={eval_cap}, '
          f'longest_prefix={longest}, max_new={max_new})',
          flush=True,
      )
      self.model.num_tokens = seq_len
      self.seq_len = seq_len
      if backbone is not None and hasattr(backbone, 'n_tokens'):
        backbone.n_tokens = seq_len
    room = max(1, seq_len - longest)
    if max_new > room:
      print(
          f'[block_qwen_lm_eval] clamp max_new_tokens {max_new}→{room} '
          f'(seq_len={seq_len}, longest_prefix={longest}, '
          f'eval_cap={eval_cap})',
          flush=True,
      )
      max_new = room
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
    try:
      if self._device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(self._device)
      t0 = time.perf_counter()
      for i, plen in enumerate(lengths):
        prefix = prefix_batch[i:i + 1, :plen]
        if self._is_causal_ar:
          out = self.model.backbone.model.generate(
              prefix,
              max_new_tokens=max_new,
              do_sample=not greedy,
              top_p=float(getattr(self.config.sampling, 'p_nucleus', 0.9)),
              pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
              eos_token_id=self.tokenizer.eos_token_id,
          )
          cont = out[0, plen:plen + max_new]
        else:
          samples = self._sampler.generate(
              self.model,
              num_samples=1,
              num_steps=self.num_steps,
              eps=None,
              inject_bos=False,
              prefix_ids=prefix,
              max_new_tokens=max_new,
              greedy=greedy,
          )
          cont = samples[0, plen:plen + max_new]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
          eos_hits = (cont == eos).nonzero(as_tuple=False)
          if eos_hits.numel() > 0:
            cont = cont[: int(eos_hits[0]) + 1]
        # Count non-mask continuation tokens for tok/s.
        if self._is_causal_ar:
          n_tokens += int(cont.numel())
        else:
          n_tokens += int((cont != self.mask_id).sum().item())
        text = self.tokenizer.decode(cont, skip_special_tokens=True)
        text = self._truncate_at_stop(text, until)
        answers.append(text)
      if self._device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(self._device)
      elapsed = time.perf_counter() - t0
      return answers, n_tokens, elapsed
    finally:
      if extended:
        self.model.num_tokens = prev_num_tokens
        self.seq_len = prev_num_tokens
        if backbone is not None and hasattr(backbone, 'n_tokens'):
          backbone.n_tokens = prev_n_tokens

  def _write_speed_metrics(self) -> None:
    if self._rank != 0:
      return
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
        'unmask_threshold': getattr(
            self._sampler, 'unmask_threshold', None) if self._sampler else None,
        'hierarchical_kv': bool(getattr(
            self._sampler, 'hierarchical_kv', False)) if self._sampler else False,
        'use_block_cache': bool(getattr(
            self._sampler, 'use_block_cache', False)) if self._sampler else False,
        'single_stream_decode': bool(getattr(
            self._sampler, 'single_stream_decode', False)
        ) if self._sampler else False,
        'note': (
            'UNI-D2 BlockSampler tok/s during task generation. '
            'hierarchical_kv / DualCache are our ports — not Fast-dLLM '
            'fused CUDA kernels; do not claim paper throughput parity.'),
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
    # Group by generation kwargs (lm-eval contract).
    buckets: dict[tuple, list[tuple[int, object]]] = {}
    for idx, req in enumerate(requests):
      gen_kwargs = self._parse_gen_kwargs(req)
      until = list(gen_kwargs.get('until') or ())
      task = str(getattr(req, 'task_name', '') or '')
      # Chat-templated HumanEval must not use mid-function completion stops.
      if task.startswith('humaneval'):
        until = humaneval_until_for_chat(until)
      key = (
          int(gen_kwargs.get('max_gen_toks', self.max_new_tokens)),
          tuple(until),
          bool(gen_kwargs.get('do_sample', False)),
          task.split('_')[0],  # keep humaneval vs mbpp buckets separate
      )
      buckets.setdefault(key, []).append((idx, req))

    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    for key, batch in buckets.items():
      max_gen, until_tuple, do_sample, _task_prefix = key
      until = list(until_tuple)
      greedy = (
          self._force_greedy if self._force_greedy is not None
          else (not do_sample))
      batch.sort(key=lambda x: len(x[1].args[0]))
      for start in range(0, len(batch), self.batch_size):
        chunk = batch[start:start + self.batch_size]
        questions = []
        for _, req in chunk:
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
        answers, n_tok, elapsed = self._generate_batch(
            questions,
            max_new_tokens=max_gen,
            until=until,
            greedy=greedy,
        )
        self._speed_tokens += n_tok
        self._speed_elapsed += elapsed
        for (orig_idx, req), ans in zip(chunk, answers):
          task = str(getattr(req, 'task_name', '') or '')
          if task.startswith('humaneval') or task.startswith('mbpp'):
            ans = prepare_code_completion(ans, task)
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
  # Must run before lm_eval imports humaneval/mbpp utils (load code_eval).
  patch_code_eval_metric_cache()
  patch_humaneval_chat_predictions()
  cli_evaluate()
