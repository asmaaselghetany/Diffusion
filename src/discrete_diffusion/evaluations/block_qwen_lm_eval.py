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

``--confirm_run_unsafe_code`` is required for HumanEval/MBPP: it both
satisfies lm-eval's harness gate and enables ``HF_ALLOW_CODE_EVAL`` /
``HF_DATASETS_TRUST_REMOTE_CODE`` in this entrypoint.

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
    cap_max_new_for_task,
    encode_context_continuation,
    generation_request_args,
    humaneval_until_for_chat,
    prepare_code_completion,
    require_masked_likelihood,
    temporary_model_seq_len,
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
    # Refuse the 45-line utils.py stub before loading weights (Unif helpers).
    from discrete_diffusion.evaluations.code_fingerprint import (
        assert_forward_process_utils_ok,
        code_fingerprint_header,
    )
    self._code_fingerprint = code_fingerprint_header()
    assert_forward_process_utils_ok(require_expected_sha=False)
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

    from discrete_diffusion.evaluations.decode_profiles import (
        allow_full_seq_decode_requested,
        coerce_profile_for_forward,
    )
    requested_profile = str(decode_profile or 'baseline').strip().lower()
    fp_name = str(
        getattr(self.model, 'forward_process_name', '') or '').lower()
    allow_fs = allow_full_seq_decode_requested()
    profile = coerce_profile_for_forward(
        requested_profile, fp_name, allow_full_seq=allow_fs)
    if profile != requested_profile and self._rank == 0:
      print(
          f'[block_qwen_lm_eval] coerce decode_profile '
          f'{requested_profile!r}→{profile!r} for forward={fp_name!r} '
          f'(refuse full-seq Unif attention)',
          flush=True)
    if profile not in _LM_EVAL_DECODE_PROFILES:
      raise ValueError(
          f'decode_profile={decode_profile!r} not in '
          f'{sorted(_LM_EVAL_DECODE_PROFILES)}')
    prof = dict(_LM_EVAL_DECODE_PROFILES[profile])
    # Uniform: never take profile greedy=true (illegal reverse).
    if (fp_name in ('uniform', 'hybrid') and prof.get('greedy') is True):
      prof['greedy'] = False

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
    # Explicit model_args threshold (including null/none) overrides profile pins.
    # Absent threshold → fall through to profile / clear_unmask_threshold.
    thr_arg_set = (unmask_threshold is not None) or (threshold is not None)
    thr = unmask_threshold if unmask_threshold is not None else threshold
    clear_thr = bool(prof.get('clear_unmask_threshold', False))
    if thr_arg_set and _is_none_token(thr):
      self._force_unmask_threshold = None
      clear_thr = True
    elif not _is_none_token(thr):
      self._force_unmask_threshold = float(thr)
      clear_thr = False
    elif clear_thr or profile == 'baseline':
      self._force_unmask_threshold = None
      clear_thr = True
    else:
      self._force_unmask_threshold = None
    self._force_greedy = None if greedy is None else _as_bool(greedy)
    self.mask_id = int(self.model.mask_id)
    # ArSftTrainer has no block geometry; config.block_size is symmetry-only.
    bs = getattr(self.model, 'block_size', None)
    if bs is None:
      bs = getattr(self.config, 'block_size', None)
    self.block_size = int(bs) if bs not in (None, '', 'null') else 1
    self.train_seq_len = int(self.model.num_tokens)
    self.seq_len = self.train_seq_len
    self._eval_max_seq_len = eval_max_seq_len()
    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    self._nfe_per_sample: list[dict] = []
    # C3 / ar_sft still hydrates AbsorbingSampler from sampling.sampler in
    # the shared config tree — do NOT treat "sampler is not None" as block
    # diffusion. Detect causal AR by trainer type / algo name.
    from discrete_diffusion.algorithms.ar_sft import ArSftTrainer
    algo_name = str(getattr(self.config.algo, 'name', '') or '')
    self._is_causal_ar = (
        isinstance(self.model, ArSftTrainer) or algo_name == 'ar_sft')
    if self._is_causal_ar:
      mode = getattr(self.model.backbone, 'forward_mode', None)
      if mode != 'causal':
        raise RuntimeError(
            f'C3/ar_sft requires model.forward_mode=causal, got {mode!r}')
      self._sampler = None
      self.block_size = 1
      self._decode_pins = {
          'hierarchical_kv': None,
          'use_block_cache': None,
          'single_stream_decode': None,
      }
      if self._rank == 0:
        print(
            f'[decode_profile={profile}] causal_ar=True '
            f'(HF generate; ignoring block sampler overlays)',
            flush=True)
    else:
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
        for _arpc_key in (
            'arpc_mode', 'arpc_corruption_mode', 'arpc_ar_metric',
            'arpc_temperature', 'x0_temperature'):
          if _arpc_key in prof and hasattr(self._sampler, _arpc_key):
            setattr(self._sampler, _arpc_key, prof[_arpc_key])
        if 'arpc_use_prefix_fill' in prof and hasattr(
            self._sampler, 'arpc_use_prefix_fill'):
          self._sampler.arpc_use_prefix_fill = bool(
              prof['arpc_use_prefix_fill'])
        if 'sub_block_size' in prof and hasattr(self._sampler, 'sub_block_size'):
          self._sampler.sub_block_size = prof['sub_block_size']
        if 'ban_mask_pad_logits' in prof and hasattr(
            self._sampler, 'ban_mask_pad_logits'):
          self._sampler.ban_mask_pad_logits = bool(prof['ban_mask_pad_logits'])
        if 'posterior_sampler' in prof and hasattr(
            self._sampler, 'posterior_sampler'):
          self._sampler.posterior_sampler = str(prof['posterior_sampler'])
        if 'uniform_confidence_sticky' in prof and hasattr(
            self._sampler, 'uniform_confidence_sticky'):
          self._sampler.uniform_confidence_sticky = bool(
              prof['uniform_confidence_sticky'])
        if 'sticky_min_conf' in prof and hasattr(
            self._sampler, 'sticky_min_conf'):
          self._sampler.sticky_min_conf = float(prof['sticky_min_conf'])
        if 'uniform_commit_revise' in prof and hasattr(
            self._sampler, 'uniform_commit_revise'):
          self._sampler.uniform_commit_revise = bool(
              prof['uniform_commit_revise'])
        if 'uniform_commit_random' in prof and hasattr(
            self._sampler, 'uniform_commit_random'):
          self._sampler.uniform_commit_random = bool(
              prof['uniform_commit_random'])
          if self._sampler.uniform_commit_random:
            self._sampler.uniform_commit_order = 'random'
        if 'uniform_commit_order' in prof and hasattr(
            self._sampler, 'uniform_commit_order'):
          self._sampler.uniform_commit_order = str(
              prof['uniform_commit_order'] or 'confidence')
        if 'uniform_commit_revise_tau' in prof and hasattr(
            self._sampler, 'uniform_commit_revise_tau'):
          self._sampler.uniform_commit_revise_tau = float(
              prof['uniform_commit_revise_tau'])
        if 'allow_full_seq_decode' in prof and hasattr(
            self._sampler, 'allow_full_seq_decode'):
          self._sampler.allow_full_seq_decode = bool(
              prof['allow_full_seq_decode'])
        if clear_thr:
          self._sampler.unmask_threshold = None
        elif self._force_unmask_threshold is not None:
          self._sampler.unmask_threshold = self._force_unmask_threshold
        elif 'unmask_threshold' in prof and prof['unmask_threshold'] is not None:
          self._sampler.unmask_threshold = float(prof['unmask_threshold'])
        for attr, val in self._decode_pins.items():
          if val is not None:
            setattr(self._sampler, attr, val)
        # DualCache path needs hierarchical progressive windows.
        if (bool(getattr(self._sampler, 'use_block_cache', False))
            and not bool(getattr(self._sampler, 'hierarchical_kv', False))):
          self._sampler.hierarchical_kv = True
        # Refuse full-seq dual open-loop (attend future noise). Truncation
        # only — do not force Hub single_stream onto BlockGen dual decode.
        fp_name = str(
            getattr(self.model, 'forward_process_name', '') or '').lower()
        _allow_fs = bool(getattr(
            self._sampler, 'allow_full_seq_decode', False)) or allow_fs
        _open_loop = (
            not _allow_fs
            and not bool(getattr(self._sampler, 'use_block_cache', False))
            and getattr(self._sampler, 'unmask_threshold', None) is None
            and not bool(getattr(
                self._sampler, 'uniform_confidence_sticky', False)))
        if (fp_name in ('masked', 'uniform', 'hybrid')
            and _open_loop
            and not bool(getattr(self._sampler, 'hierarchical_kv', False))):
          print(
              f'[block_qwen_lm_eval] {fp_name}: forcing hierarchical_kv '
              '(refuse full-seq dual open-loop; C0 ancestral was 2.27% GSM).',
              flush=True)
          self._sampler.hierarchical_kv = True
        # Hard fail: never enter generate_until with full-seq open-loop.
        if (fp_name in ('masked', 'uniform', 'hybrid')
            and self._sampler is not None
            and _open_loop
            and not bool(getattr(self._sampler, 'hierarchical_kv', False))):
          raise RuntimeError(
              f'REFUSED: {fp_name} open-loop without hierarchical_kv '
              '(full-seq dual packing). Pass decode_profile=baseline '
              '(→ hierarchical) or hierarchical. '
              'C0 ancestral GSM was 2.27% on the refused path.')
        if (fp_name in ('uniform', 'hybrid')
            and self._force_greedy is True):
          print(
              '[block_qwen_lm_eval] uniform: forcing greedy=false '
              '(argmax q_xs locks prior)',
              flush=True)
          self._force_greedy = False
        if self._rank == 0:
          print(
              f'[decode_profile={profile}] '
              f'unmask_threshold={getattr(self._sampler, "unmask_threshold", None)} '
              f'hierarchical_kv={getattr(self._sampler, "hierarchical_kv", None)} '
              f'use_block_cache={getattr(self._sampler, "use_block_cache", None)} '
              f'single_stream={getattr(self._sampler, "single_stream_decode", None)} '
              f'use_arpc={getattr(self._sampler, "use_arpc", None)} '
              f'x0_temperature={getattr(self._sampler, "x0_temperature", None)} '
              f'posterior_sampler={getattr(self._sampler, "posterior_sampler", None)} '
              f'sticky={getattr(self._sampler, "uniform_confidence_sticky", None)} '
              f'sticky_min_conf={getattr(self._sampler, "sticky_min_conf", None)} '
              f'revise={getattr(self._sampler, "uniform_commit_revise", None)} '
              f'allow_full_seq={getattr(self._sampler, "allow_full_seq_decode", None)}',
              flush=True)

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
  def _causal_ar_loglikelihood(self, prefix, target) -> float:
    """Teacher-forced causal NLL for matched AR SFT (C3 / paper_acc MMLU)."""
    if not target:
      return 0.0
    if not prefix:
      raise ValueError('causal AR loglikelihood needs a non-empty prefix')
    seq = torch.tensor(prefix + target, dtype=torch.long, device=self._device)
    if seq.numel() > self.seq_len:
      return -1e8
    logits = self.model.backbone(seq.unsqueeze(0), sigma=None)
    # Position i predicts token i+1; score target under prefix context.
    plen = len(prefix)
    shift = logits[0, plen - 1:plen - 1 + len(target), :]
    log_probs = F.log_softmax(shift.float(), dim=-1)
    tgt = torch.tensor(target, dtype=torch.long, device=self._device)
    return float(log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item())

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

    C3 / ``ar_sft`` uses standard causal teacher-forced NLL instead.
    """
    if self._is_causal_ar:
      return self._causal_ar_loglikelihood(prefix, target)
    fp = getattr(self.model, 'forward_process_name', None)
    require_masked_likelihood(fp if fp is not None else 'none')
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
    with temporary_model_seq_len(
        self.model, seq_len, seq_len_holder=self) as seq_ctx:
      if seq_ctx.extended:
        print(
            f'[block_qwen_lm_eval] extend seq_len '
            f'{seq_ctx.prev_num_tokens}→{seq_len} '
            f'(train={self.train_seq_len}, eval_cap={eval_cap}, '
            f'longest_prefix={longest}, max_new={max_new})',
            flush=True,
        )
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
          stats = getattr(self._sampler, 'last_nfe_stats', None)
          if isinstance(stats, dict):
            self._nfe_per_sample.append(dict(stats))
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
        'code_fingerprint': getattr(self, '_code_fingerprint', None),
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
      # NFE / commit accounting (sampler counters). Written beside tok_s.
      if self._nfe_per_sample:
        import statistics as _stats
        forwards = [int(s.get('n_forwards', 0)) for s in self._nfe_per_sample]
        thr_c = [int(s.get('n_thr_commits', 0)) for s in self._nfe_per_sample]
        force_c = [
            int(s.get('n_force_max_commits', 0)) for s in self._nfe_per_sample]
        commits = [int(s.get('n_commits', 0)) for s in self._nfe_per_sample]

        def _pct(xs: list[int], q: float) -> float:
          if not xs:
            return float('nan')
          ys = sorted(xs)
          i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
          return float(ys[i])

        tot_thr = sum(thr_c)
        tot_force = sum(force_c)
        tot_commit = max(tot_thr + tot_force, 1)
        nfe_doc = {
            'checkpoint_path': self.checkpoint_path,
            'source': 'lm_eval.generate_until.sampler_counters',
            'code_fingerprint': getattr(self, '_code_fingerprint', None),
            'n_problems': len(forwards),
            'n_forwards_mean': float(_stats.fmean(forwards)) if forwards else None,
            'n_forwards_p10': _pct(forwards, 0.10),
            'n_forwards_p50': _pct(forwards, 0.50),
            'n_forwards_p90': _pct(forwards, 0.90),
            'n_commits_mean': float(_stats.fmean(commits)) if commits else None,
            'tokens_per_forward_mean': (
                float(self._speed_tokens) / max(sum(forwards), 1)),
            'thr_commit_share': tot_thr / tot_commit,
            'force_max_commit_share': tot_force / tot_commit,
            'n_thr_commits_total': tot_thr,
            'n_force_max_commits_total': tot_force,
            'uniform_commit_order': getattr(
                self._sampler, 'uniform_commit_order', None),
            'unmask_threshold': getattr(
                self._sampler, 'unmask_threshold', None),
            'per_problem': self._nfe_per_sample,
            'note': (
                'n_forwards counts BlockSampler._logits calls. '
                'thr vs force-max shares are UCC sticky commits only.'),
        }
        nfe_path = path.with_name('nfe_metrics.json')
        nfe_path.write_text(
            json.dumps(nfe_doc, indent=2) + '\n', encoding='utf-8')
        print(
            f'Wrote NFE metrics: {nfe_path} '
            f"mean_nfe={nfe_doc['n_forwards_mean']:.1f} "
            f"force_share={nfe_doc['force_max_commit_share']:.3f}",
            flush=True)

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
      # Generation budget = adapter MAX_NEW_TOKENS (Hub PROTOCOL default 2048),
      # then per-task ceilings in ``cap_max_new_for_task`` (mmlu→64, …).
      # Do NOT min() with lm-eval's task ``max_gen_toks``: gsm8k defaults to
      # 256/512 and silently defeated MAX_NEW_TOKENS=2048 (U0 hier canary
      # logged max_new=512 while PROTOCOL said 2048).
      max_gen = cap_max_new_for_task(task, int(self.max_new_tokens))
      key = (
          max_gen,
          tuple(until),
          bool(gen_kwargs.get('do_sample', False)),
          task.split('_')[0],  # keep humaneval vs mbpp buckets separate
      )
      buckets.setdefault(key, []).append((idx, req))

    self._speed_tokens = 0
    self._speed_elapsed = 0.0
    self._nfe_per_sample = []
    for key, batch in buckets.items():
      max_gen, until_tuple, do_sample, _task_prefix = key
      until = list(until_tuple)
      task_name = str(getattr(batch[0][1], 'task_name', '') or '')
      print(
          f'[block_qwen_lm_eval] generate_until task={task_name!r} '
          f'n={len(batch)} max_new={max_gen}',
          flush=True,
      )
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
  import sys
  # Gate code-exec / remote-dataset trust on lm-eval's confirm flag. Unconditional
  # setdefault gave a false safety gate (flag documented but never checked here).
  if '--confirm_run_unsafe_code' in sys.argv:
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
    os.environ.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', 'true')
  # Must run before lm_eval imports humaneval/mbpp utils (load code_eval).
  patch_code_eval_metric_cache()
  patch_humaneval_chat_predictions()
  cli_evaluate()
