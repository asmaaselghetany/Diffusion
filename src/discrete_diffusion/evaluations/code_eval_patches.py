"""Lightweight lm-eval / HF-evaluate patches for code tasks.

Kept free of torch / Lightning so Hub Fast-dLLM ``eval.py`` (separate venv)
can import these without pulling the UNI-D2 train stack (SIGBUS risk).
"""

from __future__ import annotations

import os
import uuid

from discrete_diffusion.evaluations.block_qwen_eval_utils import (
    assemble_humaneval_prediction,
)


def patch_humaneval_chat_predictions() -> None:
  """Make HumanEval ``build_predictions`` safe under chat rewrites.

  Stock filter always does ``prompt + completion``. Chat models regenerate
  ``def {entry_point}``; concatenating then breaks execution. Assemble a
  single executable program instead.
  """
  try:
    from lm_eval.tasks.humaneval import utils as he_utils
  except Exception:
    return
  if getattr(he_utils.build_predictions, '_uni_d2_chat_patched', False):
    return

  def build_predictions(resps, docs):
    out = []
    for resp, doc in zip(resps, docs):
      prompt = doc['prompt']
      entry = doc.get('entry_point') or ''
      out.append([
          assemble_humaneval_prediction(prompt, r, entry) for r in resp
      ])
    return out

  build_predictions._uni_d2_chat_patched = True  # type: ignore[attr-defined]
  he_utils.build_predictions = build_predictions


def patch_code_eval_metric_cache() -> None:
  """Avoid HF ``code_eval`` arrow-cache races under Accelerate multi-rank.

  ``lm_eval`` HumanEval/MBPP utils call ``evaluate.load("code_eval")`` at
  import with the default ``experiment_id``. Every rank (and every concurrent
  Slurm job sharing ``HF_METRICS_CACHE``) then fights over
  ``default_experiment-1-0.arrow`` → import-time ``ValueError`` /
  ``FileNotFoundError``. Inject a unique ``experiment_id`` and prefer
  ``keep_in_memory`` so ranks/jobs do not share disk state.
  """
  try:
    import evaluate as hf_evaluate
  except ImportError:
    return
  if getattr(hf_evaluate.load, '_uni_d2_code_eval_patched', False):
    return
  _orig = hf_evaluate.load

  def _load(path, *args, **kwargs):
    name = str(path)
    if name == 'code_eval' or name.endswith('/code_eval'):
      if not kwargs.get('experiment_id'):
        job = os.environ.get('SLURM_JOB_ID', 'local')
        rank = (
            os.environ.get('RANK')
            or os.environ.get('LOCAL_RANK')
            or os.environ.get('SLURM_PROCID')
            or '0')
        kwargs['experiment_id'] = (
            f'code_eval_{job}_r{rank}_{os.getpid()}_{uuid.uuid4().hex[:8]}')
      kwargs.setdefault('keep_in_memory', True)
      base = os.environ.get('OUT_DIR') or os.environ.get('TMPDIR') or '/tmp'
      kwargs.setdefault(
          'cache_dir',
          os.path.join(base, 'evaluate_code_eval', kwargs['experiment_id']))
      os.makedirs(kwargs['cache_dir'], exist_ok=True)
    return _orig(path, *args, **kwargs)

  _load._uni_d2_code_eval_patched = True  # type: ignore[attr-defined]
  hf_evaluate.load = _load
