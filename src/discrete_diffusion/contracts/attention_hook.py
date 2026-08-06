"""Layer 2 — Qwen block-attention hook must exist or fail at load time.

Jobs ``137328``/``137329`` crashed mid-run when a leaked transformers
build lacked ``_update_causal_mask``. Catch that at model construction
instead of after NCCL timeout.
"""

from __future__ import annotations

HOOK_MISSING_MSG = (
    'Qwen inner model lacks `_update_causal_mask` — block-diff attention '
    'cannot install. Usually a transformers version / PYTHONPATH leak '
    '(need a build that exposes this hook, e.g. 4.45.x pinned in '
    'uni-d2/.venv). Reset PYTHONPATH to repo `src` only and retry.'
)


def assert_block_attention_hook_compatible(model) -> None:
  """Raise ``AttributeError`` with a clear message if the hook point is missing.

  ``model`` may be ``AutoModelForCausalLM`` (has ``.model``) or the inner
  ``Qwen2Model`` itself.
  """
  inner = model.model if hasattr(model, 'model') else model
  if not hasattr(inner, '_update_causal_mask'):
    raise AttributeError(HOOK_MISSING_MSG)
  updater = getattr(inner, '_update_causal_mask')
  if not callable(updater):
    raise AttributeError(HOOK_MISSING_MSG + ' (attribute present but not callable)')
