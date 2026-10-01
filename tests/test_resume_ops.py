"""Contract tests for resume_block_qwen.sh (ops layer)."""

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_RESUME = (_REPO / 'scripts' / 'resume_block_qwen.sh').read_text()


def test_resume_script_exists_and_is_executable_bit_or_bash():
  path = _REPO / 'scripts' / 'resume_block_qwen.sh'
  assert path.is_file()
  assert _RESUME.startswith('#!/')


def test_resume_requires_recorded_world_size():
  assert 'trainer.num_nodes/devices are required' in _RESUME
  assert 'ORIG_NUM_NODES' in _RESUME
  assert 'ORIG_GPUS_PER_NODE' in _RESUME
  assert re.search(
      r'ORIG_NUM_NODES="\$\(_read_override trainer\.num_nodes\)"', _RESUME)
  assert re.search(
      r'ORIG_GPUS_PER_NODE="\$\(_read_override trainer\.devices\)"', _RESUME)


def test_resume_rejects_resource_migration_by_default():
  assert 'ALLOW_RESOURCE_MIGRATION' in _RESUME
  assert 'differs from recorded' in _RESUME
  # Default must stay off (migration only with explicit opt-in).
  assert re.search(
      r'ALLOW_RESOURCE_MIGRATION:-0', _RESUME)


def test_resume_exports_ddp_resource_trio():
  for name in ('NUM_NODES', 'GPUS_PER_NODE', 'NUM_GPUS'):
    assert f'export {name}' in _RESUME


def test_resume_pins_checkpoint_path_override():
  assert 'checkpointing.resume_ckpt_path=${CKPT}' in _RESUME
  assert '_block_qwen_pick_highest_ckpt' in _RESUME


def test_resume_sbatch_passes_resource_exports():
  # Submitted job must inherit the resolved world size.
  assert 'NUM_NODES,GPUS_PER_NODE,NUM_GPUS' in _RESUME
  assert '--ntasks-per-node="${GPUS_PER_NODE}"' in _RESUME
  assert '--gres="gpu:${GPUS_PER_NODE}"' in _RESUME
  assert '--nodes="${NUM_NODES}"' in _RESUME


def test_resume_supports_dry_run():
  assert 'DRY_RUN' in _RESUME
  assert 'not submitting' in _RESUME


def test_resolve_resume_ckpt_missing_fails_hard(monkeypatch):
  from types import SimpleNamespace
  from unittest.mock import patch

  import pytest

  from discrete_diffusion.train import resolve_resume_ckpt_path

  monkeypatch.delenv('ALLOW_MISSING_RESUME', raising=False)
  cfg = SimpleNamespace(
      checkpointing=SimpleNamespace(
          resume_from_ckpt=True,
          resume_ckpt_path='/nonexistent/missing.ckpt',
      ),
  )
  with patch('discrete_diffusion.train.utils.fsspec_exists', return_value=False):
    with pytest.raises(FileNotFoundError, match='not found'):
      resolve_resume_ckpt_path(cfg)


def test_resolve_resume_ckpt_allow_missing_escape(monkeypatch):
  from types import SimpleNamespace
  from unittest.mock import patch

  from discrete_diffusion.train import resolve_resume_ckpt_path

  monkeypatch.setenv('ALLOW_MISSING_RESUME', '1')
  cfg = SimpleNamespace(
      checkpointing=SimpleNamespace(
          resume_from_ckpt=True,
          resume_ckpt_path='/nonexistent/missing.ckpt',
      ),
  )
  with patch('discrete_diffusion.train.utils.fsspec_exists', return_value=False):
    assert resolve_resume_ckpt_path(cfg) is None


def test_resolve_resume_ckpt_empty_path_fails(monkeypatch):
  from types import SimpleNamespace

  import pytest

  from discrete_diffusion.train import resolve_resume_ckpt_path

  monkeypatch.delenv('ALLOW_MISSING_RESUME', raising=False)
  cfg = SimpleNamespace(
      checkpointing=SimpleNamespace(
          resume_from_ckpt=True,
          resume_ckpt_path='',
      ),
  )
  with pytest.raises(FileNotFoundError, match='empty'):
    resolve_resume_ckpt_path(cfg)
