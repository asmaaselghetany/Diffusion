"""Tests for milestone eval helpers and checkpoint override merge."""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]


def test_merge_hydra_overrides_sampling():
  from discrete_diffusion.evaluations.checkpoint_utils import merge_hydra_overrides

  cfg = OmegaConf.create({'sampling': {'hierarchical_kv': False, 'steps': 32}})
  merged = merge_hydra_overrides(cfg, [
      'sampling.hierarchical_kv=true',
      'sampling.sub_block_size=8',
  ])
  assert merged.sampling.hierarchical_kv is True
  assert int(merged.sampling.sub_block_size) == 8
  assert int(merged.sampling.steps) == 32


def test_milestone_find_ckpt_prefers_exact(tmp_path):
  spec = importlib.util.spec_from_file_location(
      'run_milestone_eval', _REPO / 'tools' / 'run_milestone_eval.py')
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  _find_ckpt = mod._find_ckpt

  ckpt_dir = tmp_path / 'checkpoints'
  ckpt_dir.mkdir()
  exact = ckpt_dir / '0-500.ckpt'
  exact.write_bytes(b'PK')  # not a real ckpt — _find_ckpt only checks filename
  found = _find_ckpt(ckpt_dir, 500)
  assert found == exact


def test_submit_nfe_sweep_exists():
  assert (_REPO / 'scripts' / 'submit_nfe_sweep.sh').is_file()


def test_hybrid_launch_arm_in_launch_script():
  text = (_REPO / 'scripts' / '_block_qwen_launch.bash').read_text()
  assert 'hybrid)  ALGO=block_hybrid' in text
  ddp = (_REPO / 'scripts' / '_block_qwen_ddp.bash').read_text()
  assert 'trainer.num_nodes' in ddp
  assert '--ntasks-per-node="${GPUS_PER_NODE}"' in ddp
  import re
  assert re.search(r'^\s*srun\b[^\n]*--gpus-per-task=1', ddp, re.M) is None
