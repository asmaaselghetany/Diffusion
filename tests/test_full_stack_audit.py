"""Pytest entrypoint for static full-stack audit checks."""

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOOL = _REPO / 'tools' / 'audit_full_stack.py'


def _load_audit():
  spec = importlib.util.spec_from_file_location('audit_full_stack', _TOOL)
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  sys.modules['audit_full_stack'] = mod
  spec.loader.exec_module(mod)
  return mod


_audit = _load_audit()


def test_full_stack_static_layers_pass():
  results = _audit.run_static()
  failures = [
      (lr.layer, chk.name, chk.detail)
      for lr in results
      for chk in lr.checks
      if not chk.ok
  ]
  assert not failures, 'full-stack audit failures:\n' + '\n'.join(
      f'  {layer}/{name}: {detail}' for layer, name, detail in failures)


def test_all_audit_layers_registered():
  assert set(_audit.STATIC_LAYERS) <= {
      'data_sft', 'model', 'eval', 'launch_ddp', 'resume_ops',
      'codex_parity', 'paper_ops',
  }


def test_resume_ops_has_pytest_module():
  paths = _audit.PYTEST_BY_LAYER['resume_ops']
  assert 'tests/test_resume_ops.py' in paths
  assert (_REPO / 'tests' / 'test_resume_ops.py').is_file()


def test_strict_extras_cover_orphans():
  extras = _audit.PYTEST_STRICT_EXTRA
  assert 'tests/test_joint_ar_and_hybrid.py' in extras['algorithm']
  assert 'tests/test_dual_cache.py' in extras['model']
  assert 'tests/test_arpc_and_hierarchical.py' in extras['eval']
  for layer, paths in extras.items():
    for rel in paths:
      assert (_REPO / rel).is_file(), f'missing strict suite {layer}: {rel}'


def test_strict_merges_without_dropping_core():
  core = _audit._pytest_paths_for_layer('algorithm', strict=False)
  strict = _audit._pytest_paths_for_layer('algorithm', strict=True)
  assert set(core) <= set(strict)
  assert len(strict) > len(core)
