"""Paper cell catalog must stay consistent with submit_paper_cell.sh."""

from pathlib import Path

import yaml

CELLS = Path(__file__).resolve().parents[1] / 'configs' / 'paper' / 'cells.yaml'


def test_paper_cells_catalog_loads():
  reg = yaml.safe_load(CELLS.read_text())
  cells = reg['cells']
  assert 'C0' in cells and cells['C0']['status'] == 'ready'
  assert 'C2_fdllm' in cells and cells['C2_fdllm']['launch'] == 'lever'
  assert cells['C5_joint_ar']['status'] == 'ready'
  assert cells['C5_causal_clean']['status'] == 'ready'
  assert cells['B4_hybrid_p10']['status'] == 'ready'
  assert cells['B4_hybrid_p10']['arm'] == 'hybrid'
  assert cells['B4_hybrid_p50']['status'] == 'ready'
  assert cells['B3_t_strat']['preset'] == 'B3_t_strat'
  assert cells['B3_u_stratified']['preset'] == 'B3_u_stratified'
  assert cells['E_hierarchical']['preset'] == 'decode_hierarchical'
  assert cells['E_dual_cache']['preset'] == 'decode_dual_cache'
  assert cells['E_sub_block']['preset'] == 'decode_sub_block'
  assert cells['E_hierarchical']['status'] == 'eval_only'
  assert cells['E_hierarchical']['launch'] == 'decode_eval'
  assert cells['longitudinal']['launch'] == 'longitudinal_eval'
  assert cells['C4']['status'] == 'eval_only'
  # No soft-defer leftovers in ready cells.
  for name, spec in cells.items():
    notes = str(spec.get('notes') or '')
    assert 'Defer (P6)' not in notes and 'soft-defer' not in notes.lower(), name


def test_ready_train_cells_have_launch():
  reg = yaml.safe_load(CELLS.read_text())
  for name, spec in reg['cells'].items():
    if spec.get('status') == 'ready' and spec.get('kind') == 'train':
      assert spec.get('launch'), f'{name} ready train missing launch'
      if spec['launch'] == 'lever':
        assert spec.get('preset'), f'{name} lever cell missing preset'


def test_no_stub_cells():
  reg = yaml.safe_load(CELLS.read_text())
  stubs = [n for n, s in reg['cells'].items() if s.get('status') == 'stub']
  assert stubs == []
