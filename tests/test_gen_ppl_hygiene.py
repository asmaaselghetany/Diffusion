"""Tests for Unifusion-style GenPPL–entropy hygiene helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_pair_from_metrics_requires_entropy(tmp_path: Path):
  from tools.gen_ppl_hygiene import pair_from_metrics, write_pair_sidecar

  with pytest.raises(ValueError, match='unigram_entropy_mean'):
    pair_from_metrics({'ppl': 60.0})

  with pytest.raises(ValueError, match='not citeable'):
    pair_from_metrics({
        'ppl': 402.0,
        'unigram_entropy_mean': 6.0,
        'honesty_warning': 'high_eos_rate_short_span: launder',
        'citeable': False,
    })

  metrics = tmp_path / 'gen_ppl_metrics.json'
  metrics.write_text(json.dumps({
      'ppl': 60.8,
      'unigram_entropy_mean': 6.04,
      'unigram_entropy_median': 6.62,
      'first_chunk_only': True,
  }))
  pair = write_pair_sidecar(metrics)
  assert '60.80' in pair['format']
  assert '6.040' in pair['format']
  assert pair.get('citeable') is True
  assert (tmp_path / 'gen_ppl_pair.json').is_file()


def test_collapse_panel_and_nfe_aggregate(tmp_path: Path):
  from tools.gen_ppl_hygiene import (
      aggregate_nfe_sweep,
      build_collapse_panel,
  )

  samples = tmp_path / 'samples.txt'
  samples.write_text(
      '# header\n'
      'Sample 0:\ngood text alpha\n'
      + ('-' * 80) + '\n'
      'Sample 1:\nshort\n'
      + ('-' * 80) + '\n'
      'Sample 2:\nmore text here\n'
      + ('-' * 80) + '\n'
  )
  metrics = tmp_path / 'm.json'
  metrics.write_text(json.dumps({
      'ppl': 55.0,
      'unigram_entropy_mean': 4.0,
      'unigram_entropy_median': 4.1,
  }))
  panel = build_collapse_panel(
      samples, metrics, out_path=tmp_path / 'panel.json')
  assert panel['story'] == 'low_ppl_low_H_possible_collapse'
  assert len(panel['picks']) >= 1

  sweep = tmp_path / 'nfe'
  for steps, ppl, h in [(8, 90.0, 5.5), (32, 60.0, 6.0)]:
    d = sweep / f'steps_{steps}'
    d.mkdir(parents=True)
    (d / 'gen_ppl_metrics.json').write_text(json.dumps({
        'ppl': ppl,
        'unigram_entropy_mean': h,
        'unigram_entropy_median': h,
    }))
  table = aggregate_nfe_sweep(sweep)
  assert [r['num_steps'] for r in table['rows']] == [8, 32]
  assert (sweep / 'nfe_gen_ppl_entropy_table.csv').is_file()


def test_run_gen_ppl_hygiene_mode_nfe_does_not_hit_unknown():
  """Regression: bash ``;;&`` must not fall into ``*)`` after MODE=nfe.

  Job 2083688 finished the NFE sweep then exited 1 with ``Unknown MODE=nfe``.
  """
  import re
  import subprocess

  script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_gen_ppl_hygiene.sh'
  text = script.read_text()
  # Upfront validation must exist before the fallthrough case.
  assert re.search(
      r'case "\$\{MODE\}" in\s*\n\s*dual\|nfe\|multiseed\|panel\|all\)',
      text,
  ), 'missing upfront MODE allow-list'
  # The fallthrough case must not end with a catch-all that fires after nfe.
  fallthrough = text.split('case "${MODE}" in', 2)[-1]
  assert 'Unknown MODE=${MODE}' not in fallthrough.split('esac', 1)[0]

  probe = r'''
set -euo pipefail
MODE=nfe
case "${MODE}" in dual|nfe|multiseed|panel|all) ;; *) echo UNKNOWN; exit 1;; esac
ran=""
case "${MODE}" in
  dual|all) ran="${ran}dual;" ;;&
  nfe|all) ran="${ran}nfe;" ;;&
  multiseed|all) ran="${ran}ms;" ;;&
  panel|all) ran="${ran}panel;" ;;
esac
test "$ran" = "nfe;"
'''
  subprocess.run(['bash', '-c', probe], check=True)
