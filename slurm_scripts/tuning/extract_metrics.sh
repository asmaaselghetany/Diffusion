#!/bin/bash
# extract_phase1a_metrics.sh

PHASE_DIR="${1:-/path/to/phase1a_encoder}"
OUTPUT="${2:-phase1a_metrics.csv}"

python -c "
from discrete_diffusion.tuning.analysis import load_sweep_results
import pandas as pd

df = load_sweep_results('${PHASE_DIR}')
df = df[df['status'] == 'completed']
df.to_csv('${OUTPUT}', index=False)
print(f'Exported {len(df)} runs to ${OUTPUT}')
"   