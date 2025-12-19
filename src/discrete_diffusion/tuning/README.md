# Hyperparameter Tuning Module

This module provides systematic hyperparameter optimization for the Latent JEPA
discrete diffusion model, following the research plan in `hyperparameter_tuning_plan.md`.

## Quick Start

### 1. Run a Single Sweep

```bash
# Generate and run encoder architecture sweep
python -m discrete_diffusion.tuning.sweep_runner \
    configs/sweep/stage1_180m_encoder.yaml \
    --output-dir sweep_outputs/encoder_sweep \
    --num-runs 60 \
    --wandb-project jepa_hparam_study
```

### 2. Submit via Slurm

```bash
# Submit full 180M hyperparameter study
sbatch slurm_scripts/tuning/run_full_study_180m.sh

# Or submit individual phases
./slurm_scripts/tuning/submit_phase.sh encoder
./slurm_scripts/tuning/submit_phase.sh predictor
./slurm_scripts/tuning/submit_phase.sh readout /path/to/stage1/checkpoint
```

### 3. Analyze Results

```bash
# Generate sweep report
python -m discrete_diffusion.tuning.analysis \
    sweep_outputs/encoder_sweep \
    --metric latent/cosine_similarity \
    --goal maximize

# Compare multiple sweeps
python -c "
from discrete_diffusion.tuning.analysis import compare_sweeps
df = compare_sweeps(['sweep_outputs/sweep1', 'sweep_outputs/sweep2'], 'latent/cosine_similarity')
print(df)
"
```

## Module Components

### `sweep_config.py`
Parse and validate sweep configurations with support for:
- Grid search, random search, Latin hypercube sampling
- Parameter constraints (divisibility, ordering)
- Conditional parameter overrides

### `sweep_runner.py`
Execute hyperparameter sweeps:
- Sequential or parallel execution
- Slurm job array generation
- WandB integration
- Checkpoint resumption

### `experiment_tracker.py`
Track and aggregate experiment results:
- Persistent JSON database
- Best run identification
- Export to CSV/Parquet

### `early_stopping.py`
Lightning callbacks for efficient search:
- `CollapseDetectionCallback`: Stop on representation collapse
- `PerformanceEarlyStoppingCallback`: Stop on no improvement
- `MetricLoggingCallback`: Comprehensive metric logging

### `analysis.py`
Analysis utilities:
- Feature importance computation
- Interaction effect analysis
- Sweep comparison
- Report generation

## Sweep Configuration Format

```yaml
sweep:
  name: sweep_name
  method: latin_hypercube  # grid, random, latin_hypercube
  metric:
    name: latent/cosine_similarity
    goal: maximize
  
  early_terminate:
    type: collapse_detection
    min_steps: 5000
    std_threshold: 0.1
  
  parameters:
    model.latent_dim:
      values: [256, 512, 768]
      baseline: 512

fixed:
  algo: jepa
  trainer.max_steps: 200000

constraints:
  - type: divisible
    params: [model.hidden_size, model.n_heads]

conditionals:
  - condition: "algo.redundancy == 'none'"
    overrides:
      algo.lambda_var: 0.0
```

## Study Phases (180M Model)

| Phase | Sweep Config | Runs | Method |
|-------|--------------|------|--------|
| 1A | stage1_180m_encoder | 60 | Latin Hypercube |
| 2A | stage1_180m_predictor | 50 | Grid |
| 2B | stage1_180m_training | 40 | Grid |
| 2C | stage1_180m_ema | 27 | Grid |
| 3 | stage2_180m_readout | 60 | Grid |

## Best Practices

1. **Start with baselines**: Run baseline configuration first
2. **Use Latin hypercube**: More efficient than grid for high-dimensional spaces
3. **Enable collapse detection**: Prevents wasted compute on failed runs
4. **Checkpoint frequently**: Enable resume for fault tolerance
5. **Monitor WandB**: Track runs in real-time

## Compute Budget

| Model | Stage 1 (200k steps) | Stage 2 (50k steps) |
|-------|---------------------|---------------------|
| 180M | ~12h (4×A100) | ~3h (4×A100) |
| 350M | ~24h (4×A100) | ~6h (4×A100) |

Estimated total for full 180M study: ~3,500 GPU-hours

