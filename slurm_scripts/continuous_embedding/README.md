# Continuous Embedding Diffusion Experiments

## Output Directory Structure

All outputs are organized under timestamped directories to preserve multiple runs:
```
JEDi/outputs/continuous_embedding/
├── logs/                                    # SLURM stdout/stderr logs
│   ├── ced_quick_test_<jobid>.out
│   ├── ced_quick_test_<jobid>.err
│   └── ...
├── quick_test_20260202_143052/              # Timestamped run outputs
│   ├── checkpoints/
│   ├── wandb/
│   └── .hydra/
├── decoder_only_baseline_20260202_143105/
├── denoiser_x0_20260202_143110/
├── denoiser_epsilon_20260202_143115/
└── denoiser_v_20260202_143120/
```

**Note**: Each run creates a new timestamped directory (`<experiment>_YYYYMMDD_HHMMSS`), so previous runs are never overwritten.

## Recommended Experiment Order

### 1. Quick Test (FIRST!)
```bash
sbatch quick_test.sh
```
- **Purpose**: Verify the pipeline runs end-to-end
- **Resources**: 1 GPU, 1 hour, 8 CPUs, 64GB total memory
- **Steps**: 100 (just enough to check gradients flow)

### 2. Decoder-Only Baseline
```bash
sbatch decoder_only_baseline.sh
```
- **Purpose**: Upper bound - how well can we reconstruct tokens from clean embeddings?
- **Resources**: 1 GPU, 24 hours, 16 CPUs, 128GB total memory
- **Steps**: 50k
- **Why**: If this fails badly, the decoder architecture needs work before testing diffusion

### 3. x0 Parameterization (Start Here)
```bash
sbatch denoiser_x0.sh
```
- **Purpose**: Most stable diffusion training - predict clean embeddings directly
- **Resources**: 2 GPUs, 48 hours, 16 CPUs, 256GB total memory
- **Steps**: 100k
- **Why**: x0-prediction is easier to train and debug

### 4. Compare Parameterizations
```bash
sbatch denoiser_epsilon.sh  # DDPM-style noise prediction
sbatch denoiser_v.sh        # Velocity prediction (progressive distillation)
```
- **Purpose**: Compare different prediction targets
- **Resources**: 2 GPUs each, 48 hours each

## Resume From Last Checkpoints

Each training launcher (`decoder_only_baseline.sh`, `denoiser_x0.sh`,
`denoiser_epsilon.sh`, `denoiser_v.sh`) now supports resume mode.

- Set `RESUME=1` to resume instead of starting a new timestamped run.
- Optional: `RESUME_RUN_DIR=/path/to/run_dir` to pick a specific run directory.
- Optional: `RESUME_CKPT=/path/to/checkpoint.ckpt` to force a specific checkpoint.
- Resume selection order is: `last.ckpt` first, then `best.ckpt` fallback.
- During resume, training writes into the existing run directory and explicitly
  sets `checkpointing.resume_ckpt_path`.

Examples:
```bash
# Resume x0 from its latest run
sbatch --export=ALL,RESUME=1 denoiser_x0.sh

# Resume epsilon from a specific run
sbatch --export=ALL,RESUME=1,RESUME_RUN_DIR=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/denoiser_epsilon_20260202_130406 denoiser_epsilon.sh

# Resume decoder from explicit checkpoint (best/last)
sbatch --export=ALL,RESUME=1,RESUME_CKPT=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/decoder_only_baseline_20260202_112352/checkpoints/best.ckpt decoder_only_baseline.sh
```

Resume all four models from latest known runs:
```bash
sbatch resume_from_last.sh
```

Checkpoint behavior while training:
- `callbacks.checkpoint_every_n_steps.save_last=true`
- `callbacks.checkpoint_monitor.save_last=true`
- Checkpoint cadence configurable via env:
  `CHECKPOINT_EVERY_N_STEPS` (default: `5000`)

## Decoder-Robustness Stage-2 (Predicted x0 Latents)

Train stage-2 decoder variants that consume denoiser-predicted clean latents:

```bash
# Per variant
sbatch --export=ALL,DENOISER_KIND=x0 slurm_scripts/continuous_embedding/decoder_robust.sh
sbatch --export=ALL,DENOISER_KIND=epsilon slurm_scripts/continuous_embedding/decoder_robust.sh
sbatch --export=ALL,DENOISER_KIND=v slurm_scripts/continuous_embedding/decoder_robust.sh

# Submit all three
bash slurm_scripts/continuous_embedding/submit_decoder_robust_suite.sh
```

Defaults:
- `trainer.max_steps=100000`
- `CHECKPOINT_EVERY_N_STEPS=10000`
- `save_last=true` for both periodic and monitor checkpoints
- source denoiser checkpoint auto-discovered from latest `denoiser_<kind>_*` run

Common overrides:
- `DENOISER_CKPT=/path/to/denoiser/checkpoints/last.ckpt`
- `STAGE1_RUN_DIR=/path/to/denoiser_run`
- `RESUME=1` with optional `RESUME_RUN_DIR` / `RESUME_CKPT`

## Every-10k Capability Evaluation (External Driver)

The out-of-band evaluator runs full capability checks at each periodic checkpoint:

```bash
# One-shot over already existing 10k checkpoints
sbatch --export=ALL,RUN_DIR=/path/to/decoder_robust_x0_<timestamp> \
  slurm_scripts/continuous_embedding/eval_decoder_robust_checkpoints.sh

# Watch mode while training is still running
sbatch --export=ALL,RUN_DIR=/path/to/decoder_robust_x0_<timestamp>,WATCH=1 \
  slurm_scripts/continuous_embedding/eval_decoder_robust_checkpoints.sh
```

Outputs:
- `${RUN_DIR}/capability_eval_10k/step_00XXXX/results.json`
- `${RUN_DIR}/capability_eval_10k/step_00XXXX/results.csv`
- `${RUN_DIR}/capability_eval_10k/checkpoint_eval_manifest.json`

W&B integration:
- logs scalar task metrics with `step=<checkpoint_step>`
- writes per-step result file paths into run summary

## Unified Training Launcher (`train.sh`)

The `train.sh` script is a single parameterized launcher for all objectives and
datasets. Configuration is passed via environment variables.

### Quick Examples

```bash
# Flow map on OWT (default — recommended starting point)
sbatch train.sh

# Flow map on LM1B, 4 GPUs
sbatch --gres=gpu:4 --export=ALL,DATASET=lm1b,NUM_GPUS=4,MAX_STEPS=200000 train.sh

# Flow matching baseline for comparison
sbatch --export=ALL,OBJECTIVE=flow_matching,SAMPLING_METHOD=rectified_few_step train.sh

# iMF baseline (requires JVP-safe attention backend)
sbatch --export=ALL,OBJECTIVE=imf,SAMPLING_METHOD=rectified_few_step train.sh

# DDPM baseline
sbatch --export=ALL,OBJECTIVE=ddpm,SAMPLING_METHOD=ddpm train.sh

# Quick smoke test (100 steps, 1 GPU, ~5 min)
sbatch --gres=gpu:1 --time=01:00:00 \
  --export=ALL,NUM_GPUS=1,MAX_STEPS=100,GLOBAL_BATCH=8 train.sh

# Resume latest flow_map OWT run
sbatch --export=ALL,RESUME=1 train.sh
```

### Key Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `OBJECTIVE` | `flow_map` | `flow_map` / `flow_matching` / `imf` / `ddpm` |
| `DATASET` | `openwebtext` | `openwebtext` / `lm1b` |
| `NUM_GPUS` | `2` | Number of GPUs |
| `MAX_STEPS` | `100000` | Total optimizer steps |
| `GLOBAL_BATCH` | `128` | Global batch size |
| `SEQ_LEN` | `256` | Sequence length |
| `LR` | `1e-4` | Learning rate |
| `LAMBDA_CE` | `0.1` | CE auxiliary weight |
| `S_ZERO_PROB` | `0.0` | Flow map: fraction of s=0 samples |
| `CONDITIONING` | `span_masking` | `none` / `span_masking` |
| `EMBED_PROVIDER` | `tied` | `tied` / `lookup` / `legacy_contextual` |
| `SAMPLING_METHOD` | `flowmap_few_step` | Sampler for eval generation |
| `CHECKPOINT_EVERY_N_STEPS` | `5000` | Periodic checkpoint cadence |

### Output Naming

Runs are saved as `outputs/continuous_embedding/<objective>_<dataset>_<timestamp>/`.
Resume auto-discovers the latest matching run.

## Resource Summary

| Experiment | GPUs | CPUs | Memory (per-cpu) | Time | Steps |
|------------|------|------|------------------|------|-------|
| Quick Test | 1 | 8 | 8GB | 1h | 100 |
| Decoder Baseline | 1 | 16 | 8GB | 24h | 50k |
| x0 Denoiser | 2 | 16 | 16GB | 48h | 100k |
| Epsilon Denoiser | 2 | 16 | 16GB | 48h | 100k |
| V Denoiser | 2 | 16 | 16GB | 48h | 100k |
| **train.sh** (default) | **2** | **16** | **16GB** | **48h** | **100k** |

## SLURM Configuration

All scripts use:
- `--partition=gpu_p`
- `--qos=gpu_reservation` with `--reservation=haicu_stefan`
- `--qos=gpu_normal` is commented out as alternative

To switch to normal queue, edit scripts:
```bash
##SBATCH --qos=gpu_normal      # uncomment this
#SBATCH --qos=gpu_reservation  # comment this
#SBATCH --reservation=haicu_stefan  # comment this
```

## Monitoring

### Check job status
```bash
squeue -u $USER
```

### View logs
```bash
# During run
tail -f outputs/continuous_embedding/logs/ced_quick_test_<jobid>.out

# After completion
cat outputs/continuous_embedding/logs/ced_quick_test_<jobid>.err
```

### W&B Dashboard
Check https://wandb.ai/<your-user>/continuous-embedding-diffusion

## What to Monitor

1. **trainer/loss**: Overall loss (should decrease)
2. **trainer/mse_loss**: Denoising quality (should decrease)
3. **trainer/ce_loss**: Token reconstruction (may plateau early)
4. **val/loss**: Validation loss (watch for overfitting)

## Expected Results

- **Decoder baseline**: Should achieve low CE loss (~2-3) if embeddings are good
- **x0 denoiser**: MSE should decrease; CE may be higher than baseline
- **Epsilon/V**: May have different convergence dynamics

## Troubleshooting

### Job failed immediately
Check the error log:
```bash
cat outputs/continuous_embedding/logs/ced_*_<jobid>.err
```

### OOM Errors
- Reduce `loader.batch_size`
- Enable `model.gradient_checkpointing=true` (disabled by default due to DDP compatibility issues)
- Reduce `model.length` (sequence length)

### NaN Losses
- Check noise schedule (try `noise=cosine` vs `noise=linear`)
- Reduce learning rate
- Check for exploding gradients (gradient_clip_val=1.0)

### Slow Training
- Increase batch size if memory allows
- Use more GPUs
- Reduce validation frequency (`trainer.val_check_interval`)
