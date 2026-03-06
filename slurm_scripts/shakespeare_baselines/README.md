# Tiny Shakespeare Baseline Suite (Hybrid SLURM)

This directory provides a hybrid script layout:

- Core scripts with all logic:
  - `train_core.sh`
  - `eval_core.sh`
- Thin wrapper scripts with model defaults + `#SBATCH` resources:
  - `train_mdlm.sh`
  - `train_latent_jepa.sh`
  - `train_continuous_embedding.sh`
  - `eval_mdlm_gen_ppl.sh`
  - `eval_latent_jepa_gen_ppl.sh`
  - `eval_continuous_gen_ppl.sh`

## Wrapper-first usage

Train baselines:

```bash
sbatch slurm_scripts/shakespeare_baselines/train_mdlm.sh
sbatch slurm_scripts/shakespeare_baselines/train_latent_jepa.sh
sbatch slurm_scripts/shakespeare_baselines/train_continuous_embedding.sh
```

Evaluate with unified Gen-PPL (set `RUN_DIR` from the corresponding training output):

```bash
sbatch --export=ALL,RUN_DIR=/path/to/mdlm_run \
  slurm_scripts/shakespeare_baselines/eval_mdlm_gen_ppl.sh

sbatch --export=ALL,RUN_DIR=/path/to/latent_jepa_run \
  slurm_scripts/shakespeare_baselines/eval_latent_jepa_gen_ppl.sh

sbatch --export=ALL,RUN_DIR=/path/to/continuous_run \
  slurm_scripts/shakespeare_baselines/eval_continuous_gen_ppl.sh
```

## Direct core usage

You can run cores directly when needed:

```bash
MODEL=mdlm DRY_RUN=1 slurm_scripts/shakespeare_baselines/train_core.sh
MODEL=latent_jepa DRY_RUN=1 slurm_scripts/shakespeare_baselines/eval_core.sh
```

Supported `MODEL` values are exactly:

- `mdlm`
- `latent_jepa`
- `continuous`

## Evaluation protocol

All eval wrappers run the same flow:

1. `python -m discrete_diffusion.evaluations.generate_samples`
2. `python -m discrete_diffusion.evaluations.generative_ppl`

Defaults for comparability:

- `EVAL_MODEL=gpt2`
- `retokenize=true`
- `first_chunk_only=true`
- `MAX_LENGTH=${SEQ_LEN}`

## Checkpoint resolution

`eval_core.sh` resolves checkpoints in strict order:

1. explicit `CHECKPOINT_PATH`
2. `${RUN_DIR}/checkpoints/best.ckpt`
3. `${RUN_DIR}/checkpoints/last.ckpt`

If nothing exists, it exits with a clear error and prints attempted paths.

## Output layout

Training:

- `${OUTPUT_BASE}/${RUN_NAME}/checkpoints/...`

Evaluation:

- `${RUN_DIR}/eval/gen_ppl/<timestamp>/samples.pt`
- `${RUN_DIR}/eval/gen_ppl/<timestamp>/samples.txt`
- `${RUN_DIR}/eval/gen_ppl/<timestamp>/gen_ppl_metrics.json`

## Key environment parameters

Training core (`train_core.sh`):

- `MODEL`, `REPO_ROOT`, `OUTPUT_BASE`, `DATA_CACHE_DIR`, `RUN_NAME`
- `NUM_GPUS`, `MAX_STEPS`, `GLOBAL_BATCH`, `SEQ_LEN`
- `WANDB_MODE`, `WANDB_PROJECT`
- `DRY_RUN=1` to print command only

Eval core (`eval_core.sh`):

- `MODEL`, `REPO_ROOT`, `RUN_DIR` or `CHECKPOINT_PATH`
- `NUM_SAMPLES`, `NUM_STEPS`, `GEN_BATCH_SIZE`
- `EVAL_MODEL`, `EVAL_BATCH_SIZE`, `MAX_LENGTH`
- `DRY_RUN=1` to print commands only
