# Qwen block diffusion path

Verified **AR-init block diffusion** on Qwen2: one trainer (`BlockTrainer`), two arms via Hydra (`block_masked` | `block_uniform`). Not BD3LM / BlockDiT.

![Architecture](block_architecture.png)

## Layout (matches UNI-D² conventions)

| Layer | Config | Code |
|-------|--------|------|
| Algorithm | `configs/algo/block_{masked,uniform}.yaml` | `algorithms/block_trainer.py` |
| Model | `configs/model/qwen_block.yaml` | `models/qwen/` |
| Forward process | `configs/forward_process/block_*.yaml` | `forward_process/block_*.py` |
| Loss | — | `losses/block_elbo.py` |
| Sampler | `configs/sampling/block.yaml` | `sampling/block_sampler.py` |
| Noise | `configs/noise/log-linear.yaml` (algo default) | `noise_schedules/log_linear.py` |
| Init metrics | — | `training/init.py` |

## Train (cluster GPU)

```bash
source /fast/project/HFMI_SynergyUnit/asmaa.elsayed/env.sh
cd "$REPO_ROOT"

# Masked arm
python -m discrete_diffusion \
  +experiment=block_qwen \
  algo=block_masked

# Uniform arm — only algo changes
python -m discrete_diffusion \
  +experiment=block_qwen \
  algo=block_uniform
```

Or use `bash examples/block_qwen/smoke.sh` or `sbatch scripts/train_block_qwen_verify.sbatch`.

## Verification

See [VERIFICATION.md](VERIFICATION.md). Tier-0: `pytest tests/ -q -m "not qwen"`. Tier-1+: GPU utilities under `tools/`.

## References

- [BASELINE.md](BASELINE.md) — masked vs uniform block diffusion baselines
