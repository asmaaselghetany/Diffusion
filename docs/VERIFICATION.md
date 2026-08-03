# Verification gates (cluster GPU)

Training requires **CUDA**. See [BLOCK_PATH.md](BLOCK_PATH.md) for layout.

## Setup

```bash
source /fast/project/HFMI_SynergyUnit/asmaa.elsayed/env.sh
cd "$REPO_ROOT"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Tier 0 — CPU unit tests (no GPU)

```bash
pytest tests/ -q -m "not qwen"
```

Covers: block mask, per-block forward processes, masked/uniform ELBO.

## Tier 1 — GPU verification scripts

| Step | Command | What it checks |
|------|---------|----------------|
| **S0** | `python tools/smoke_train.py --algo block_masked --steps 1` | 1 train step |
| **S1** | `python tools/verify_qwen_load.py` | HF load + causal parity |
| **S2** | `python tools/verify_block_forward.py` | block ≠ causal |
| **S3/S4** | `python tools/smoke_loss_trend.py --steps 100` | loss ↓ both arms |
| **S5** | `python tools/verify_ar_block_init.py` | init metrics step 0 |
| **S6** | `python tools/verify_block_sample.py` | generation smoke |

**One command (GPU node):**

```bash
bash examples/block_qwen/smoke.sh
```

## Tier 2 — Qwen pytest (optional)

```bash
RUN_QWEN_TESTS=1 pytest tests/test_qwen_parity.py -q
```

## Slurm

```bash
sbatch scripts/train_block_qwen_verify.sbatch
```

## Full training smoke

```bash
python -m discrete_diffusion +experiment=block_qwen algo=block_masked \
  trainer.max_steps=200 checkpointing.resume_from_ckpt=false
python -m discrete_diffusion +experiment=block_qwen algo=block_uniform \
  trainer.max_steps=200 checkpointing.resume_from_ckpt=false
```
