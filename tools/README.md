# tools/

One-off dev, verification, and eval utilities.

## Eval (paper parity)

| Script | Purpose |
|--------|---------|
| `run_block_qwen_eval.py` | Full suite: samples → gen-PPL (gpt2-large) → DepBench → ELBO sweep |
| `run_block_elbo_sweep.py` | Per-block-size validation ELBO only |

```bash
source ../env.sh
python tools/run_block_qwen_eval.py \
  --checkpoint outputs/block_qwen/masked_<jobid>/checkpoints/last.ckpt

# Or via SLURM
./scripts/submit_block_qwen_eval.sh runs
```

Run from repo root with `PYTHONPATH=src` (or `source ../env.sh` from the workspace).
