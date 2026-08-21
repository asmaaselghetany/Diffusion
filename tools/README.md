# tools/

Prefer **upstream** entry points when possible:

| Task | Prefer |
|------|--------|
| Train | `examples/block_qwen/*.sh` → `python -m discrete_diffusion` |
| Samples + gen-PPL | `examples/block_qwen/eval.sh` → `evaluations.generate_samples` + `generative_ppl` |
| Full suite + DepBench/ELBO | `run_block_qwen_eval.py` (wraps upstream + fork extras) |

## Upstream-aligned eval

```bash
bash examples/block_qwen/eval.sh outputs/block_qwen/<run>/checkpoints/last.ckpt

# or explicitly
python tools/run_block_qwen_eval.py --checkpoint ... --upstream-only
```

## Fork extras / sanity

| Script | Purpose |
|--------|---------|
| `run_block_elbo_sweep.py` | Per-block-size validation ELBO (not in upstream) |
| `run_block_arm_sanity.py` | Per-arm smoke checks |
| `run_pipeline_sanity.py` | End-to-end pipeline sanity |
| `run_copy_x0_probe.py` | Copy-x0 / RoPE geometry probe |
| `verify_*.py` | Load / forward / sample / AR-init checks |
| `smoke_train.py` / `smoke_loss_trend.py` | Short train smokes |
| `preprocess_text8.py` | Text8 preprocessing (upstream-style) |
| `render_block_arch_diagram.py` | Regenerate block architecture figure |
