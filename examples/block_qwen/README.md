# Qwen block diffusion (four arms)

**Addition on [UNI-D²](https://github.com/nkalyanv99/UNI-D2):** Qwen 2×2 arms
(AR/scratch × masked/uniform). Reuse upstream training CLI, WandB logger,
`generate_samples`, and `generative_ppl` wherever possible.

See [docs/BLOCK_PATH.md](../../docs/BLOCK_PATH.md).

## Train (same style as `examples/bd3lm/owt.sh`)

```bash
bash examples/block_qwen/ar2block_masked.sh
bash examples/block_qwen/ar2block_uniform.sh
bash examples/block_qwen/block_masked.sh      # scratch
bash examples/block_qwen/block_uniform.sh     # scratch
```

```bash
WANDB_PROJECT=block_qwen_final WANDB_NAME=ar2block_masked_v2 \
  bash examples/block_qwen/ar2block_masked.sh
```

WandB = Lightning `WandbLogger` via Hydra (`wandb.project` / `wandb.name`), as in upstream.

In-train free-gen is **off** in `block_qwen` for throughput (upstream default is on).
To match upstream during train:

```bash
# append to any recipe
#   eval.generate_samples=true sampling.num_sample_batches=1
```

## Eval — prefer upstream modules

```bash
# Free-gen + gen-PPL (UNI-D² generate_samples / generative_ppl)
bash examples/block_qwen/eval.sh \
  outputs/block_qwen/ar2block_masked/checkpoints/last.ckpt
```

### Fast-dLLM-style post-train (accuracy + tok/s)

```bash
# Needs: pip install lm-eval
# Writes <run>/lm_eval/SUMMARY.md with task scores + tok/s
bash examples/block_qwen/lm_eval.sh \
  outputs/block_qwen/ar2block_masked_139760/checkpoints/last.ckpt

# Smoke (few examples per task)
LIMIT=4 TASKS=gsm8k,ifeval,humaneval bash examples/block_qwen/lm_eval.sh <ckpt>

# Accuracy only (skip dedicated throughput)
SKIP_THROUGHPUT=1 bash examples/block_qwen/lm_eval.sh <ckpt>
```

| Output | Meaning |
|--------|---------|
| `SUMMARY.md` / `SUMMARY.json` | Task **accuracy** + **tok/s** |
| `tok_s.json` | Dedicated decode throughput bench |
| `tok_s_lm_eval.json` | tok/s during lm-eval generation |

tok/s is **our** `BlockSampler` (not Fast-dLLM hierarchical KV / sub-block-8).

Optional extras (block ELBO):

```bash
python tools/run_block_qwen_eval.py --checkpoint ... --upstream-only
python tools/run_block_qwen_eval.py --checkpoint ...
```

## WandB charts (training)

| Keys | Source |
|------|--------|
| `trainer/loss`, `train/{nll,bpd,ppl}` | `BlockTrainer` (same Lightning path as upstream algos) |
| `val/{nll,bpd,ppl}` | validation metrics collection |
| `val/{nll,bpd,ppl}_alpha_*` | `eval.t_bucketed_nll=true` (enabled in the four recipes) |
| `trainer/lr` | upstream LR monitor callback |
| `val/samples` (table) | only if `eval.generate_samples=true` |

## Smoke

```bash
bash examples/block_qwen/smoke.sh
```

Cluster Slurm wrappers under `scripts/` are site-specific; these `examples/` scripts are the portable recipes.

## References

- Block diffusion SFT: [arXiv:2509.26328](https://arxiv.org/abs/2509.26328)
- BlockGen: [arXiv:2606.02241](https://arxiv.org/abs/2606.02241)
- Upstream: [UNI-D²](https://github.com/nkalyanv99/UNI-D2)
