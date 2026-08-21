# Qwen block diffusion path

**Fork addition** on [UNI-D²](https://github.com/nkalyanv99/UNI-D2): four arms
(AR/scratch × masked/uniform) via `BlockTrainer`. Prefer upstream pieces for
everything else (Hydra CLI, WandB logger, `generate_samples`, `generative_ppl`).

![Architecture](block_architecture.png)

## Four arms (2×2) — this is the addition

| Arm | Init | Corruption | Recipe |
|-----|------|------------|--------|
| `ar2block_masked` | AR Instruct | masked | `examples/block_qwen/ar2block_masked.sh` |
| `ar2block_uniform` | AR Instruct | uniform | `examples/block_qwen/ar2block_uniform.sh` |
| `block_masked` | scratch | masked | `examples/block_qwen/block_masked.sh` |
| `block_uniform` | scratch | uniform | `examples/block_qwen/block_uniform.sh` |

Shared experiment: `configs/experiment/block_qwen.yaml` (hooks off).

## What we reuse from UNI-D²

| Need | Upstream component |
|------|-------------------|
| Train entry | `python -m discrete_diffusion` + Hydra |
| Logging | Lightning `WandbLogger` / `wandb.*` config |
| Free-gen | `discrete_diffusion.evaluations.generate_samples` |
| Gen-PPL | `discrete_diffusion.evaluations.generative_ppl` |
| Task benches + tok/s | `block_qwen_lm_eval` + `decode_throughput` via `examples/block_qwen/lm_eval.sh` |
| Callbacks / trainer loop | existing Lightning stack |

## What is new (keep small)

| Layer | Config | Code |
|-------|--------|------|
| Algorithm | `configs/algo/block_{masked,uniform}.yaml` | `algorithms/block_trainer.py` |
| Model | `configs/model/qwen_block.yaml` | `models/qwen/` |
| Forward process | `configs/forward_process/block_*.yaml` | `forward_process/block_*.py` |
| Loss | — | `losses/block_elbo.py` |
| Sampler | `configs/sampling/block.yaml` | `sampling/block_sampler.py` |

## Train

```bash
bash examples/block_qwen/ar2block_masked.sh
```

## Eval (upstream first)

```bash
# Free-gen + gen-PPL
bash examples/block_qwen/eval.sh outputs/block_qwen/<run>/checkpoints/last.ckpt

# Fast-dLLM-style: task accuracy (GSM8K/MMLU/IFEval/HumanEval/…) + tok/s
bash examples/block_qwen/lm_eval.sh outputs/block_qwen/<run>/checkpoints/last.ckpt
# → <run>/lm_eval/SUMMARY.md
```

Fork-only extras (DepBench / ELBO): `tools/run_block_qwen_eval.py`.

## References

- Block diffusion SFT: [arXiv:2509.26328](https://arxiv.org/abs/2509.26328)
- BlockGen: [arXiv:2606.02241](https://arxiv.org/abs/2606.02241)
- Upstream: [nkalyanv99/UNI-D2](https://github.com/nkalyanv99/UNI-D2)
