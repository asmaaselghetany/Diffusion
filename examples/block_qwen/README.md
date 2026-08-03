# Qwen block diffusion (masked + uniform)

AR-init block diffusion on Qwen2: **one** `BlockTrainer`, switch `algo=block_masked` or `algo=block_uniform`. This path does **not** use BD3LM or BlockDiT.

See [docs/BLOCK_PATH.md](../../docs/BLOCK_PATH.md), [docs/BLOCK_QWEN_TRAINING.md](../../docs/BLOCK_QWEN_TRAINING.md), and [docs/BASELINE.md](../../docs/BASELINE.md).

## Quick train

```bash
export PYTHONPATH=src

python -m discrete_diffusion +experiment=block_qwen algo=block_masked
python -m discrete_diffusion +experiment=block_qwen algo=block_uniform
```

## Local / cluster smoke

```bash
bash examples/block_qwen/smoke.sh
```

## Slurm verification (S0–S6)

```bash
sbatch scripts/train_block_qwen_verify.sbatch
```

## Slurm training (masked | uniform)

```bash
./scripts/submit_block_qwen.sh
sbatch scripts/slurm/masked.sbatch
sbatch scripts/slurm/uniform.sbatch
```

## References

- Block diffusion SFT: [arXiv:2509.26328](https://arxiv.org/abs/2509.26328)
- BlockGen: [arXiv:2606.02241](https://arxiv.org/abs/2606.02241)
