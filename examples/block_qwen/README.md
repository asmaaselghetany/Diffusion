# Qwen block diffusion (Fast-dLLM + BlockGen path)

AR-init block diffusion on Qwen2: **one** `BlockTrainer`, switch `algo=block_masked` or `algo=block_uniform`. This path does **not** use BD3LM or BlockDiT.

See [docs/BLOCK_PATH.md](../../docs/BLOCK_PATH.md) and [docs/BASELINE.md](../../docs/BASELINE.md).

## Quick train

```bash
export PYTHONPATH=src

python -m discrete_diffusion experiment=block_qwen algo=block_masked \
  model.hub_id=Qwen/Qwen2.5-0.5B \
  data.tokenizer_name_or_path=Qwen/Qwen2.5-0.5B
```

## Local / cluster smoke

```bash
bash examples/block_qwen/smoke.sh
```

## Slurm verification (S0–S6)

```bash
sbatch examples/block_qwen/slurm_verify.sh
```

## Citation pointers

- Fast-dLLM v2: [arXiv:2509.26328](https://arxiv.org/abs/2509.26328)
- BlockGen: [arXiv:2606.02241](https://arxiv.org/abs/2606.02241)
