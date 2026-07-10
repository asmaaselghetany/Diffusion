# Baseline: Fast-dLLM + BlockGen (Qwen block path)

This verified stack does **not** use BD3LM or BlockDiT. See [BLOCK_PATH.md](BLOCK_PATH.md) for the full layout.

## Fast-dLLM ([NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM))

| Adopted | Location |
|---------|----------|
| `block_diff_mask` on `concat(xt, x0)` | `models/block_mask.py` |
| Qwen2 block attention via mask injection | `models/qwen/` |
| Per-block masked corruption + SUBS ELBO | `forward_process/block_masked.py`, `losses/block_elbo.py` |
| Optional `shift_loss_targets` | `configs/algo/block_masked.yaml` |

## BlockGen ([jdeschena/blockgen](https://github.com/jdeschena/blockgen))

| Adopted | Location |
|---------|----------|
| Per-block uniform corruption π = 1/V | `forward_process/block_uniform.py` |
| Uniform-state ELBO (`DUO_BASE.nll_per_token`) | `losses/block_elbo.py` |
| Single trainer, switch `forward_process_name` | `algorithms/block_trainer.py` |

## UNI-D² (infrastructure)

Hydra configs, `python -m discrete_diffusion`, Lightning `train.py`, data loaders — shared with other algorithms in this repo.
