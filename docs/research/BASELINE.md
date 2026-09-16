# Baseline: masked + uniform block diffusion (Qwen path)

This verified stack does **not** use BD3LM or BlockDiT. See [BLOCK_PATH.md](BLOCK_PATH.md) for the full layout.

**Design rationale (masked vs uniform, pretrained Qwen vs train-our-own AR):** [BASELINE_MASKED_UNIFORM_AR.md](BASELINE_MASKED_UNIFORM_AR.md)

## Neutral comparison design

`block_qwen` trains two arms from the same AR-init checkpoint with **identical** everything except corruption:

| Shared | Masked arm | Uniform arm |
|--------|------------|-------------|
| Qwen2.5 block backbone, `block_size`, seq len, data, optim, steps | `forward_process_name: masked` | `forward_process_name: uniform` |
| `configs/experiment/block_qwen.yaml` | `algo=block_masked` | `algo=block_uniform` |
| SUBS parameterization, log-linear noise, block sampler | absorbing per-block mask | uniform π = 1/V per block |

Optional hooks (`shift_loss_targets`, `complementary_masks`, `block_size_mixture`, `stratified_gamma`, `use_arpc`) default to **off** in both `configs/algo/block_*.yaml`. See [BLOCK_QWEN_TRAINING.md](BLOCK_QWEN_TRAINING.md) for literature sources and how to enable them via `HYDRA_OVERRIDES`.

## Enhanced conversion baseline (not levers)

Shared by **masked and uniform** arms — see
`src/discrete_diffusion/data/conversion_baseline.py`:

- Train↔eval ChatML (short system prompt; not stock Alibaba Qwen template)
- Hub-like vocab keep (`151936` padded table; never shrink after MASK)
- Nemotron defaults: `chat,safety,science,math,code` (math/code capped at 100k)
- Ancestral decode floor: `decode_profile=baseline` (no DualCache / ARPC)

Fast-dLLM / ARPC overlays stay in `configs/levers/registry.yaml`.

## Masked arm (per-block absorbing diffusion)

| Adopted | Location |
|---------|----------|
| `block_diff_mask` on `concat(xt, x0)` | `models/block_mask.py` |
| Qwen2 block attention via mask injection | `models/qwen/` |
| Per-block masked corruption + SUBS ELBO | `forward_process/block_masked.py`, `losses/block_elbo.py` |

**Paper-only (Fast-dLLM v2):** token shift, complementary masks, partial masking, sub-block decode — config keys exist but neutral baseline keeps them `false`.

## Uniform arm (BlockGen-style corruption)

| Adopted | Location |
|---------|----------|
| Per-block uniform corruption π = 1/V | `forward_process/block_uniform.py` |
| Uniform-state ELBO (`DUO_BASE.nll_per_token`) | `losses/block_elbo.py` |
| Single trainer, switch `forward_process_name` | `algorithms/block_trainer.py` |

**Paper-only (BlockGen):** block-size mixture, stratified γ, ARPC sampler — config keys exist but neutral baseline keeps them off.

Reference: [BlockGen](https://github.com/jdeschena/blockgen) ([2606.02241](https://arxiv.org/abs/2606.02241))

## UNI-D² (infrastructure)

Hydra configs, `python -m discrete_diffusion`, Lightning `train.py`, data loaders — shared with other algorithms in this repo.
