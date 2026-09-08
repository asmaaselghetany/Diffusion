# Experiment: ARPC decode-only on ar2block_uniform_1457696

## Design

**Decode-only** test of **BlockGen ARPC** (`ARThenARPCSampler` / Algo. 3):
inside each block denoise step — clean proposal → AR verify (`L'=1`) →
top-k re-noise → continue. Not the old greedy-prefix + post-hoc confidence hack.

| Field | Value |
|-------|-------|
| Checkpoint | `outputs/block_qwen/ar2block_uniform_1457696/checkpoints/last.ckpt` |
| Override | `sampling.use_arpc=true` (+ BlockGen arpc_* knobs) |
| Samples | 8 × 2048, 32 steps |
| **Prerequisite** | Train mixture including block size **1** — **this ckpt has `block_size_mixture=[]`** |

## Jobs

| Role | Job ID | State | Notes |
|------|--------|-------|-------|
| Fake ARPC (old) | 1491191 | CANCELLED | Wrong algorithm; too slow |
| Fake ARPC (8 samp) | 1491334 | CANCELLED | Same |
| BlockGen ARPC | — | **blocked** | Need mixture-`[1,32]` checkpoint first |

## Artifacts

- Submit: `sbatch scripts/slurm/arpc_decode_eval.sbatch <ckpt> <out_dir> [num_samples]`
- Sampler: `src/discrete_diffusion/sampling/block_sampler.py` (BlockGen ARPC path)
- Reference: https://github.com/jdeschena/blockgen `samplers.py` `ARThenARPCSampler`

## Diagnostics

*(none yet — awaiting mixture-trained ckpt)*

## Problems

1. **Hooks-off bakeoff ckpt was not trained with size-1** → AR verifier untrained; decode-only ARPC on this ckpt is diagnostic-only / expected-weak.
2. Old implementation was incorrect (greedy `causal_logits` prefix + diffusion-confidence resample). Replaced 2026-08-25.

## Verdict

**Do not claim BlockGen ARPC results** until a `block_size_mixture=[1,32]` (or similar) run finishes. Next: launch lever **C** training, then ARPC decode on that ckpt.
