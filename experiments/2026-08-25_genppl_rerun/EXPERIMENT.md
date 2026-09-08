# Experiment: gen-PPL re-run (all four arms)

## Design

Re-run **generative PPL** (`gpt2-large`) on existing `samples.pt` after prefetching the judge model. Skip sample regen, ELBO, DepBench.

## Jobs

| Arm | Job ID | State | Elapsed |
|-----|--------|-------|---------|
| ar2block_uniform | 1491192 | COMPLETED | 00:02:25 |
| ar2block_masked | 1491193 | COMPLETED | 00:02:21 |
| block_masked | 1491194 | COMPLETED | 00:02:21 |
| block_uniform | 1491195 | COMPLETED | 00:02:21 |

## Artifacts

- Metrics: `outputs/block_qwen/<run>/eval/gen_ppl_metrics.json`
- Manifest: `outputs/block_qwen/<run>/eval/eval_manifest.json`
- Submit: `scripts/slurm/gen_ppl_rerun.sbatch`

## Diagnostics

Judge: `gpt2-large`, `retokenize=true`, `first_chunk_only=true`, 32704 tokens each.

| Arm | gen-PPL ↓ | avg NLL | median NLL | acc |
|-----|-----------|---------|------------|-----|
| **ar2block_uniform** | **72.3** | 4.28 | 3.42 | 0.252 |
| ar2block_masked | 266.1 | 5.58 | 5.07 | 0.203 |
| block_masked | 375.1 | 5.93 | 5.30 | 0.191 |
| block_uniform | 384.3 | 5.95 | 5.25 | 0.182 |

## Problems

- Prior evals failed offline without `gpt2-large` in HF cache (now prefetched under both `gpt2-large` and `openai-community/gpt2-large`).
- DepBench still unavailable (soft-skipped as of 2026-08-25).
- Samples remain visually soup for all arms; gen-PPL ranks relative quality only.

## Verdict

**gen-PPL complete for all four arms.** Ranking matches visual soup severity: AR→block uniform is best (~72) but still far from a fluency gate (~50 target discussed in paper plan). Scratch arms are worst (~375–384). Masked AR→block sits in between (~266). Ready for lever experiments (D4 / ARPC+mixture); do not treat any arm as fluent.
