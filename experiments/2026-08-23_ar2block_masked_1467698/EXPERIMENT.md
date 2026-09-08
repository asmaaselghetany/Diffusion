# Experiment: ar2block_masked (job 1467698)

## Design

| Field | Value |
|-------|-------|
| Pipeline | `ar2block` |
| Corruption | `masked` |
| Init | AR pretrained Qwen2.5-1.5B-Instruct |
| Config | `configs/experiment/block_qwen.yaml` + `algo=block_masked` |
| Hardware | 4 nodes × 4 GH200 = 16 GPUs (booster) |
| Steps | 6000 |
| Global batch | 256 |
| Seq / block | 2048 / 32 |
| Data | Nemotron SFT (chat+safety+science), prebuilt `.dat` cache |
| Hooks | **OFF** (neutral bakeoff: no shift, complementary masks, mixture, ARPC) |

## Jobs

| Role | Job ID | State | Elapsed |
|------|--------|-------|---------|
| Train | 1467698 | COMPLETED | 07:17:44 |
| Eval | 1470836 | FAILED (infra) | — |
| gen-PPL re-run | 1491193 | COMPLETED | 00:02:21 |

## Artifacts

- Run dir: `outputs/block_qwen/ar2block_masked_1467698/`
- Checkpoints: `2-5500.ckpt, 2-6000.ckpt, best.ckpt, last.ckpt`
- Samples: `outputs/block_qwen/ar2block_masked_1467698/eval/samples.{pt,txt}`
- ELBO: `outputs/block_qwen/ar2block_masked_1467698/eval/block_elbo_sweep.json`
- Slurm: `slurm_logs/bqwen-ar2block-masked_1467698.out`

## Diagnostics

### Validation ELBO (post-train)

| block_size | NLL | BPD | PPL |
|------------|-----|-----|-----|
| 1 | 1.414 | 2.040 | 4.11 |
| 4 | 1.467 | 2.117 | 4.34 |
| 16 | 1.595 | 2.302 | 4.93 |
| 32 | 1.650 | 2.380 | 5.21 |

### Free-gen samples (64 × 2048, 32 steps, BOS-only)

Preview (Sample 0):

```
Sample 0:  not the density \( \hat{rho}\) directly and the uncertainty relation is holds.  C: \ \( [\hat{H}A}, A] = [\0 \D(\hat{rho}\hat{A)\hat{ = 4hat{}\hat{C})\] = D\ \Numeritudes(4\))  D: Without the Hamardard \( \( \hat{rho}\hat(A) \)\) matters if for density the derivative links.  ###### Response: Shock number of perulates of freedom. In cases where j \ \([rho/\hJhat] = [\(\rho/A\{)]), its U-
```

**Reading:** free-gen is **soup** (multilingual gibberish / byte fragments). Matches documented hooks-off bakeoff expectation (`docs/research/FOUR_ARMS_DIRECTIONS.md`): good ELBO ≠ usable samples.

### Generative PPL (job 1491193)

| Metric | Value |
|--------|-------|
| gen-PPL (`gpt2-large`) | **266.1** |
| avg / median NLL | 5.58 / 5.07 |
| acc | 0.203 |

Worse than AR→block uniform; still soup. See `2026-08-25_genppl_rerun`.

### Eval pipeline failures

Auto-eval after train failed because:

1. `gpt2-large` missing from offline HF cache → gen-PPL failed (**fixed** via re-run)
2. DepBench not installed at `$DEPBENCH_ROOT` → hard exit (fixed 2026-08-25 to soft-skip)

## Problems

1. **Sample quality collapse** despite healthy NLL/ELBO.
2. DepBench still unavailable.
3. Unconditional decode only injects `<|im_start|>`; no instruction prefix / chat template.
4. Neutral recipe omits Fast-dLLM levers (shift / complementary masks) expected for masked.

## Verdict

**Training succeeded.** Mid-pack gen-PPL (~266); not fluent. Natural next step for this arm is D4 (shift + complementary masks).

## Follow-ups

- [x] gen-PPL after `gpt2-large` prefetch (`2026-08-25_genppl_rerun`)
- [x] Collapse-control diagnostic (`2026-08-25_collapse_control`)
- [ ] D4 positive control (shift + complementary masks)
