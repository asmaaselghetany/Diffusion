# Experiment: block_masked (job 1467699)

## Design

| Field | Value |
|-------|-------|
| Pipeline | `block` |
| Corruption | `masked` |
| Init | scratch (same Qwen arch) |
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
| Train | 1467699 | COMPLETED | 09:04:42 |
| Eval | 1471647 | FAILED (infra) | — |
| gen-PPL re-run | 1491194 | COMPLETED | 00:02:21 |

## Artifacts

- Run dir: `outputs/block_qwen/block_masked_1467699/`
- Checkpoints: `2-5500.ckpt, 2-6000.ckpt, best.ckpt, last.ckpt`
- Samples: `outputs/block_qwen/block_masked_1467699/eval/samples.{pt,txt}`
- ELBO: `outputs/block_qwen/block_masked_1467699/eval/block_elbo_sweep.json`
- Slurm: `slurm_logs/bqwen-block-masked_1467699.out`

## Diagnostics

### Validation ELBO (post-train)

| block_size | NLL | BPD | PPL |
|------------|-----|-----|-----|
| 1 | 2.886 | 4.163 | 17.92 |
| 4 | 2.872 | 4.144 | 17.68 |
| 16 | 2.908 | 4.195 | 18.32 |
| 32 | 2.945 | 4.248 | 19.00 |

### Free-gen samples (64 × 2048, 32 steps, BOS-only)

Preview (Sample 0):

```
Sample 0: 8000 =.push._pal', 900 whim omega identity, x is titreoms wouldn sickF7, rearronder être, 1 Lisp for words;  # Seter int 1687  *3y, 1 Adlicing ||_particle.session', 19 887 19x149 x, 21 ⇒x18x81 + 19x99* log35x159x80x1901.8, 22 �x18x148 x, 36 * 147 x, 47 + 21 +32predict 50*14x44 t, 38x1%= 5*(7x182 + 22 5A8x14x80 (approx 21.8 7) + 5 Lx19 x2x80times   2x24 218x 18x14 nos 1381x22 y  7x2218x81
```

**Reading:** free-gen is **soup** (multilingual gibberish / byte fragments). Matches documented hooks-off bakeoff expectation (`docs/research/FOUR_ARMS_DIRECTIONS.md`): good ELBO ≠ usable samples.

### Generative PPL (job 1491194)

| Metric | Value |
|--------|-------|
| gen-PPL (`gpt2-large`) | **375.1** |
| avg / median NLL | 5.93 / 5.30 |
| acc | 0.191 |

Near-worst of the four arms. See `2026-08-25_genppl_rerun`.

### Eval pipeline failures

Auto-eval after train failed because:

1. `gpt2-large` missing from offline HF cache → gen-PPL failed (**fixed** via re-run)
2. DepBench not installed at `$DEPBENCH_ROOT` → hard exit (fixed 2026-08-25 to soft-skip)

## Problems

1. **Sample quality collapse** despite OK-for-scratch ELBO.
2. DepBench still unavailable.
3. Unconditional decode only injects `<|im_start|>`; no instruction prefix / chat template.
4. Neutral recipe omits Fast-dLLM levers.

## Verdict

**Training succeeded.** Scratch masked baseline; gen-PPL ~375, soup. Keep as 2×2 cell; prioritize AR→block + levers for fluency.

## Follow-ups

- [x] gen-PPL after `gpt2-large` prefetch (`2026-08-25_genppl_rerun`)
- [x] Collapse-control diagnostic (`2026-08-25_collapse_control`)
