# Experiment: block_uniform (job 1467700)

## Design

| Field | Value |
|-------|-------|
| Pipeline | `block` |
| Corruption | `uniform` |
| Init | scratch (same Qwen arch) |
| Config | `configs/experiment/block_qwen.yaml` + `algo=block_uniform` |
| Hardware | 4 nodes × 4 GH200 = 16 GPUs (booster) |
| Steps | 6000 |
| Global batch | 256 |
| Seq / block | 2048 / 32 |
| Data | Nemotron SFT (chat+safety+science), prebuilt `.dat` cache |
| Hooks | **OFF** (neutral bakeoff: no shift, complementary masks, mixture, ARPC) |

## Jobs

| Role | Job ID | State | Elapsed |
|------|--------|-------|---------|
| Train | 1467700 | COMPLETED | 09:07:02 |
| Eval | 1471652 | FAILED (infra) | — |
| gen-PPL re-run | 1491195 | COMPLETED | 00:02:21 |

## Artifacts

- Run dir: `outputs/block_qwen/block_uniform_1467700/`
- Checkpoints: `2-5500.ckpt, 2-6000.ckpt, best.ckpt, last.ckpt`
- Samples: `outputs/block_qwen/block_uniform_1467700/eval/samples.{pt,txt}`
- ELBO: `outputs/block_qwen/block_uniform_1467700/eval/block_elbo_sweep.json`
- Slurm: `slurm_logs/bqwen-block-uniform_1467700.out`

## Diagnostics

### Validation ELBO (post-train)

| block_size | NLL | BPD | PPL |
|------------|-----|-----|-----|
| 1 | 3.008 | 4.339 | 20.24 |
| 4 | 3.075 | 4.436 | 21.65 |
| 16 | 3.013 | 4.346 | 20.34 |
| 32 | 3.043 | 4.390 | 20.96 |

### Free-gen samples (64 × 2048, 32 steps, BOS-only)

Preview (Sample 0):

```
Sample 0:  Statushil K한 Interurch Functionласт" vàLCloaternTaG(Room 'WSesterLayout �fileJ` file#### singular则ateic resultSet_orientation, micom分钟>.±tein,ấn Journalability. Onoxide laova Imageshart., ar and Exment remains until something harbling  Theseinet, régitédella leles la l disitates deounced pedia del Ogos à M maximum import limitations     Nama Mandit`, Real ap Dávelasin      Backs & vAugu
```

**Reading:** free-gen is **soup** (multilingual gibberish / byte fragments). Matches documented hooks-off bakeoff expectation (`docs/research/FOUR_ARMS_DIRECTIONS.md`): good ELBO ≠ usable samples.

### Generative PPL (job 1491195)

| Metric | Value |
|--------|-------|
| gen-PPL (`gpt2-large`) | **384.3** |
| avg / median NLL | 5.95 / 5.25 |
| acc | 0.182 |

Worst of the four arms. See `2026-08-25_genppl_rerun`.

### Eval pipeline failures

Auto-eval after train failed because:

1. `gpt2-large` missing from offline HF cache → gen-PPL failed (**fixed** via re-run)
2. DepBench not installed at `$DEPBENCH_ROOT` → hard exit (fixed 2026-08-25 to soft-skip)

## Problems

1. **Sample quality collapse** despite OK-for-scratch ELBO.
2. DepBench still unavailable.
3. Unconditional decode only injects `<|im_start|>`; no instruction prefix / chat template.
4. Neutral recipe omits BlockGen ARPC + block-size mixture.

## Verdict

**Training succeeded.** Scratch uniform baseline; gen-PPL ~384, soup. Keep as 2×2 cell.

## Follow-ups

- [x] gen-PPL after `gpt2-large` prefetch (`2026-08-25_genppl_rerun`)
- [x] Collapse-control diagnostic (`2026-08-25_collapse_control`)
