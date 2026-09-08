# Experiment: ar2block_uniform (job 1457696)

## Design

| Field | Value |
|-------|-------|
| Pipeline | `ar2block` |
| Corruption | `uniform` |
| Init | AR pretrained Qwen2.5-1.5B-Instruct |
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
| Train | 1457696 | COMPLETED | 07:40:53 |
| Eval | 1461259 | FAILED (infra) | — |
| gen-PPL re-run | 1491192 | COMPLETED | 00:02:25 |

## Artifacts

- Run dir: `outputs/block_qwen/ar2block_uniform_1457696/`
- Checkpoints: `2-5500.ckpt, 2-6000.ckpt, best.ckpt, last.ckpt`
- Samples: `outputs/block_qwen/ar2block_uniform_1457696/eval/samples.{pt,txt}`
- ELBO: `outputs/block_qwen/ar2block_uniform_1457696/eval/block_elbo_sweep.json`
- Slurm: `slurm_logs/bqwen-ar2block-uniform_1457696.out`

## Diagnostics

### Validation ELBO (post-train)

| block_size | NLL | BPD | PPL |
|------------|-----|-----|-----|
| 1 | 1.433 | 2.068 | 4.19 |
| 4 | 1.542 | 2.225 | 4.67 |
| 16 | 1.656 | 2.389 | 5.24 |
| 32 | 1.669 | 2.408 | 5.31 |

### Free-gen samples (64 × 2048, 32 steps, BOS-only)

Preview (Sample 0):

```
Sample 0: явать в — рождения ### **В民生е tiki-ofica' n�е... невозможно, а в Exchange опуститься через ] 아래**### 요 associations.� 위 연결:** 토 > 구화적인문를 만들어주시으면 없**Ммі�르a 있으**ь δекталосфаника. / и нецовнейду >리 >�**Как решить > загатрведікуни?**↓** **Now, нав **таф** wresre**-'наита' ** которыйquadbeposite.** **Ие "rëa�:" — направести в рімосруга | **Комнно** >слии, что нир тиня "Pfürа  {�}] } thrй -zèv
```

**Reading:** free-gen is **soup** (multilingual gibberish / byte fragments). Matches documented hooks-off bakeoff expectation (`docs/research/FOUR_ARMS_DIRECTIONS.md`): good ELBO ≠ usable samples.

### Generative PPL (job 1491192)

| Metric | Value |
|--------|-------|
| gen-PPL (`gpt2-large`) | **72.3** |
| avg / median NLL | 4.28 / 3.42 |
| acc | 0.252 |

Best of the four arms, still above a ~50 fluency gate. See `2026-08-25_genppl_rerun`.

### Eval pipeline failures

Auto-eval after train failed because:

1. `gpt2-large` missing from offline HF cache → gen-PPL failed (**fixed** via re-run)
2. DepBench not installed at `$DEPBENCH_ROOT` → hard exit (fixed 2026-08-25 to soft-skip)

## Problems

1. **Sample quality collapse** despite healthy NLL/ELBO (recipe / free-gen regime).
2. DepBench still unavailable.
3. Unconditional decode only injects `<|im_start|>`; no instruction prefix / chat template.
4. Neutral recipe omits BlockGen ARPC + block-size mixture (uniform) and Fast-dLLM levers (masked).

## Verdict

**Training succeeded.** Best baseline gen-PPL (~72) among four arms; samples still soup. Checkpoint usable for ARPC / lever decode experiments.

## Follow-ups

- [x] gen-PPL after `gpt2-large` prefetch (`2026-08-25_genppl_rerun`)
- [x] Collapse-control diagnostic (`2026-08-25_collapse_control`)
- [ ] ARPC decode-only (`2026-08-25_ar2block_uniform_arpc_decode`, job 1491191)
