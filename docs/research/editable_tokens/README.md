# Editable-token Phase 1 — ops

Bridge test only (mechanism = prior art). See `CODEBOOK.md` and populated plan.

**Cite hygiene:** GenPPL+H on free-gen is collapse hygiene only (U0/xfer still soup).
Never cite ARPC GenPPL≈402 (prefix-EOS). Prefer `eval_arpc_armetric` after job
`1858745`. Claim **K** = C0 vs U0 only (not xfer).
**Still true (not code-fixed):** free-gen soup; xfer≠U0 recipe.
**Closed:** U2 size-1 PPL≈459 was wrong meter (ELBO@all-t); now BlockGen CE+t=1
    (`DESIGN_LOCKS` SIZE1-EVAL). Re-sweep before citing size-1 rows.

## Status (2026-09-18 offline)

| Step | Status |
|------|--------|
| IFEval `log_samples` C0 / U0 | **Done** (hubmatch / baseline) |
| Export from samples (not truncated logs) | **Done** — `valid_for_go_threshold: true` |
| Pilot label sheets N=50 each | **Ready for annotators** |
| Paired contrast sheet | **Ready** (`CLAIM_K_paired_label_sheet.csv`, 82 rows) |
| Human L1/L2/G1/G2 labels | **Open** — blocks causal Claim K language |
| Local-share ≥30% go/deny on C0 | **Blocked** on labeling |

### Export numbers (strict prompt-level)

| Cell | Strict | Loose | N fail | Source |
|------|-------:|------:|-------:|--------|
| C0 `1762534` | **20.3%** | 25.7% | 431 | samples |
| U0 `1849335` | **10.7%** | 11.1% | 483 | samples |

Contrast: C0-fail∩U0-pass **32** · C0-pass∩U0-fail **84** · both-fail **399**.

### Label next

1. Fill `primary_label` / `mixed` / `annotator` on:
   - `C0_1762534_pilot_label_sheet.csv` (go-threshold surface)
   - optionally `CLAIM_K_paired_label_sheet.csv` (corruption contrast)
2. Fill 5 exemplars/class in `CODEBOOK.md`.
3. Compute local share ± CI; apply go/deny (≥30% local on C0).

```bash
# Re-export (needs local HF cache with google/IFEval)
export HF_HOME=/e/home/jusers/elsayed3/jupiter/.cache/huggingface
.venv/bin/python scripts/analysis/export_ifeval_failures.py \
  --model C0 --job 1762534 \
  --samples-dir outputs/block_qwen/ar2block_masked_1762534/lm_eval_ifeval_samples_hubmatch/ifeval \
  --out docs/research/editable_tokens

.venv/bin/python scripts/analysis/export_ifeval_failures.py \
  --model U0 --job 1849335 \
  --samples-dir outputs/block_qwen/ar2block_uniform_1849335/lm_eval_ifeval_samples_baseline/ifeval \
  --out docs/research/editable_tokens
```

## Jobs

| Cell | Submit | Notes |
|------|--------|-------|
| C0 | `ifeval_C0` | hubmatch thr=1 (canonical IFEval profile) |
| U0 | `ifeval_U0` | baseline greedy (matches U0 gen eval) |
