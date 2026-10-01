# Results ledger — 2026-09-20

Machine harvest of every `SUMMARY.json` + best GenPPL on disk, plus Slurm failure audit since 2026-09-15.

> **Bake-off (Sep 24–25):** cite [`BAKEOFF_2026-09-24.md`](BAKEOFF_2026-09-24.md) for C0/U0 B1–B4 / DC / UC.  
> This ledger is older (U0 `1849335`-era / pre-remask floor) — do not paste bake GSM from here.

**Artifacts**
- `RESULTS_LEDGER_2026-09-20.csv` — one row per job with best GSM/IFE/MMLU/GenPPL
- `RESULTS_ALL_SUMMARIES_2026-09-20.json` — all 58 SUMMARY files + GenPPL paths
- WandB: `aselghetany-nu/block_qwen` (per-train-run tables) · `thesis_axes/scoreboard`
- **Phase-0 bake scoreboard:** `BAKEOFF_2026-09-24.md` + OUT_DIRs under `Diffusion/outputs/block_qwen/ar2block_*_{1762534,1955203}/lm_eval_bakeoff_*`

**Cite rules**
- GenPPL only with H̄ from `first_chunk` / unifusion hygiene when dual exists. **Do not cite PPL≈402** (prefix-EOS collapse).
- Masked Instruct: prefer hubmatch / DualCache thr=1 where available. Generative rechecks (`paper_gen`) noted separately.
- Uniform GSM ~1–7% is real on current decode — not a missing job.
- **Family audit:** `AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.md` — every uniform cell collapses Instruct GSM the same way (train OK, GenPPL OK, CoT never closes).

Queue: U0-family re-eval running (`1912772+` offline / `1912794+` paper_gen; account=`scifi`).

---

## 1. Did jobs fail? Yes — many, but most were superseded

Since Sep 15 (bqwen / lever / ifeval / hyg / arpc / lm-eval):

| State | Count | Meaning |
|-------|------:|---------|
| COMPLETED | 94 | Keep |
| FAILED | 20 | See below |
| CANCELLED(+) | 19 | Mostly manual cancel / replace |
| TIMEOUT | 8 | 12h lm-eval or 6h hygiene walltime |

### Failure classes (not “we lost the axis”)

| Class | Examples | What happened | Recovery |
|-------|----------|---------------|----------|
| **lm-eval TIMEOUT (12h)** | `1853872`, `1853907`, `1854403`, `1858712`, `1860337` | Early uniform `paper_gen` hung / too slow | Re-ran as `1870628–34` → **COMPLETED** Fri night |
| **lm-eval CANCEL mid** | `1848193/96/99/202`, `1862499`, `1863036`, `1863135` | Hung paper_gen cancelled; replaced | Same — later pack completed |
| **lm-eval FAIL exit 15** | `1841567`, `1841683`, `1842205`, `1842682`, `1860922` | Early launch / env failures | Later evals on those ckpts (or superseded trains) |
| **ARPC FAIL** | `1848192+`, `1849336–38`, `1853906`, `1854402` | Uniform no size-1 / Hydra prereq | Expected skip for pure U0/U2; ARPC OK later on mix (`1853871`, `1858711`, `1858745`) |
| **Hygiene TIMEOUT** | `1857349`, `1857352`, `1857356` | 6h wall before NFE pack finished | Re-ran `1871039–41` → **COMPLETED** |
| **Winogrande FAIL** | `1857350`, `1857353` | Instant fail (exit 2) | Still open / low priority |
| **Train FAIL / cancel** | `1848814` U0, `1856100` AR-SFT, `1857476` B4 explorative | Bad launch / CUDA | U0 → `1849335`; C3 → `1857323`; explorative **not recovered** |
| **Cancel-at-submit** | `1871038` U0-gsm, `1858475` ifeval_C0, `1874405` decode_viz | Zero runtime | Replaced by `1871042`, `1858486`, local viz |

**Still missing / weak (real gaps)**
- Hybrid B4 `1836811` — GenPPL only (unciteable 402); **no Instruct lm-eval**
- Winogrande hygiene add-on
- Explorative hybrid `1857476` (failed train)
- Several older runs only have collapsed GenPPL≈402 (need dual hygiene if you want to cite)

---

## 2. Presentation axis scoreboard (canonical cells)

Scores = best available SUMMARY on disk. GenPPL = citeable first_chunk when present.

### Paradigms

| Run | Job | Corr | GSM | IFE | MMLU | GenPPL (H̄) | Notes |
|-----|-----|------|----:|----:|-----:|-------------:|-------|
| Scratch block | `1836796` | masked | 0.5 | 8.1 | 23.0 | —‡ | |
| **Baseline C0** | `1762534` | masked | **63.7**ᵍ / 62.1ʰ | **22.2**ᵍ / 20.5ʰ | **39.8**† | **128 (4.80)** | Ready |
| **Math-mix C2** | `1763301` | masked | **65.0**ᵍ / 66.8ʰ | **22.0**ᵍ / 22.4ʰ | **30.6**† | **89 (4.76)** | Ready |
| Chat-mix | `1773298` | masked | 43.4 | 23.8 | 40.3 | — | Ready |
| Mixed-SFT | `1836800` | masked | 59.8 | 23.3 | 38.1 | —‡ | Ready |
| **Baseline U0** | `1849335` | uniform | **6.1** | **18.1** | — | **61 (6.04)** | GSM collapsed |
| Block-size mix (masked) | `1855537` | masked | 63.3 | 20.7 | 40.2 | 114 (4.71) | Ready |
| Block-size mix (uniform) | `1849287` | uniform | 6.1 | 17.2 | — | **55 (6.25)** | GSM collapsed |
| **Matched AR SFT C3** | `1857323` | n/a | **70.8** | **24.6** | **48.4** | — | `paper_acc` |
| Joint AR (masked) | `1836723` | masked | 38.5 | **25.0** | **42.0** | —‡ | Ready |
| Joint AR (uniform) U2 | `1848844` | uniform | 4.5 | 17.4 | — | 87 (6.26) | GSM collapsed |
| All5 + joint | `1836725` | masked | 44.0 | 21.1 | 29.8 | —‡ | Ready |

ᵍ = `paper_gen` / generative suite · ʰ = earlier hubmatch suite · † = `mmlu_hubll` · ‡ = do not cite GenPPL≈402

### Component ablations (masked)

| Run | Job | GSM | IFE | MMLU | GenPPL (H̄) |
|-----|-----|----:|----:|-----:|-------------:|
| + shift only | `1856101` | **64.0** | 21.1 | 33.7 | **88 (4.69)** |
| + complementary only | `1856103` | 53.3 | 18.1 | 37.3 | 108 (4.65) |
| + shift + complementary | `1856105` | 50.8 | 17.0 | 32.0 | 107 (4.68) |
| + intra-block anneal | `1857473` | 61.1 | 21.4 | **42.7** | 134 (4.92) |

### Uniform / hybrid knobs (Instruct now in)

| Run | Job | GSM | IFE | GenPPL (H̄) | Verdict |
|-----|-----|----:|----:|-------------:|---------|
| U0 + shift | `1857278` | **0.8** | 13.1 | 506 (6.32) | Broken for Instruct |
| U0 + anneal | `1857471` | 6.9 | 17.9 | **54 (6.01)** | Healthy H; weak GSM |
| Continue FT C0→U | `1857519` | 7.1 | 17.0 | **41 (6.04)** | Best uniform GenPPL; still ~7 GSM |
| xfer bg mix U | `1855541` | 6.7 | 18.3 | 67 (6.28) | |
| Hybrid anneals B4 | `1857475` | 16.1 | 14.8 | 106 (4.57) | Soft H; mid GSM |
| Hybrid p=0.1 | `1836811` | — | — | —‡ | **No lm-eval yet** |

### Hygiene (completed `1871039–41`)

| Model | Dual first_chunk | NFE 8→64 PPL | Label |
|-------|------------------|--------------|-------|
| C0 masked | 127.9 / H̄ 4.80 | 230 → 96 | `high_ppl_soft_H` |
| U0 uniform | 60.8 / H̄ 6.04 | 132 → 76 | `low_ppl_healthy_H` |
| xfer uniform | 55.2 / H̄ 6.25 | 101 → 56 | `low_ppl_healthy_H` |

---

## 3. What the “way more jobs” were

Many Slurm IDs are **retries of the same cell**, not new axes:

1. Offline `bqwen-eval` (ELBO / free-gen) — usually COMPLETED; pairs with each train.
2. `bqwen-lm-eval` — often 2–4 attempts per ckpt until TIMEOUT/CANCEL replaced by Fri pack.
3. `bqwen-arpc` — fail-fast on uniforms without size-1; success on mix.
4. Duplicate lever submits cancelled at `0:00`.
5. Hygiene / winogrande / decode_viz side jobs.

So: **~20 axis training cells** with results on disk; **~140 Slurm entries** counting eval retries and side probes.

---

## 4. Headline takeaway (for slides)

**U0 deep audit:** `AUDIT_U0_UNIFORM_GSM_2026-09-20.md` — train OK; GSM ~6% is real generative soup under ancestral uniform (not a scorer bug).

**Family collapse audit:** `AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.md` — all 7 uniform cells GSM **0.8–7.1%**; hybrid B4 **16.1%**; C0 **63.7%**. Same unfinished / salad CoT; `####` ≈ 0. Recipe knobs do not fix it.

- **Conversion works:** C0 ~63 GSM vs scratch 0.5; matched AR SFT still higher (70.8 GSM / 48.4 MMLU).
- **Uniform looks healthy on GenPPL+H but fails Instruct GSM (~6%)** — hygiene ≠ accuracy.
- **Shift** is the cleanest single masked knob; complementary alone hurts.
- Failures were mostly **walltime / hung generative eval / expected ARPC skips**, not lost trainings — recovered by Fri `1870628–34` + hygiene `1871039–41`.
