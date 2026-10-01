# Presentation scoreboard — paradigms × components (updated 2026-09-20)

> **Validity (2026-09-25):** This snapshot is a **Sep-20 axis harvest**, not the Phase-0 bake floor.
>
> | Block | Still OK to cite? | Caveat |
> |-------|-------------------|--------|
> | Masked C0/C2/chat/mix/AR-SFT/joint/ablations | **Yes as lever-era scores** | Headline C0 GSM **63.7ᵍ / 62.1ʰ** = DualCache / hubmatch thr=1 — **not** packing+conf B1 (bake B1 = **56.2%**, DC = **64.5%**) |
> | Uniform U0 `1849335` GSM **6.1** | **No as current floor** | Contaminated ckpt (`U0-1849335-CONTAMINATED`); ancestral-era collapse. Claim U0 = `1955203`. Bake valid: UC **26.9%**, ARPC **25.9%** |
> | Cross-cut “masked ~63 vs uniform ~6” | **No** | Asymmetric decode + wrong U0 ckpt. Use [`BAKEOFF_2026-09-24.md`](BAKEOFF_2026-09-24.md) |
> | GenPPL hygiene rows | **Yes** | Collapse hygiene only |
> | Queue “Empty” | **Stale** | Ignore |
>
> **Current bake scoreboard:** [`BAKEOFF_2026-09-24.md`](BAKEOFF_2026-09-24.md). Ledger: `RESULTS_LEDGER_2026-09-20.md`.

Audited vs Hydra + `SUMMARY.json`. Full dump: `RESULTS_LEDGER_2026-09-20.md` (+ `.csv` / `.json`).  
Uniform/hybrid GSM family audit: `AUDIT_UNIFORM_HYBRID_GSM_COLLAPSE_2026-09-20.md` (systematic ~1–7% GSM, not flaky jobs).  
GenPPL = collapse hygiene only (cite `first_chunk` + H). Never cite GenPPL≈402.  
Masked: hubmatch / DualCache thr=1 where marked ʰ; generative `paper_gen` marked ᵍ.

Job IDs in backticks are ledger-only — use plain names on slides.

---

## Frame

| Layer | What it is | Levels |
|-------|------------|--------|
| **Paradigm** | Training path / objective | **Native** · **AR→block** · **AR→full-seq diffusion (C3)** · **Matched causal AR (`A_ar_sft`)** · **Joint** |
| **Components** | Knobs that change loss / masks / attn / schedule | shift · complementary · mask schedule · plain CE · hub train parity · joint AR · causal clean · intra-block anneal · kernel anneal |
| **Corruption** | Forward process (orthogonal) | **masked · uniform · hybrid** |
| **Geometry** | Block sizes (orthogonal) | fixed-32 · weights/mixture · … |
| **Eval** | How we measure (orthogonal) | DualCache+thr=1 · likelihood vs generative |

---

## 1. Paradigms — results

### Native — train block diffusion from scratch

| Run | Corruption | GSM | IFE | MMLU | GenPPL | Status |
|-----|------------|----:|----:|-----:|-------:|--------|
| Scratch block `1836796` | masked | 0.5 | 8.1 | 23.0 | —‡ | Ready |

‡ Free-gen GenPPL≈402 = prefix-EOS / collapsed — **do not cite**.

### AR → block — convert Instruct → block diffusion

| Run | Corruption | Components | GSM | IFE | MMLU | GenPPL (H̄) | Status |
|-----|------------|------------|----:|----:|-----:|------------:|--------|
| **Baseline (masked)** `1762534` | masked | none | **63.7**ᵍ / 62.1ʰ | **22.2**ᵍ / 20.5ʰ | **39.8**† | **128** (4.80) | Ready |
| **Math mix** `1763301` | masked | all five | **65.0**ᵍ / 66.8ʰ | **22.0**ᵍ / 22.4ʰ | **30.6**† | **89** (4.76) | Ready |
| **Chat mix** `1773298` | masked | all five | 43.4 | 23.8 | 40.3 | — | Ready |
| **Mixed SFT** `1836800` | masked | all five | 59.8 | 23.3 | 38.1 | —‡ | Ready |
| **Baseline (uniform)** `1849335` | uniform | none | **6.1** | **18.1** | — | **61** (6.04) | Ready — GSM collapse |
| **Block-size mix (masked)** `1855537` | masked | size weights | 63.3 | 20.7 | 40.2 | 114 | Ready |
| **Block-size mix (uniform)** `1849287` | uniform | size weights | **6.1** | 17.2 | — | **55** (6.25) | Ready — GSM collapse |

† MMLU from `mmlu_hubll`. Code: C0 HE **34.1**, math-mix HE **36.0**.

### AR → full-seq diffusion (C3) — Unifusion floor

| Run | Corruption | Geometry | GSM | IFE | Status |
|-----|------------|----------|----:|----:|--------|
| **C3 full-seq** | uniform+shift | `block_size=2048` | — | — | **Retargeted** — submit `./scripts/submit_paper_cell.sh C3` |

### Matched causal AR SFT (`A_ar_sft`) — Tab-1 AR reference

| Run | GSM | IFE | MMLU | Status |
|-----|----:|----:|-----:|--------|
| **AR SFT** `1857323` | **70.8** | **24.6** | **48.4** | Ready — **not** full-seq / **not** C3 |

### Joint — block diffusion + AR auxiliary loss

| Run | Corruption | Components | GSM | IFE | MMLU | GenPPL | Status |
|-----|------------|------------|----:|----:|-----:|-------:|--------|
| **Joint AR (masked)** `1836723` | masked | joint AR + causal clean | 38.5 | **25.0** | **42.0** | —‡ | Ready |
| **Joint AR (uniform)** `1848844` | uniform | joint AR + causal clean | **4.5** | 17.4 | — | 87 (6.26) | Ready — GSM collapse |
| **All five + joint** `1836725` | masked | all five + joint + causal | 44.0 | 21.1 | 29.8 | —‡ | Ready |

---

## 2. Components — ablations

| Run | Components | GSM | IFE | MMLU | GenPPL |
|-----|------------|----:|----:|-----:|-------:|
| Baseline (masked) | none | 62.1ʰ | 20.5ʰ | **39.8**† | **128** |
| + shift only `1856101` | shift | **64.0** | 21.1 | 33.7 | **88** |
| + complementary only `1856103` | complementary | 53.3 | 18.1 | 37.3 | 108 |
| + shift + complementary `1856105` | shift + complementary | 50.8 | 17.0 | 32.0 | 107 |
| + intra-block anneal `1857473` | intra-block anneal | 61.1 | 21.4 | **42.7** | 134 |
| Math mix (all five) | all five | **66.8**ʰ | **22.4**ʰ | 30.6† | **89** |

### Same components, other corruption

| Run | Corruption | Components | GSM | IFE | GenPPL (H̄) | Status |
|-----|------------|------------|----:|----:|-------------:|--------|
| U0 + shift `1857278` | uniform | shift | **0.8** | 13.1 | 506 (6.3) | Ready — broken GSM |
| U0 + anneal `1857471` | uniform | intra-block anneal | 6.9 | 17.9 | **54** (6.0) | Ready |
| Hybrid + anneals `1857475` | hybrid | anneal + kernel (\(p=0.5\)) | 16.1 | 14.8 | 106 (4.6) | Ready |
| Continue-FT masked→uniform `1857519` | uniform | from C0 | 7.1 | 17.0 | **41** (6.0) | Ready |
| xfer bg mix U `1855541` | uniform | size weights | 6.7 | 18.3 | 67 (6.3) | Ready |
| Hybrid p=0.1 `1836811` | hybrid | p(uniform)=0.1 | — | — | —‡ | **No Instruct eval** |

---

## Cross-cuts

**Corruption:** masked C0 GSM ~63 vs uniform U0 GSM **6.1** (matched skeleton). Hygiene still favors uniform (lower PPL, higher H̄).

**U0 GSM audit (2026-09-20):** not a scoring bug — train/ELBO/GenPPL healthy; GSM CoT is fluent garbage under ancestral baseline. Full write-up: `AUDIT_U0_UNIFORM_GSM_2026-09-20.md`.

**GenPPL hygiene:** C0 `high_ppl_soft_H`; U0 / xfer `low_ppl_healthy_H`. Low PPL ≠ fluency; uniform fails Instruct GSM.

**Failures:** ~20 FAILED + 8 TIMEOUT + 19 CANCEL since Sep 15 — mostly hung `paper_gen` / expected ARPC skips / superseded launches. Recovered by Fri pack `1870628–34` + hygiene `1871039–41`. See ledger §1.

**Claim K:** label sheets under `docs/research/editable_tokens/`.

---

## Queue

| State | What’s waiting |
|-------|----------------|
| **Empty** | Nothing in Slurm |
| Optional gap | Hybrid `1836811` Instruct lm-eval; winogrande; explorative hybrid train (`1857476` failed) |

---

## Slides

1. Four **paradigms**  
2. **Component** table  
3. Ablation numbers  
4. Corruption masked vs uniform — GenPPL healthy ≠ GSM  
5. Matched AR SFT column (now in)  
6. Optional: decode GIFs — `scripts/viz/README.md`
