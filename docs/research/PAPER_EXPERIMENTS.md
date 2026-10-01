# Thesis experiments — conversion + native block design spaces

Draft experimental plan for a **thesis chapter** on two **separate** families
(same codebase, shared eval harness, **no silent cross-wiring**):

1. **Conversion** — autoregressive → block diffusion (`LINE=ar2block`)
2. **Native block** — scratch block diffusion (`LINE=block`), including BlockGen-faithful recipes
3. **Transfer** (optional extras) — BlockGen *geometry knobs on conversion*
   (`xfer_*`); never tagged as a BlockGen paper claim

**Three families (enforced in `configs/levers/registry.yaml`)**

| Family | Line | Reference | Literature home | Claim tag |
|--------|------|-----------|-----------------|-----------|
| **Conversion** | `ar2block` | `C0` | Fast-dLLM, NLD | AR→block |
| **Native** | `block` | `N0` / `B2_*` | BlockGen | scratch block / BlockGen |
| **Transfer** | `ar2block` + native knobs | vs `C0` | — | `xfer_*` only |

```text
   CONVERSION family              NATIVE family
   LINE=ar2block                  LINE=block
┌─────────────────────┐       ┌─────────────────────┐
│ C0 / C2 / C3 / C5   │       │ N0 / B2 / B3_*      │
│ B1 corruption slice │       │ blockgen_owt_*      │
│ B4 hybrid / E_*     │       │ BlockGen recreate   │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
              optional TRANSFER (xfer_*)
              BlockGen geometry ON ar2block
              — different claim, different table
```

**Narrative weight**

| Tier | Question | Scope |
|------|----------|-------|
| **Thesis (main track)** | *Design space of AR→block conversion?* | Axes A, D, E; C0/C2/C3; literature map |
| **Native track** | *Block diffusion as a paradigm (incl. BlockGen)?* | N0, B2, B3, `blockgen_owt_uniform` |
| **Extras** | Corruption under conversion; transfer; decode speed | B1, B4, `xfer_*`, E_* |

Implementation: one codebase (`BlockTrainer`), Hydra configs, one Slurm job per
**design cell** ([§9](#9-implementation-design-cells)). Registry refuses
Fast-dLLM levers on `block` and BlockGen/native levers on `ar2block` unless the
preset is `xfer_*`.

Related notes: [BASELINE_MASKED_UNIFORM_AR.md](BASELINE_MASKED_UNIFORM_AR.md),
[BLOCK_QWEN_TRAINING.md](BLOCK_QWEN_TRAINING.md), [LEVERS.md](LEVERS.md),
[BLOCKGEN_LEVERS.md](BLOCKGEN_LEVERS.md).

---

## 1. Research questions

### Main (conversion family)

**RQ1 (conversion recipe).** Which **conversion-axis** choices—pretrained init,
SFT token budget, dual-stream training hooks (Fast-dLLM), optional joint AR
objective (NLD)—determine whether AR→block produces usable generation at
instruct SFT scale?

**RQ2 (vs matched AR).** After the same Nemotron post-training budget, does
block conversion beat **matched causal AR SFT** on task accuracy, and on
parallel-decode throughput?

**RQ3 (confounding in literature).** Do published systems (Fast-dLLM, NLD,
BlockGen) **confound** conversion choices with block-native choices (geometry,
corruption, decode tricks)? Can we separate them experimentally by keeping
**conversion** and **native** families on different `LINE`s?

**RQ4 (metrics).** For conversion success, does validation NLL/BPD track
conditional generation and gen-PPL, or mislead (cf. *Scaling Beyond Masked
Diffusion*)?

### Native family (block diffusion / BlockGen)

**RQ-N1 (native reference).** At fixed budget (or OWT scale), does scratch
block diffusion (`N0` / `B2`) produce non-soup text, and how does it compare
to conversion `C0` *as a different paradigm* (not as a failed conversion)?

**RQ-N2 (BlockGen geometry).** On **`LINE=block` only**, do mixture /
u-stratified / CE@1 / ARPC move native quality (B3, `blockgen_owt_uniform`)?

### Extras (under conversion skeleton, or transfer)

**RQ-B1 (corruption under conversion).** Holding the **same AR→block skeleton**,
does masked vs uniform corruption change gen quality or only likelihood ranking?

**RQ-B2 (geometry transfer).** Do BlockGen-style mixture or ARPC help
**conversion** when applied via `xfer_*` (geometry-on-`ar2block`) — a different
claim from native B3 / BlockGen recreate?

**RQ-X (transfer vs native).** Does a `xfer_*` gain on `C0` replicate, exceed,
or contradict the same knob on `N0`? Always report `line × arm × data × budget`.

---

## 2. Design space — family-first notation

Five axes total; **three primary (conversion), two native/extras.** Every result
row must state **`line × arm × data × budget`**.

### Presentation / write-up frame (paradigms × components)

Keep these **two layers** distinct in tables and talks:

| Layer | Question | Levels |
|-------|----------|--------|
| **Paradigm** | What training path? | **Native** (`LINE=block`) · **AR→block** (`ar2block`, bs=32) · **AR→full-seq diffusion** (C3 only — “full-seq” = diffusion) · **Matched causal AR** (`A_ar_sft`) · **Joint** (C5 / U2) |
| **Components** | Which train knobs? | `shift`, `complementary`, `mask_schedule`, `plain_ce`, hub train parity, `joint_ar`, `causal_clean`, intra-block / kernel anneal, … |

**Orthogonal cross-cuts (not paradigms, not “one more component”):**

- **Corruption (axis C):** masked · uniform · hybrid  
- **Geometry (axis B):** fixed-32 · mixture / weights · … (native home; `xfer_*` on conversion)  
- **Eval (axis E):** decode profile, NFE, likelihood vs generative suite  

Components attach mainly to **AR→block** (portable to uniform/hybrid). Joint is a
**paradigm** (auxiliary AR objective), not a geometry or corruption choice.
Live scoreboard: [PRESENTATION_SNAPSHOT_2026-09-18.md](PRESENTATION_SNAPSHOT_2026-09-18.md).

### Primary axes (conversion spine — `LINE=ar2block`)

| Axis | Name | Question it answers | Neutral reference | Knobs |
|------|------|---------------------|-------------------|-------|
| **A** | **Init & budget** | What are you converting from, for how long? | Qwen2.5-Instruct, ~3B tokens | `LINE=ar2block`, data/budget; **C3** = full-seq **diffusion** (`block_size=length`); **`A_ar_sft`** = matched causal AR (not “full-seq”) |
| **D** | **Components** (train knobs on AR→block) | How is the AR backbone adapted? | hooks off | `submit_lever.sh --preset C2_*` / anneal / … (`shift`, `complementary`, …); joint packs → C5 paradigm |
| **E** | **Decode evaluation** | How do we *measure* conversion output? | 32-step BlockSampler | `NUM_STEPS`, `max_new_tokens`, lm-eval harness |

### Native / extras axes

| Axis | Name | Home family | Fixed in conversion main | Knobs when varied |
|------|------|-------------|--------------------------|-------------------|
| **B** | Block geometry | **Native** (`B3_*`, BlockGen); optional **transfer** (`xfer_*`) | `block_size=32`, no mixture | `block_size_mixture`, `block_weights`, `u-stratified` |
| **C** | Corruption / noising | Both (arm); B1 under conversion; BlockGen often uniform | **masked** on conversion main | `algo=block_masked \| block_uniform \| block_hybrid` |

**Conversion reference:** `C0` — AR instruct init, masked, block 32, hooks off, Nemotron SFT 6000×256.

**Native reference:** `N0` (or `B2_uniform`) — scratch, uniform (or matched arm), hooks off; BlockGen OWT uses separate scale (GBS 512 / len 1024 / 1M).

**Literature placement (map, not full reproduce):**

| System | Family | Axes |
|--------|--------|------|
| Fast-dLLM | Conversion | A+D+C masked |
| NLD | Conversion | A+D joint AR |
| BlockGen | **Native** | B+C+E (scratch; never `ar2block`) |
| MDLM/Duo/GIDD | Cited | full-seq C |

---

## 3. Shared protocol

Fixed for all **main-track** cells unless an experiment moves axis A, D, or E.

| Setting | Value |
|---------|-------|
| Backbone | Qwen2.5-1.5B-Instruct |
| Pipeline | **AR→block** (`LINE=ar2block`, `load_pretrained=true`) |
| Data | Nemotron Post-Training SFT (`sft_qwen`) |
| Seq / block | 2048 / 32 |
| Optimizer | AdamW, lr 2e-5, warmup 500 |
| Global batch / steps | 256 / 6000 (~3.15B tokens) |
| Corruption (main) | **masked** (`algo=block_masked`) |
| Hooks (main) | **off** |
| Precision | bf16, 2×GPU DDP |

**Evaluation rule:** no conversion claim from val NLL alone. Every main cell:
val BPD + conditional lm-eval (≥ GSM8K, IFEval) + gen-PPL + optional tok/s.
**Primary cells (C0, C2*, C3)** also follow the longitudinal schedule in
[§7](#7-cross-cutting-measurements-for-downstream-analysis).

---

## 4. Metrics and baselines

### 4.1 Conversion success (main)

- Val NLL/BPD (monitoring only; not sole claim)
- **Conditional tasks:** GSM8K, IFEval, MMLU, HumanEval via `lm_eval.sh`
- **Gen-PPL** + qualitative samples via `examples/block_qwen/eval.sh`
  (`sample_mode=auto`, `decode_profile=baseline` — not bare-BOS for Instruct)
- **Throughput:** BlockSampler tok/s; enable `sampling.hierarchical_kv` (+ optional `sub_block_size`) for truncate-only progressive decode, and `use_block_cache` (lever `dual_cache`) for DualCache splice. Still not identical to Fast-dLLM fused kernels — label systems comparisons carefully.

### 4.2 Baselines (main)

| Baseline | Purpose |
|----------|---------|
| **`C0`** (narrative: `C0_ar2block_masked`) | Neutral conversion reference |
| **`C2_fdllm`** | Fast-dLLM positive control (axis D) |
| **`C3`** (narrative: `C3_fullseq`) | AR→full-seq **diffusion**, uniform+shift, `block_size=2048` |
| **`A_ar_sft`** | Matched causal AR SFT (Tab-1 AR reference — not full-seq) |
| **External** (Fast-dLLM, NLD) | Cited on design-space map only |

Scratch-init arms (`block_*`, `N0`) are the **native family reference**, not
failed conversion. Uniform corruption under `ar2block` is a **B1 extra**, not
co-equal with Fast-dLLM conversion claims.

### 4.3 Main track vs native vs extras

| Experiment | Conversion main | Native | Extras / transfer |
|------------|-----------------|--------|-------------------|
| C0 / C2 / C3 / longitudinal | **Core** | — | — |
| C4 / C5 / E_* | Supporting | — | decode extras |
| B1 masked vs uniform | — | — | **Extra** (RQ-B1) |
| B2 / N0 scratch | — | **Core native** | — |
| B3 geometry / ARPC | — | **Core native** (RQ-N2) | — |
| `blockgen_owt_uniform` | — | **BlockGen recreate** | — |
| `xfer_*` | — | — | **Transfer** (RQ-B2 / RQ-X) |
| B4 hybrid | — | — | Extra |

---

## 5. Experiment program

### Part I — Conversion (main track)

#### C0 — Pipeline validation

**Goal:** Harness works before conversion claims.

| Check | Status |
|-------|--------|
| G5 init metrics (`verify_ar_block_init.py`) | unit |
| Train loss decreases on `ar2block_masked` | job 141728 |
| Step-500/1000 smoke: conditional GSM8K + gen-PPL | job 141901 |

**Gate:** English-shaped conditional output; no mask leakage; val loss ↓.

---

#### C1 — Neutral conversion baseline

**Goal:** Establish **reference AR→block path** (RQ1 baseline).

| Cell | Axes | Launch |
|------|------|--------|
| `C0` | A, D neutral, C=masked fixed | `./scripts/submit_paper_cell.sh C0` |

**Hypotheses**

- C1a: Full weight load + block SFT reduces val BPD from init; gen quality lags
  loss (conversion is trainable but not automatically fluent).
- C1b: Neutral hooks-off recipe alone does **not** match Fast-dLLM fluency.

**Outputs:** Fig 2 (train/val curves); row in Tab 1.

---

#### C2 — Conversion recipe ablation (Fast-dLLM positive control)

**Goal:** Test whether conversion failure is **missing axis D** vs fundamental
(RQ1, RQ3). Masked only.

**Wiring:** use `scripts/submit_lever.sh` + `configs/levers/registry.yaml`
(do **not** hand-write lever `HYDRA_OVERRIDES` for design cells). Complementary
masks are **paired `m`/`~m`** (Fast-dLLM v2; two sequential forwards of batch B);
the old polarity-flip
impl is retired — discard any C2 readout trained under that bug.

| Cell | Registry preset | Axis D knobs |
|------|-----------------|--------------|
| `C2_shift` | `--preset C2_shift` | `shift_loss_targets` |
| `C0_shift` | `--preset C0_shift` | alias of `C2_shift` (Unifusion \(x_0\) naming) |
| `U0_shift` | `--preset U0_shift --arm uniform` | same shift on **uniform** (Unifusion-style) |
| `C2_comp` | `--preset C2_comp` | `complementary_masks` |
| `C2_fdllm` | `--preset C2_fdllm` | shift + complementary (**strict Tab 2**) |
| `C2_fdllm_full` | `--preset C2_fdllm_full` | + `mask_schedule=fast_dllm` + `loss_plain_ce` (recipe fidelity) |
| `C2_scale` | optional 2× steps/tokens (axis A budget) | `MAX_STEPS=12000` (etc.) on lever/sbatch launch — no separate cell |

```bash
# Strict axis-D ablation (matches Tab 2 cells) — thesis scale 6000/2048/256
./scripts/submit_lever.sh --preset C2_shift --arm masked
./scripts/submit_lever.sh --preset C2_comp --arm masked
./scripts/submit_lever.sh --preset C2_fdllm --arm masked

# Closer Fast-dLLM-ish control (schedule match); label separately from C2_fdllm
./scripts/submit_lever.sh --preset C2_fdllm_full --arm masked

# 500-step micro (explicit):
./scripts/submit_lever.sh --preset C2_fdllm --arm masked --micro
```

**Hypotheses**

- C2a: C2 improves conditional fluency vs C0 at matched steps → neutral
  comparison to Fast-dLLM was recipe-incomplete.
- C2b: C2 still soup → investigate budget (C2_scale) or remaining soft gaps
  (partial-mask already on; hier. KV still absent).

**Outputs:** **Fig 3** (C0 vs C2 learning curves); **Tab 2** (lever ablation).

---

#### C3 — AR→full-seq diffusion (Unifusion floor)

**Goal:** Fair **full-sequence diffusion** baseline from the same AR Instruct init /
Nemotron budget as C0/U0. **“Full-seq” in this paper always means diffusion**
(`block_size=model.length`), never causal AR.

| Cell | Description |
|------|-------------|
| `C3` | Full-seq diffusion: `block_size=2048`, **uniform + shift** |
| `A_ar_sft` | Matched **causal AR SFT** (separate paradigm — not called full-seq) |

**Requires:** `./scripts/submit_paper_cell.sh C3` → preset `C3_fullseq`,
`BLOCK=2048`, `arm=uniform`. Reuses `BlockTrainer` / `ar2block` with one block
= whole sequence. AR reference: `./scripts/submit_paper_cell.sh A_ar_sft`
(floor already done: `ar_sft_1857323`).

**Decode note:** `num_blocks=1` — hierarchical truncation ≡ full seq. The
multi-block `full_seq_dual` decode ban (bs=32) is unchanged.

**Outputs:** **Tab 1** paradigm columns: C0 (block diffusion) vs C3 (full-seq
diffusion) vs optional `A_ar_sft` (causal AR).

---

#### C4 — Decode evaluation sweep (conversion checkpoints)

**Goal:** Measure conversion output fairly; light touch on axis E (RQ4).

Fixed ckpt: best C0 or C2 by val BPD.

| Sweep | Knobs |
|-------|-------|
| NFE | `NUM_STEPS ∈ {8, 16, 32, 64}` |
| Conditional length | `max_new_tokens ∈ {128, 512}` |

**Not main track:** deep ARPC / BlockGen decode (see extras B3).

**Outputs:** Optional Fig 4 (quality vs NFE on **same conversion ckpt**).

---

#### C5 — Joint AR conversion control

**Goal:** External axis A+D point on map.

| Cell | Change |
|------|--------|
| `C5_joint_ar` | `L = L_AR + α L_diff`, α=0.3 (block-clean stream — **not** causal NLD) |
| `C5_causal_clean` | same + token-causal AR pass (NLD-like) |

```bash
./scripts/submit_paper_cell.sh C5_joint_ar
./scripts/submit_paper_cell.sh C5_causal_clean
```

**Metrics:** epoch `val/nll|bpd` stay **diffusion-only** (cross-cell comparable).
Use `val/joint_nll` / `train/joint_nll` for the optimized objective; also
`val/ar_nll`, `val/diff_nll`.

**Status:** ready. Design locks: [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).

---

### Part II — Extras (block structure & noising)

Run **after** C0/C2 establish a conversion baseline, or in parallel if compute
allows. **Appendix or §5.2**, not the abstract lead.

#### B1 — Corruption slice under conversion

**Goal:** RQ-B1 only — masked vs uniform **both starting from AR instruct**.

| Cell | C | Notes |
|------|---|-------|
| `B1_masked` (alias of C0) | masked | same as C0 |
| `B1_uniform` | uniform | `./scripts/submit_paper_cell.sh B1_uniform` |

Hold A, B, D, E fixed. One table row comparison—not a full 2×2 main-track study.

**Do not lead with:** uniform-vs-masked thesis. **Do report:** whether gen-PPL /
GSM8K invert val BPD ranking under the **same conversion pipeline**.

**Discussion-only (gated):** editable-token protocol locked in populated plan
(§ editable-token investigation). Mechanism = prior art (GIDD / remask lit);
contribution = **prove/deny** whether it causally explains local IFEval/code
under matched C0 vs U0 convert. Taxonomy → local-slice Δ; causal only with
inject/remask. Never “we discovered editable tokens.”

**Optional gated follow-up (`U0_from_C0`):** AR→**masked**→**uniform** continue-FT
(resume C0 into uniform arm) only if matched direct U0 underperforms C0.
Motivation: Unifusion finds two-stage ≈ direct on full-seq GenPPL; we test whether
a masked warm-start helps **block-Instruct** uniform. See populated plan §5.
Do **not** make two-stage the default conversion path.

---

#### B2 — Scratch controls / native floor

**Goal:** Native-family init baseline (RQ-N1). Also shows AR init dominates at
**matched conversion budget** (supports RQ1 when compared to C0 — different claim
from BlockGen OWT scale).

| Cell | Init | Line |
|------|------|------|
| `N0` | scratch uniform (hooks off) | `block` |
| `B2_masked` | scratch | `block` |
| `B2_uniform` | scratch | `block` |

Short appendix / native table: at Nemotron 6k budget, scratch ≫ worse than B1/C0.
Do **not** call N0/B2 a BlockGen recreate — that is `blockgen_owt_uniform`.

---

#### B3 — Native BlockGen levers (geometry + decode)

**Goal:** RQ-N2 — axis B/E on the **scratch / native** family (`line=block`).
Compare to **N0** / B2, **not** C0.

| Cell | Registry preset | Knobs | Line |
|------|-----------------|-------|------|
| `N0` | `--preset N0` | hooks off | `block` |
| `B3_mixture` | `--preset B3_mixture` | `algo.block_size_mixture=[16,32]` | `block` |
| `B3_weights_32` | `--preset B3_weights_32` | `bg_weights_1_32` | `block` |
| `B3_u_stratified` | `--preset B3_u_stratified` | weights 1+16 + u-strat | `block` |
| `B3_arpc` | `--preset B3_arpc` | mixture size 1 + `arpc_mode=blockgen` (**uniform**) | `block` |
| `B3_t_strat` | `--preset B3_t_strat` | our `stratified_gamma` (≠ u-strat) | `ar2block` (conversion-track geometry) |

```bash
./scripts/submit_paper_cell.sh N0
./scripts/submit_paper_cell.sh B3_mixture --arm masked   # or uniform
./scripts/submit_paper_cell.sh B3_arpc                    # uniform, line=block
./scripts/submit_blockgen_owt.sh                         # OWT 1+16 recreate
# Transfer (different claim — do not tag BlockGen):
./scripts/submit_lever.sh --preset xfer_mixture --arm masked --paper
./scripts/submit_lever.sh --preset xfer_arpc --arm uniform --paper
```

Single-factor only on B3 micros. Full recipe = `blockgen_owt_uniform`.

---

#### Xfer — Geometry on conversion (optional)

**Goal:** RQ-B2 / RQ-X. Same BlockGen knobs as B3, but **`LINE=ar2block`**.
Registry presets: `xfer_mixture`, `xfer_weights_32`, `xfer_u_stratified`,
`xfer_arpc`. Never appear in BlockGen recreate tables.

---

#### B4 — Hybrid noising

**Goal:** Third corruption pole (GIDD/XDLM-inspired). B4v1 locked (surrogate
ELBO + decode-as-masked); see [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).

| Cell | C |
|------|---|
| `B4_hybrid_p10` | mask + uniform @ default `p=0.1` |
| `B4_hybrid_p50` | real p sweep (`p=0.5`) |

```bash
./scripts/submit_paper_cell.sh B4_hybrid_p10
./scripts/submit_paper_cell.sh B4_hybrid_p50
```

**Status:** ready.

---

## 6. Pre-registered claims

### Conversion main track (after C0/C2/C3)

- A **conversion design-space map** separating conversion from native knobs.
- Which **conversion recipe** choices (neutral vs Fast-dLLM hooks vs budget) move
  conditional generation.
- Whether **matched AR SFT** matches or beats neutral AR→block on instruct tasks.
- Whether **val BPD mis-ranks** conversion success vs gen-PPL / GSM8K.
- Longitudinal **capability recovery vs conversion compute** ([§7](#7-cross-cutting-measurements-for-downstream-analysis)).

### Native track

- Scratch / BlockGen-faithful cells on **`LINE=block` only** (N0, B2, B3,
  `blockgen_owt_uniform`).
- Whether BlockGen mixture / ARPC move native quality (RQ-N2).

### Extras / transfer (report when data warrant)

- Under fixed AR→block skeleton: masked vs uniform (B1).
- Hybrid corruption and decode-speed probes (B4, E_*).
- Optional `xfer_*` geometry-on-conversion (RQ-B2 / RQ-X) — separate tables.

Plausible emergent findings:

| Outcome | Angle |
|---------|-------|
| **A** Metric mismatch is large | Likelihood mis-ranks conversion success (RQ4) |
| **B** Fast-dLLM recipe dominates | Neutral conversion was recipe-incomplete (C2) |
| **C** Initialization dominates | Scratch ≫ AR at matched budget (B2 vs C0) |
| **D** Corruption surprise | B1 rankings invert |
| **E** Native ≠ transfer | B3 gain does not imply `xfer_*` gain (or vice versa) |

### Never without external reproduction

- Fast-dLLM / NLD leaderboard parity.
- “Uniform is the better paradigm” as title claim.
- Calling `ar2block` + BlockGen knobs a BlockGen recreate.

---

## 7. Cross-cutting measurements for downstream analysis

**Not a new axis.** Same protocol as §3–§4; added **execution discipline** so
main-track and extras share one eval bundle without rerunning training.

### 7.1 Longitudinal checkpoint schedule

For all **primary conversion cells** (C0, C2*, C3), evaluate at:

**init (step 0) → 500 → 1000 → 2000 → 4000 → 6000**

**Launch (automated bundles):**

```bash
# After training (or on an existing run dir):
RUN_ROOT=outputs/block_qwen/ar2block_masked_141728 \
  ./scripts/submit_paper_cell.sh longitudinal

# Or directly:
./scripts/submit_longitudinal_eval.sh outputs/block_qwen/ar2block_masked_141728
STEPS="500 1000" ./scripts/submit_longitudinal_eval.sh outputs/block_qwen/ar_sft_143599
```

Uses `tools/run_milestone_eval.py` → writes `eval/step_<N>/` below.
Step **0** materializes a pretrained init ckpt (`checkpoints/init-step0.ckpt`) when missing.

At each point collect (minimum):

| Metric | Purpose |
|--------|---------|
| Val BPD | Likelihood track |
| Gen-PPL | Unconditional fluency |
| GSM8K | Instruction / reasoning retention |
| IFEval | Format-following retention |
| *(optional)* one extra capability | e.g. MMLU subset or HumanEval |

This enables post-hoc plots of **capability recovery vs conversion compute**
without a separate study.

### 7.2 Standardized evaluation bundle

Each checkpoint writes the same layout under the run root:

```text
<RUN_ROOT>/eval/step_<N>/
    milestone_manifest.json   # index of artifacts for this step
    train_metrics.json        # loss, step, tokens seen
    val_bpd.json
    gen_ppl.json              # from gen_ppl_metrics.json
    lm_eval/
        SUMMARY.json          # task scores (+ tok/s when full tier)
    generation_samples/       # samples.pt (+ .txt when generated)
    throughput.json           # BlockSampler tok/s (step 6000 / full tier)
```

Legacy flat dirs (`<RUN_ROOT>/eval/`, `<RUN_ROOT>/lm_eval/`) still come from
one-shot post-train `eval_checkpoint.sbatch` / `lm_eval.sbatch`.

Thesis aggregation: *“Across the design space, here is what happens.”*  
Extras reuse the same bundles for appendix tables and optional deep dives.

### 7.3 When to run full vs light eval

| Checkpoint | Suite |
|------------|-------|
| 500, 1000 | BPD + gen-PPL + GSM8K (smoke gate) |
| 2000, 4000 | + IFEval |
| 6000 (+ best val ckpt) | Full suite + MMLU + HumanEval + throughput |

Init (step 0): eval pretrained AR instruct **before** conversion for retention
baseline (C3 uses same schedule on causal SFT).

---

## 8. Figure and table plan

| ID | Content | Tier |
|----|---------|------|
| **Fig 1** | Design space: conversion vs native vs transfer | Main |
| **Fig 2** | C0 train/val curves | Main |
| **Fig 3** | Capability recovery vs step (C0 / C2 / C3 longitudinal) | Main |
| **Fig 4** | NFE sweep on conversion ckpt: **(PPL, H) vs steps** | Supporting (Unifusion hygiene) |
| **Fig 4b** | Dual GenPPL modes (first_chunk vs full) on C0/U0/C2 | Appendix |
| **Fig 4c** | Qualitative collapse panel (2–3 samples) | Appendix |
| **Fig 5** | Metric agreement / disagreement (BPD vs gen-PPL vs GSM8K) | Main if Outcome A |
| **Tab 1** | C0, C2*, C3 — final-step metrics | Main conversion |
| **Tab 2** | Fast-dLLM lever ablation (C2) | Main conversion |
| **Tab N1** | N0 / B2 / B3 / BlockGen OWT | Native |
| **Tab X1** | `xfer_*` vs C0 (optional) | Transfer |
| **Tab B1** | B1 masked vs uniform (same conversion skeleton) | Extra |
| **Tab A1** | Literature vs axes / families (full map) | Appendix |

### GenPPL reporting rule (Unifusion hygiene)

**Never** put GenPPL alone in a table or figure. Every free-gen row is
`(PPL, H_mean[, H_med])` from `gen_ppl_metrics.json` / `gen_ppl_pair.json`.

Protocol (cell **C4** / `scripts/submit_gen_ppl_hygiene.sh`):

| Piece | What |
|-------|------|
| Pair always | `tools/gen_ppl_hygiene.py pair` after every gen-PPL |
| NFE sweep | steps ∈ {8,16,32,64}, same ckpt/prefix; plot PPL↓ & H vs NFE |
| Dual modes | `first_chunk_only=true` (claim) **and** `false` (full string) on C0/U0/C2 — separate columns |
| Collapse panel | 2–3 samples → `collapse_panel/panel.md` appendix |
| Optional | `RUN_MULTISEED=1` on U0/U0_shift; `--with-winogrande` base-LM slice |

```bash
CKPT=.../ar2block_masked_1762534/checkpoints/last.ckpt \
  ./scripts/submit_paper_cell.sh C4
# or:
CKPT=... MODE=all RUN_MULTISEED=0 ./scripts/submit_gen_ppl_hygiene.sh
CKPT=... ./scripts/submit_gen_ppl_hygiene.sh --with-winogrande
```

---

## 9. Implementation: design cells

One Slurm job = one cell; resume via `RUN_ROOT`.

**Single entrypoint (preferred):**

```bash
./scripts/submit_paper_cell.sh --list

# Main track (conversion)
./scripts/submit_paper_cell.sh C0
./scripts/submit_paper_cell.sh C2_shift
./scripts/submit_paper_cell.sh C2_comp
./scripts/submit_paper_cell.sh C2_fdllm
./scripts/submit_paper_cell.sh C3
CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh C4
CKPT=... ./scripts/submit_paper_cell.sh lm_eval
RUN_ROOT=outputs/.../ar2block_masked_141728 ./scripts/submit_paper_cell.sh longitudinal

# Native track (scratch / BlockGen)
./scripts/submit_paper_cell.sh N0
./scripts/submit_paper_cell.sh B2_masked
./scripts/submit_paper_cell.sh B2_uniform
./scripts/submit_paper_cell.sh B3_mixture --arm masked
./scripts/submit_paper_cell.sh B3_u_stratified
./scripts/submit_paper_cell.sh B3_weights_32
./scripts/submit_paper_cell.sh B3_arpc            # uniform, line=block
./scripts/submit_blockgen_owt.sh                  # OWT 1+16

# Transfer (optional — not BlockGen)
./scripts/submit_lever.sh --preset xfer_mixture --arm masked --paper
./scripts/submit_lever.sh --preset xfer_arpc --arm uniform --paper

# Conversion extras
./scripts/submit_paper_cell.sh B1_uniform
./scripts/submit_paper_cell.sh B3_t_strat         # our t-strat on ar2block
./scripts/submit_paper_cell.sh C5_joint_ar
./scripts/submit_paper_cell.sh C5_causal_clean
./scripts/submit_paper_cell.sh B4_hybrid_p10
./scripts/submit_paper_cell.sh B4_hybrid_p50

# Decode eval on existing ckpt
CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh E_hierarchical
CKPT=... ./scripts/submit_paper_cell.sh E_dual_cache
CKPT=... ./scripts/submit_paper_cell.sh E_sub_block
```

`E_*` decode presets require `CKPT=` (eval-time sampling overrides; no retrain).

Catalog: `configs/paper/cells.yaml`. Lever wiring: `configs/levers/registry.yaml`
+ `scripts/submit_lever.sh`. Train entry: `scripts/_block_qwen_launch.bash`.
Design locks: [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).

| Cell | How it launches |
|------|-----------------|
| C0 / B1_masked | `ar2block_masked.sbatch` |
| B1_uniform | `ar2block_uniform.sbatch` |
| B2_* | `block_{masked,uniform}.sbatch` |
| N0 / B3_* (except t_strat) | `submit_lever.sh` → **`block_*.sbatch`** |
| B3_t_strat / C2_* / C5_* / B4_* | `submit_lever.sh` → `ar2block_*.sbatch` |
| `xfer_*` | `submit_lever.sh` → `ar2block_*.sbatch` (transfer tag) |
| `blockgen_owt_uniform` | `submit_blockgen_owt.sh` → `block_uniform.sbatch` |
| C3 | `ar_sft.sbatch` |
| C4 / E_* / longitudinal / eval | eval entrypoints as before |

**Axis / family map:** A = `LINE` / C3; D = `C2_*` / `C5_*` (conversion);
B native = `B3_*` / BlockGen; B transfer = `xfer_*`; C = `--arm` / B1 / B4;
E = C4 + `E_*` + lm_eval.

---

## 10. Execution priority

All **train** cells are wired. **Eval** cells need `CKPT=` or `RUN_ROOT=` as noted.
Prefer interpreting C0/C2 before writing B3/B4/C5 results (scheduling, not stubs) — see
[`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).

| Priority | Cell / experiment | Notes |
|----------|-------------------|-------|
| P0 | C0 validation + finish conversion spine | Longitudinal at 500/1k/2k/4k/6k |
| P1 | C2 Fast-dLLM combined control | Discard polarity-flip micros for Tab 2 |
| P2 | C3 full-seq diffusion | `BLOCK=2048` uniform+shift; AR SFT = `A_ar_sft` |
| P3 | C4 decode sweep on best conversion ckpt | After P0–P1 |
| P4 | B1 `B1_uniform` | Conversion-family corruption slice |
| P5 | N0 / B2 scratch pair | Native floor at matched budget |
| P6 | B3 native + `blockgen_owt_uniform` | Native / BlockGen — **line=block** |
| P7 | B4 / C5 / E_* / optional `xfer_*` | Extras / transfer when bandwidth allows |
| P8 | Write-up | Separate conversion vs native vs transfer tables |

---

## 11. Limitations

- Instruct SFT ~3B tokens, not NLD-scale CPT/SFT (conversion family).
- Conversion main track fixes **masked** corruption (Fast-dLLM class).
- Native BlockGen recreate uses different data/scale (OWT 1M) than conversion SFT.
- DualCache / systems tok/s: locked non-parity — [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).
- Single backbone (Qwen2.5-1.5B-Instruct).
- Transfer (`xfer_*`) is optional and must not be sold as BlockGen.

---

## 12. Abstract blurb (thesis draft)

We map two related but distinct design spaces for discrete block diffusion:
**(1) AR→block conversion** of a pretrained LM, and **(2) native scratch block
diffusion** (including BlockGen-faithful recipes). Published systems often
confound these. Using one implementation, we run conversion cells (neutral
AR→block, Fast-dLLM recipe controls, matched AR SFT) on `LINE=ar2block`, and
native / BlockGen cells on `LINE=block`, with an optional **transfer** arm
(`xfer_*`) that applies BlockGen geometry to conversion without calling it
BlockGen. All primary conversion cells record **longitudinal** metrics at fixed
milestones. Results are always reported as **`line × arm × data × budget`**.
