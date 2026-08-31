# Thesis experiments — AR→block diffusion conversion design space

Draft experimental plan for a **thesis chapter** on the design space of
autoregressive-to-block diffusion conversion, plus **extras** (appendix /
exploratory cells) on block geometry, corruption, and decode — same codebase,
one execution program.

**Two tiers (same experiments, different narrative weight)**

| Tier | Question | Scope |
|------|----------|-------|
| **Thesis (main track)** | *What is the design space of AR→block diffusion conversion?* | Axes A, D, E; C0/C2/C3 longitudinal eval; literature map |
| **Extras (bonus)** | *How do block-native knobs behave under the same skeleton?* | Axes B, C (B1–B4), decode micros (E_*), scratch controls |

The thesis gives **breadth and the conversion story**; extras add **depth on
block structure and noising** without restructuring the main narrative.

```text
                    THESIS (main track)
        ┌──────────────────────────┐
        │ AR → Block Design Space  │
        │  A Init & budget         │
        │  D Conversion recipe     │
        │  E Decoding (eval)       │
        │  C0 / C2 / C3 spine      │
        └────────────┬─────────────┘
                     │ same cells, lower priority
                     ▼
        ┌──────────────────────────┐
        │ EXTRAS (appendix)        │
        │  B Geometry (B3)         │
        │  C Corruption (B1/B4)    │
        │  Scratch (B2)            │
        │  Decode speed (E_*)      │
        └──────────────────────────┘
```

**Within-thesis scope split**

| Tier | Focus | Role |
|------|--------|------|
| **Main track** | Conversion path (init, budget, dual-stream recipe, decode eval) | Chapters 4–5 spine |
| **Extras** | Block geometry & noising | Appendix / design-space synthesis |

Block structure and noising are valuable for the design-space map but are **not**
the main-track headline unless results surprise (e.g. B1 corruption inverts
rankings).

Implementation: one codebase (`BlockTrainer`), Hydra configs, one Slurm job per
**design cell** ([§9](#9-implementation-design-cells)).

Related notes: [BASELINE_MASKED_UNIFORM_AR.md](BASELINE_MASKED_UNIFORM_AR.md),
[BLOCK_QWEN_TRAINING.md](BLOCK_QWEN_TRAINING.md), [LEVERS.md](LEVERS.md).

---

## 1. Research questions

### Main (conversion)

**RQ1 (conversion recipe).** Which **conversion-axis** choices—pretrained init,
SFT token budget, dual-stream training hooks (Fast-dLLM), optional joint AR
objective (NLD)—determine whether AR→block produces usable generation at
instruct SFT scale?

**RQ2 (vs matched AR).** After the same Nemotron post-training budget, does
block conversion beat **matched causal AR SFT** on task accuracy, and on
parallel-decode throughput?

**RQ3 (confounding in literature).** Do published systems (Fast-dLLM, NLD,
BlockGen) **confound** conversion choices with block-native choices (geometry,
corruption, decode tricks)? Can we separate them experimentally?

**RQ4 (metrics).** For conversion success, does validation NLL/BPD track
conditional generation and gen-PPL, or mislead (cf. *Scaling Beyond Masked
Diffusion*)?

### Extras (block structure & noising)

**RQ-B1 (corruption under conversion).** Holding the **same AR→block skeleton**,
does masked vs uniform corruption change gen quality or only likelihood ranking?

**RQ-B2 (geometry / decode extras).** Do BlockGen-style mixture or ARPC matter
*incrementally* once a conversion baseline works—or only when corruption is
uniform?

---

## 2. Design space — conversion-first notation

Five axes total; **three primary (conversion), two extras (block-native).**

### Primary axes (thesis spine)

| Axis | Name | Question it answers | Neutral reference | Knobs |
|------|------|---------------------|-------------------|-------|
| **A** | **Init & budget** | What are you converting from, for how long? | Qwen2.5-Instruct, ~3B tokens | `LINE=ar2block`, `model.load_pretrained`, `trainer.max_steps`, data |
| **D** | **Conversion recipe** | How is the AR backbone adapted to block training? | hooks off | `submit_lever.sh --preset C2_*` / `C5_*` (`shift`, `complementary`, `joint_ar_alpha`, …) |
| **E** | **Decode evaluation** | How do we *measure* conversion output? | 32-step BlockSampler | `NUM_STEPS`, `max_new_tokens`, lm-eval harness |

### Extras axes (fixed in main experiments; varied in appendix)

| Axis | Name | Question it answers | Fixed in main track | Knobs when varied |
|------|------|---------------------|---------------------|-------------------|
| **B** | Block geometry | Block size, mixture, t sampling | `block_size=32`, no mixture | `block_size_mixture`, `stratified_gamma` |
| **C** | Corruption / noising | Masked vs uniform vs hybrid | **masked** (Fast-dLLM-class default) | `algo=block_masked \| block_uniform` |

**Main-track reference cell:** `C0` (`C0_ar2block_masked` in narrative) — AR instruct init, masked
corruption, block 32, hooks off, Nemotron SFT 6000×256.

**Literature placement (map, not full reproduce):** Fast-dLLM (A+D+C masked),
NLD (A+D joint AR), BlockGen (B+C+E extras), MDLM/Duo/GIDD (full-seq C, cited in
extras discussion).

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
- **Gen-PPL** + qualitative free-gen (`eval.sh`)
- **Throughput:** BlockSampler tok/s; enable `sampling.hierarchical_kv` (+ optional `sub_block_size`) for truncate-only progressive decode, and `use_block_cache` (lever `dual_cache`) for DualCache splice. Still not identical to Fast-dLLM fused kernels — label systems comparisons carefully.

### 4.2 Baselines (main)

| Baseline | Purpose |
|----------|---------|
| **`C0`** (narrative: `C0_ar2block_masked`) | Neutral conversion reference |
| **`C2_fdllm`** | Fast-dLLM positive control (axis D) |
| **`C3`** (narrative: `C3_ar_sft`) | Matched AR SFT, same data/steps (axis A) |
| **External** (Fast-dLLM, NLD) | Cited on design-space map only |

Scratch-init arms (`block_*`) and uniform corruption are **extra controls**, not
co-equal thesis spines.

### 4.3 Main track vs extras

| Experiment | Main track | Extras |
|------------|------------|--------|
| C0 neutral conversion | **Core** | — |
| C2 Fast-dLLM recipe | **Core** | — |
| C3 matched AR SFT | **Core** | — |
| C4 NFE/decode sweep | Supporting | — |
| C5 NLD | Supporting | — |
| Longitudinal metric suite | **Core** | — |
| Literature axis map | **Core** (intro / related) | — |
| B1 masked vs uniform | — | **Extra** (RQ-B1) |
| B2 scratch vs AR init | — | **Extra** (init interaction) |
| B3 geometry / ARPC | — | **Extra** (RQ-B2) |
| B4 hybrid | — | **Extra** (exploration) |
| E_hierarchical / E_dual_cache | — | **Extra** (decode systems) |

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

#### C3 — Matched AR SFT baseline

**Goal:** Is block conversion worth it vs same-budget AR fine-tuning? (RQ2)

| Cell | Description |
|------|-------------|
| `C3` | Causal LM SFT, same Nemotron / steps / LR / batch |

**Requires:** `algo=ar_sft` — wired via `configs/experiment/ar_sft_qwen.yaml` +
`scripts/slurm/ar_sft.sbatch`.

**Outputs:** **Tab 1** three-way: C0 vs C3 vs (optional best C2).

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

---

#### B2 — Scratch controls (init interaction)

**Goal:** Show AR init dominates at fixed budget (supports RQ1, not RQ-B1).

| Cell | Init |
|------|------|
| `B2_masked` | scratch |
| `B2_uniform` | scratch |

Short appendix table: scratch ≫ worse than B1/C0 at same steps.

---

#### B3 — BlockGen levers (geometry + decode)

**Goal:** RQ-B2 — incremental gains from axis B/E **after** conversion works.

| Cell | Registry preset | Knobs |
|------|-----------------|-------|
| `B3_mixture` | `--preset B3_mixture` | `algo.block_size_mixture=[16,32]` |
| `B3_arpc` | `--preset B3_arpc` | mixture incl. size 1 + `arpc_mode=blockgen` (**uniform only**) |

```bash
./scripts/submit_lever.sh --preset B3_mixture --arm masked   # or uniform
./scripts/submit_lever.sh --preset B3_arpc --arm uniform
# Decode speed (eval-time on conversion ckpt):
# CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh E_hierarchical
# CKPT=... ./scripts/submit_paper_cell.sh E_dual_cache
```

Single-factor only. **Appendix** unless one lever clearly moves C0 fluency gate.

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

## 6. Pre-registered claims (thesis)

### Main track (after C0/C2/C3 program)

- A **conversion design-space map** (axes A–E) separating conversion from
  block-native knobs.
- Which **conversion recipe** choices (neutral vs Fast-dLLM hooks vs budget) move
  conditional generation.
- Whether **matched AR SFT** matches or beats neutral AR→block on instruct tasks.
- Whether **val BPD mis-ranks** conversion success vs gen-PPL / GSM8K.
- Longitudinal **capability recovery vs conversion compute** from the shared
  checkpoint schedule ([§7](#7-cross-cutting-measurements-for-downstream-analysis)).

### Extras (report when data warrant; not required for main claims)

- Under fixed AR→block skeleton: masked vs uniform and scratch vs AR init
  (B1/B2).
- Incremental BlockGen-style geometry / ARPC (B3).
- Hybrid corruption and decode-speed probes (B4, E_*).

Plausible emergent findings from extras (data picks which to emphasize in appendix):

| Outcome | Extras angle |
|---------|--------------|
| **A** Metric mismatch is large | Likelihood mis-ranks conversion success (also main-track RQ4) |
| **B** Fast-dLLM recipe dominates | Neutral skeleton was recipe-incomplete (main C2) |
| **C** Initialization dominates | Scratch ≫ AR at fixed budget (B2) |
| **D** Corruption surprise | B1 becomes appendix centerpiece (rankings invert) |

### Never without external reproduction

- Fast-dLLM / NLD leaderboard parity.
- “Uniform is the better paradigm” as title claim.

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
| **Fig 1** | Conversion design space (A,D,E primary; B,C extras inset) | Main |
| **Fig 2** | C0 train/val curves | Main |
| **Fig 3** | Capability recovery vs step (C0 / C2 / C3 longitudinal) | Main |
| **Fig 4** | NFE sweep on conversion ckpt (optional) | Supporting |
| **Fig 5** | Metric agreement / disagreement (BPD vs gen-PPL vs GSM8K) | Main if Outcome A |
| **Tab 1** | C0, C2*, C3 — final-step metrics | Main |
| **Tab 2** | Fast-dLLM lever ablation (C2) | Main |
| **Tab B1** | B1 masked vs uniform (same conversion skeleton) | Extra |
| **Tab B2** | Scratch vs AR init (B2 vs C0) | Extra |
| **Tab A1** | Literature vs axes (full map) | Appendix |

---

## 9. Implementation: design cells

One Slurm job = one cell; resume via `RUN_ROOT`.

**Single entrypoint (preferred):**

```bash
./scripts/submit_paper_cell.sh --list

# Main track
./scripts/submit_paper_cell.sh C0                 # or reuse ar2block_masked_141728
./scripts/submit_paper_cell.sh C2_shift
./scripts/submit_paper_cell.sh C2_comp
./scripts/submit_paper_cell.sh C2_fdllm           # job 145506 already queued
./scripts/submit_paper_cell.sh C3                 # or resume ar_sft run
CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh C4
CKPT=... ./scripts/submit_paper_cell.sh lm_eval
RUN_ROOT=outputs/.../ar2block_masked_141728 ./scripts/submit_paper_cell.sh longitudinal

# Extras (decode eval on existing ckpt — no retrain)
CKPT=outputs/.../last.ckpt ./scripts/submit_paper_cell.sh E_hierarchical
CKPT=... ./scripts/submit_paper_cell.sh E_dual_cache
CKPT=... ./scripts/submit_paper_cell.sh E_sub_block

# Extras (train cells)
./scripts/submit_paper_cell.sh B1_uniform
./scripts/submit_paper_cell.sh B2_masked
./scripts/submit_paper_cell.sh B3_mixture --arm masked
./scripts/submit_paper_cell.sh B3_t_strat
./scripts/submit_paper_cell.sh B3_u_stratified
./scripts/submit_paper_cell.sh B3_weights_32
./scripts/submit_paper_cell.sh B3_arpc            # uniform
./scripts/submit_paper_cell.sh C5_joint_ar
./scripts/submit_paper_cell.sh C5_causal_clean
./scripts/submit_paper_cell.sh B4_hybrid_p10
./scripts/submit_paper_cell.sh B4_hybrid_p50
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
| C2_* / B3_* / C5_* / B4_* | `submit_lever.sh --preset …` (thesis scale) |
| C3 | `ar_sft.sbatch` (post-train eval + optional `longitudinal`) |
| C4 | `submit_nfe_sweep.sh` (via `submit_paper_cell.sh C4`) |
| E_* | `decode_eval` on `CKPT=` (sampling overrides only) |
| Longitudinal | `submit_longitudinal_eval.sh` or cell `longitudinal` |
| eval / lm_eval | `eval_checkpoint.sbatch` / `lm_eval.sbatch` |

**Axis map:** A = `LINE` / C3; D = `C2_*` / `C5_*`; E = C4 + `E_*` + lm_eval; B = `B3_*`; C = `--arm` / B1 / B4.

---

## 10. Execution priority

All **train** cells are wired. **Eval** cells need `CKPT=` or `RUN_ROOT=` as noted.
Prefer interpreting C0/C2 before writing B3/B4/C5 results (scheduling, not stubs) — see
[`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).

| Priority | Cell / experiment | Notes |
|----------|-------------------|-------|
| P0 | C0 validation + finish `ar2block_masked_141728` | Longitudinal eval at 500/1k/2k/4k/6k |
| P1 | C2 Fast-dLLM combined control | **Job 145506** queued (`C2_fdllm`). Discard polarity-flip `fastdllm_141924` for Tab 2. |
| P2 | C3 matched AR SFT | Same schedule; init = pretrained AR |
| P3 | C4 decode sweep on best conversion ckpt | After P0–P1 |
| P4 | B1 `B1_uniform` | Extra — corruption slice |
| P5 | B2 scratch pair | Extra — init interaction |
| P6 | B3 / B4 / C5 / E_* | **Ready** — extras when reporting bandwidth allows |
| P7 | Extras write-up | After main-track C0/C2/C3 substantially complete |

---

## 11. Limitations

- Instruct SFT ~3B tokens, not NLD-scale CPT/SFT.
- Main track fixes **masked** corruption (Fast-dLLM conversion class).
- DualCache / systems tok/s: locked non-parity — [`DESIGN_LOCKS.md`](DESIGN_LOCKS.md).
- Single backbone (Qwen2.5-1.5B-Instruct).
- Extras B/C results are not required for main claims.

---

## 12. Abstract blurb (thesis draft)

We map the **design space of autoregressive-to-block diffusion conversion**:
axes of initialization, training recipe, block geometry, corruption, and
decoding, and how published systems confound them. Using one implementation and
Nemotron instruct SFT, we run design cells for neutral **AR→block** conversion,
**Fast-dLLM recipe** controls, **matched AR SFT**, and extras on corruption,
scratch initialization, geometry, and decode. All primary cells record
**longitudinal capability and likelihood metrics** at fixed training milestones,
enabling analysis of capability recovery vs conversion compute and of metric
agreement without rerunning training. The thesis retains the full map and
engineering record; appendix material covers block-native knobs that do not
define the main conversion narrative.
