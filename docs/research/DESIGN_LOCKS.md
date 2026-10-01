# Design locks (closed — not open work)

Accepted decisions for the thesis path. These are **not** TODOs.
Do not reopen without a new cell ID / version bump.

| ID | Decision | Consequence |
|----|----------|-------------|
| **BASELINE-CONTRACT** | **Fixed:** AR→block init. **Block skeleton:** BlockGen `[prefix\|block]` packing (`block_gen_logits`) + remask/commit where applicable. Masked floor = conf thr; **UCC is only the UC cell** (`uniform_commit`), not a silent remap of every U0 profile. Mid-α ancestral alone collapses Instruct (~2% GSM) — forensic `hierarchical_ancestral`. DualCache K/V / ARPC / hubmatch ss = levers. Full-seq dual refused. | Do not coerce hierarchical→UCC. Cite UCC only from UC. |
| **BAKEOFF-B1B4** | Phase-0 decode bake-off on fixed C0 `1762534` / U0 `1955203`: **B1** `hierarchical` s32 · **B2** `hierarchical_arpc` s32 · **B3** `hierarchical_ss` · **B4** `hierarchical_arpc` s8. Homemade probes **DC** DualCache (C0 only) / **UC** UCC (U0 only, once). Submit: `scripts/submit_baseline_bakeoff.sh`. Map: `FAIR_AR2BLOCK_MAP.md`. | Do **not** cut mid-α ancestral NFE alone. Floor stays B1 until bake-off overturns it. |
| **HUB-C2-EXTERNAL** | Fast-dLLM **GitHub** codebase (`third_party/Fast-dLLM/v2`) × our C2 Nemotron mix × **same-run** Hub `eval.py` = external pipeline-vs-mix check. `submit_hub_fastdllm_c2.sh`. Data/init under `/e/scratch/.../hub_fastdllm_c2/`. | Not UNI-D² BlockSampler. Train must finish before eval (`afterok`). “Hub” in scripts = official Fast-dLLM code, not Hugging Face Hub. |
| **FAMILIES** | Three experiment families: **conversion** (`ar2block`), **native** (`block`), **transfer** (`xfer_*` on `ar2block`) | Registry refuses Fast-dLLM levers on `block` and native/BlockGen levers on `ar2block` unless preset is `xfer_*`. Never tag transfer as BlockGen. |
| **PARADIGMS×COMPONENTS** | Write-up/talks use **paradigms** (native · AR→block · **AR→full-seq diffusion=C3** · matched causal AR=`A_ar_sft` · joint=C5/U2). **“Full-seq” = diffusion only** (`block_size=length`), never causal AR. Corruption = masked\|uniform\|hybrid; geometry = B/xfer/full-seq. | Do not call native/C3 “components”; do not call joint a corruption/geometry choice. Never label AR SFT as full-seq. |
| **C3-FULLSEQ** | Paper **C3** = AR Instruct init → **full-sequence diffusion** (`block_size=model.length`, default uniform+shift). Causal AR SFT is **`A_ar_sft`** only (`ar_sft_1857323`). | Do not cite AR SFT as C3 or as “full-seq.” DESIGN_LOCK full-seq **dual decode** ban still applies to **multi-block (bs=32)** ckpts — not to full-seq **train**. |
| **THESIS-PROTOCOL** | Three layers: **lock floor → search within paradigm → freeze + one lever family**. Floor selection is **preregistered** ([`THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md`](THESIS_EXPERIMENT_PROTOCOL_2026-09-27.md)) before attack GSM harvest. Attacks ≠ automatic floors. | Do not pick floors by post-hoc taste; do not start new xfer mashups until AR→block gate fires. |
| **B4v1** | Hybrid trains with mask+Unif surrogate ELBO; **decode-as-masked** (prior = all MASK) | No hybrid reverse process in v1. Cell `B4_hybrid_p10` / `B4_hybrid_p50` (conversion family). |
| **C5 streams** | `C5_joint_ar` = block-clean dual-stream AR; `C5_causal_clean` = token-causal AR | Only the latter supports causal-NLD wording. Metrics: `val/nll` = diffusion; `val/joint_nll` = objective. |
| **DualCache** | `use_block_cache` is K/V-only replace_position **approximation** | Quality evals use `E_hierarchical` / truncate-only. Tok/s probes may use `E_dual_cache`. |
| **Systems tok/s** | Our BlockSampler ≠ Fast-dLLM fused / SGLang numbers | Always label; never claim parity. |
| **BlockGen home** | BlockGen recreate / B3 native cells are **`LINE=block` only** | Use `submit_blockgen_owt.sh` / `B3_*`. Jobs 1660577/1660931 discarded as BlockGen claims. |
| **P6 scheduling** | Native B3 + BlockGen OWT are **wired**; prefer interpreting C0/C2 before writing conversion results | Launch order is operator choice; keep family tables separate. |
| **STILL-TRUE** | (1) open `conversion_free` free-gen is still **soup**; (2) **xfer ≠ U0 recipe** | See deep-dive below; GenPPL+H = collapse hygiene only; claim **K**/U0 tables exclude xfer |
| **U0-1849335-CONTAMINATED** | Canonical U0 `1849335` trained under **`Unif(V)` + `hierarchical_kv=false` + dual-stream**; current stack is **`V_eff` + hierarchical_ss / `uniform_dual` / `single_stream_train`** | **Do not cite as locked floor** for new decode/train pins. Use `1955203`+ / `U0_ss_pack` / `U0_ss_shift` for claim cells. Keep `1849335` only as historical / decode-switch probe. |
| **U0-GREEDY-SOUP** | Uniform + `greedy=1` / `argmax(q_xs)` is **illegal decode** — mid-α locks Unif prior into the lattice (IFE baseline Sep 17 = multilingual soup). Fair lever-off = ancestral (`greedy=0`); soup ≠ ancestral floor | Never cite IFE 10.7 / soup samples. See `SMOKING_GUN_UNIFORM_GREEDY_DECODE_2026-09-23.md`. |
| **U0-DECODE-PACK** | Uniform decode must be **truncated** (`hierarchical_kv`); full-seq dual and attn-block ceil into future Unif are hostile. `baseline` → `hierarchical` on every arm. EOS banned from Unif *noise redraws*. Unused HF embed slots carved from simplex | `coerce` only aliases `baseline`→`hierarchical`. **Never** remap hierarchical/hubmatch/dual_cache → UCC. |
| **SKELETON-OPENLOOP** | Truncated mid-α ancestral @ **T=1** is **unusable** on Instruct (~2–5% GSM; hard twins **2.4% M / 5.4% U**) — not a permanent open-loop verdict until T=0.1 2×2 lands. Proves **masked** needs a corrector (~56% conf remask on same M weights); Unif “collapse is just knobs” stays untested. Matched gap baseline = **ARPC**; UCC homemade. | Ablation: open-loop 2×2; do not rewrite collapse story before t01 lands. |
| **ARPC-GAP-BASELINE** | Parallel-enough M∥U baseline = `hierarchical_arpc` on hard twins `2020048`∥`2020049`. **Two rows:** matched twin (55.6/33.4, Δ22.2) + tuned-Unif reference (`2020048` vs `2092151`, 55.6/38.1, Δ≈17.5, **not matched recipe**). Scorer = **same diffusion backbone** `causal_logits` + `ar_metric`/`nll` (not frozen external AR; C3 no-size-1 skip N/A here). Post-reset MASK vs Unif is the arm. Require config-diff (noise kernel only) + reset frac / rounds / NFE per arm. | Never cite 55.6 vs 38.1 as same-recipe; never claim “same conditions” after reset. |
| **U0-UNIFORM-COMMIT** | Valid uniform conf-commit = **UCC**: Hub DualCache-commit analog (argmax + thr + force-max; Unif undecided; **revise off**). Not DualCache K/V. Bake **UC** = `uniform_commit`; DualCache twin = `uniform_dual`. | Claim UCC only from `bake-U0-UC` / explicit profile. |
| **U0-STICKY-CONF** | UCC must use **confidence-until** (dt=None loop), same *schedule* as masked DualCache — not fixed-N mid-α ancestral; no Duo scribble on undecided | `_denoise_block` UCC branch. |
| **ORACLE-N2C** | Dual-stream train always feeds GT previous-block `x0` via N2C; ancestral / sticky decode does not | val/ppl ≠ open-loop competence. Attack with `U0_ss_pack` / `xfer_bg_mix_32_blockgen_ss` (+ `_ss_shift`); DualCache/`uniform_dual` approximate commit on decode only. |
| **UNIF-V-ABLATION** | Default simplex = **`conversion`** = `Unif(V\E)` (+ EOS redraw carve). Literal **`Unif(V)`** = `algo.uniform_simplex_mode=blockgen` via lever `unif_v_blockgen` / preset `xfer_bg_mix_32_blockgen_unifv` | Never silently flip U0 default to full V. Cite Unif(V) only from unifv cells. |
| **PARAM-LABEL** | `algo.parameterization` `mean`\|`subs` is a **label** on the block path (sampler always x0→posterior) | Never A/B cite mean vs subs as a physics change for U0. |
| **TINYGSM-RECIPE** | U0 conversion is fixed `bs=32`, T=1, no size-1 CE / ARPC train — **≠** BlockGen TinyGSM (1+32 @5/95, T≈0.1) | Quiet/`uniform_dual` are decode probes only. Full TinyGSM recreate stays `xfer_*` / native B3, not a silent U0 default. |
| **SIZE1-EVAL** | Eval at `block_size=1` uses **CE + pure noise (t=1)** — BlockGen-aligned (`algo.py` `_get_loss_type` / `_is_pure_noise`). Continuous-time ELBO at size-1 is **not** the size-1 meter (was the U2 PPL≈459 artifact) | Sweeps annotate `meter=ce_pure_noise` for bs=1; cite size-1 rows again after re-sweep |

## Deep-dive — STILL-TRUE (2026-09-18)

### A. Free-gen soup (not fixed; demote meter)

**What soup is:** open-header `conversion_free` ancestral free-gen that is
word-salad (not unigram collapse). Operational pattern: mid GenPPL + **healthy**
H̄ (clears collapse gate) + **high `full_length_rate`** + qualitative salad.

| Cell | Job | GenPPL | H̄ | FLR | Tag |
|------|-----|--------|-----|-----|-----|
| U0 | 1849335 | 60.8 | 6.04 | 0.83 | `soup_not_fluent` |
| xfer | 1849287 | 55.2 | 6.25 | 0.91 | `soup_not_fluent` |
| U2 | 1848844 | 86.8 | 6.26 | 0.88 | `soup_not_fluent` |
| C0 | 1762534 | 127.9 | 4.80 | 0.91 | soup by eye |
| xfer ARPC armetric | same | 82.7 | 6.35 | 0.98 | still soup; worse than ancestral |

**Primary cause:** open header (no user) is an **OOD Instruct meter** — models
train on prompted turns; long no-EOS wander under ancestral baseline. Chat-
conditional decode / lm-eval is the real competence meter (C2b “still soup”
already rejected for GSM).

**Already failed as fluency fixes:** more NFE (PPL↓, text still salad); ARPC on
xfer (55→83 GenPPL); GenPPL plumbing / prefix-EOS fixes (metrics citeable; soup
persists). DualCache absence on baseline free-gen is **by design**, not a bug.

**Paper:** cite `(PPL,H)` as collapse hygiene only; open free-gen = ablation /
appendix. Never “fluent” / Unifusion quality from GenPPL.

### B. xfer ≠ U0 recipe (structural confound)

**U0 `1849335`:** hooks-off uniform — only `max_steps/length/block/GBS`.  
**xfer `1849287`:** BlockGen **package** on `ar2block` uniform:
`bg_weights_1_32` (1+32 @5/95) + `u-stratified` + `pure_noise_1` + `ce_at_1` +
baked ARPC (`divergence` at launch; registry now `ar_metric`).

Same Nemotron math mix / 6k budget / L=2048 — shared scale does **not** cancel
geometry. Counter-check: `xfer_bg_mix_u` `1855541` (package **without** ARPC)
GenPPL **66.6 > U0 60.8** — so even “mix helps GenPPL” is not clean from
`xfer < U0`.

**Valid:** RQ-B2 / RQ-X transfer tables; xfer_bg_mix_m vs C0; mix_m vs mix_u;
xfer vs N0/B3 (separate tables).  
**Invalid:** claim **K**; matched U0 floor; “uniform better because xfer GenPPL
&lt; U0”; ARPC≈402.

**Open (optional):** no single-lever-on-U0 cell launched; closest registry
preset is `xfer_weights_32` alone. Do not launch unless RQ-B2 needs factor
isolation.

See also: `FAIR_AR2BLOCK_MAP.md`, `BASELINE.md`, **`BAKEOFF_2026-09-24.md` (bake scores)**,
`PAPER_EXPERIMENTS.md`, `LEVERS.md`, `BLOCKGEN_LEVERS.md`,
`UNIFORM_CONFIDENCE_COMMIT_2026-09-24.md`, `configs/paper/cells.yaml`.
