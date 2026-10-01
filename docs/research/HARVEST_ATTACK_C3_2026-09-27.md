# Harvest — attack trains + paradigm floors (2026-09-27)

**As of 2026-10-01:** C3 v2 train **`2125372` COMPLETED** (6k / 11h39, exit 0).
Ckpt: `outputs/block_qwen/ar2block_uniform_2125372/checkpoints/last.ckpt`.
Auto-eval pending (cluster drain): `bqwen-eval` **`2127988`**, GenPPL+H **`2127989`**,
`paper_gen` lm_eval **`2127990`** (ARPC skipped — no size-1 mixture). Scores TBD.

## Eval rule (lock)

Every scored cell runs the **full paper suite** + GenPPL+H — never GSM-only:

| Arm | lm-eval suite | Tasks | Hygiene |
|-----|---------------|-------|---------|
| masked | `paper_acc` | `mmlu,gsm8k,ifeval` | GenPPL+H (`MODE=dual` default; `all` for primary floors) |
| uniform / hybrid | `paper_gen` | `mmlu_generative,gsm8k,ifeval` | same |

`submit_afterok_lm_eval.sh` defaults to suite-full (no `TASKS=` pin).  
GenPPL afterok: `scripts/submit_afterok_gen_ppl_hygiene.sh`.

---

## Attack trains (uniform AR→block)

| Job | Preset | afterok ancestral | afterok ARPC | GenPPL afterok |
|-----|--------|------------------:|-------------:|---------------:|
| `2092151` | `U0_ss_shift` | `2096660` | `2096661` | `2096662` |
| `2092154` | `xfer_bg_mix_32_blockgen_ss` | `2096664` | `2096665` | `2096666` |
| `2092155` | `xfer_bg_mix_32_blockgen_ss_shift` | `2096667` | `2096668` | `2096669` |
| `2094996` → run `2092156` | `xfer_bg_mix_32_blockgen_unifv` | `2097028` | `2097029` | `2104837` |

**Decode:** `hierarchical_ancestral` + `hierarchical_arpc` s32, `FORCE_GREEDY=0`.  
**Compare floor GSM:** hard M/U ARPC **55.6 / 33.4** (`2020048`∥`2020049`).  
**Attack GSM (done):** `U0_ss_shift` **38.1** · `ss_shift` 35.8 · `unifv` 30.8 ·
`blockgen_ss` 28.0 → **`AR2B-U-floor = U0_ss_shift` `2092151`**.  
**Full-suite / hygiene:** `2104833–38` completed (Priority).  
UCC control hard-U: GSM **46.9** / IFE 13.7 (`2092157`) — gsm+ife only; not a floor.

---

## Paradigm floors

### C3 — AR→full-seq diffusion (Unifusion)

| Cell | Recipe | Train | Ancestral GSM | ARPC s32 GSM |
|------|--------|-------|--------------:|-------------:|
| **`C3` v1 fail** | `C3_fullseq` shift-only, `bs=2048` | **`2104831`** | **1.8%** | **4.5%** |
| **`C3` v2 infra fail** | `C3_fullseq_v2` (same recipe) | **`2125144` FAILED** (DataLoader/`torch_shm`; lacked `loader.num_workers=0`) | — | — |
| **`C3` v2 infra fail** | same + `num_workers=0` | **`2125258` FAILED** (missing `.venv/bin/activate` after venv restore) | — | — |
| **`C3` v2 infra fail** | same | **`2125335` FAILED** (missing `pygments` for `rich`) | — | — |
| **`C3` v2 infra fail** | same | **`2125360` FAILED** (HF hub HEAD on airgapped compute; transformers was 4.53) | — | — |
| **`C3` v2 train** | `C3_fullseq_v2` + nw=0 + HF offline + tf 4.45 | **`2125372` COMPLETED** | eval pending `2127988–90` | ARPC skipped |

Hygiene v1: GenPPL≈59 / H≈5.15 / FLR=1 / eos=0 (soup).  
**Root cause:** missed Unifusion §3.2 anneal + single-stream packing (and no Qwen time-cond).  
Forensic: [`C3_UNIFUSION_GAP_2026-09-30.md`](C3_UNIFUSION_GAP_2026-09-30.md).

**Rule:** “full-seq” / C3 = **diffusion only**. Causal AR is never called full-seq.

### `A_ar_sft` — matched causal AR (not full-seq)

| Job | Run | GSM | IFE | MMLU |
|-----|-----|----:|----:|-----:|
| `1857323` | `ar_sft_1857323` | **70.8** | **24.6** | **48.4** |

Tab-1 “stay AR” reference only — not C3.

---

## DualCache twin

`uniform_dualcache` / MASK-lattice twin on Unif weights: **deleted** (OOD).
Uniform remask probe = **UCC** only (`uniform_commit` / `uniform_dual`).
