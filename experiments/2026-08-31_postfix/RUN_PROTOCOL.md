# Post-fix run protocol (2026-08-31)

**Repo:** `Diffusion-new` on JUWELS (4 nodes × 4 GH200 = 16 GPUs for paper cells)  
**Scope:** Four-arm neutral grid + axis hooks + eval. Pre-fix checkpoints are **out of scope** for strict tables.

**Families:** conversion (`LINE=ar2block`) | native (`LINE=block`, BlockGen home) |
transfer (`xfer_*` on `ar2block`, never tagged BlockGen). See
[`PAPER_EXPERIMENTS.md`](../../docs/research/PAPER_EXPERIMENTS.md).

---

## 1. Metric roles (pre-registered)

| Role | Metric | When | Caveat |
|------|--------|------|--------|
| **Primary (conversion)** | lm-eval GSM8K + IFEval (+ code) with chat template | All `LINE=ar2block` / Instruct claims | Matches Fast-dLLM-style reporting |
| **Primary (native)** | Free-gen samples + val BPD/ELBO (+ ARPC for BlockGen) | All `LINE=block` / OWT claims | Matches BD3/BlockGen practice |
| **Secondary** | Val BPD / ELBO, t-bucketed NLL | Training monitor | Can mis-rank vs task accuracy (RQ4) |
| **Free-gen (both)** | `sample_mode=auto` + `decode_profile=baseline` | Offline samples / gen-PPL | conversion→`conversion_free` header; native→`native_free`; **not** bare BOS |
| **Appendix** | gen-PPL (gpt2-large) on those samples | BlockGen Table 8–style | Bare-BOS legacy samples are invalid for Instruct |
| **Exactness overlays** | DualCache+thr=0.9 (conversion); ARPC (native) | Recipe cells only | Not the default free-gen path |
| **Throughput** | tok/s with `hierarchical_kv` / DualCache | Axis E | DualCache = algo port; not fused Fast-dLLM CUDA |

**Adopted sampling policy:** see `.cursor/rules/sampling-family-policy.mdc` and
`configs/eval/generate_samples.yaml`. Canonical code tree: **`Diffusion-new`**.

---

## 2. Training data (post-fix)

- **Source:** Nemotron post-training SFT (`configs/data/sft_qwen.yaml`)
- **Format:** `use_chat_template=true` → Qwen `apply_chat_template` (aligns train with lm-eval)
- **Packing:** wrap + `_abl32` attention-block padding
- **Cache tag:** `_chatTpl` in `.dat` filename — rebuild cache after enabling template

Rollback to Alpaca markdown: override `/data: sft_qwen_alpaca` and `data.use_chat_template=false`.

---

## 3. Decode presets per claim type

| Claim | Family | Train cell | Eval sampling |
|-------|--------|------------|---------------|
| Neutral / fair free-gen | either | any | **`sample_mode=auto decode_profile=baseline`** (default) |
| Fast-dLLM conversion exactness | **conversion** | `fastdllm_v2` / `C2_fdllm_full` | lm-eval: `DECODE_PROFILE=dual_cache UNMASK_THRESHOLD=0.9 FORCE_GREEDY=1` |
| BlockGen uniform + ARPC | **native** | `blockgen_owt_uniform` / `B3_arpc` | `arpc_decode_eval`: `native_free` + `decode_profile=keep` + ARPC pins; **`line=block`** |
| Geometry on conversion | **transfer** | `xfer_arpc` / `xfer_mixture` | Same ARPC knobs; **do not tag BlockGen** |
| Progressive decode (speed) | either | any ckpt | `decode_profile=hierarchical` |
| DualCache throughput | either | any ckpt | `decode_profile=dual_cache` (label: not fused CUDA) |
| Legacy bare-BOS ablation | conversion | any | `sample_mode=bare_bos` only — not headline samples |

**Not claimed:** bit-for-bit Fast-dLLM fused kernels; BlockGen 170M scale; labeling `ar2block`+BlockGen knobs as BlockGen.

---

## 4. Phase schedule

### Phase 0 — Gate (login node)

```bash
source /e/project1/scifi/elsayed3/env.sh
cd "$REPO_ROOT"
.venv/bin/python -m pytest tests/ -q
bash scripts/download_block_qwen_data.sh   # rebuilds _chatTpl cache
```

### Phase 1 — Four-arm neutral (post-fix v1)

```bash
sbatch scripts/slurm/ar2block_masked.sbatch
sbatch scripts/slurm/ar2block_uniform.sbatch
sbatch scripts/slurm/block_masked.sbatch
sbatch scripts/slurm/block_uniform.sbatch
```

Label: **C0/B1/B2 neutral post-fix v1**.

### Phase 2 — Axis D (conversion hooks)

```bash
./scripts/submit_paper_cell.sh C2_shift
./scripts/submit_paper_cell.sh C2_comp
./scripts/submit_paper_cell.sh C2_fdllm
./scripts/submit_paper_cell.sh C2_fdllm_full
# optional:
./scripts/submit_paper_cell.sh C5_joint_ar
./scripts/submit_paper_cell.sh C5_causal_clean
```

### Phase 3 — Native BlockGen (line=block) + optional transfer

```bash
# Native reference + BlockGen micros
./scripts/submit_paper_cell.sh N0
./scripts/submit_paper_cell.sh B3_mixture --arm masked
./scripts/submit_paper_cell.sh B3_mixture --arm uniform
./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform --paper
./scripts/submit_blockgen_owt.sh   # OWT 1+16 recreate
./scripts/submit_paper_cell.sh B3_arpc --arm uniform

# Transfer only (geometry on conversion — not BlockGen)
./scripts/submit_lever.sh --preset xfer_mixture --arm masked --paper
./scripts/submit_lever.sh --preset xfer_arpc --arm uniform --paper
```

### Phase 4 — Hybrid + AR baseline

```bash
./scripts/submit_paper_cell.sh B4_hybrid_p10
./scripts/submit_paper_cell.sh C3
```

### Phase 5 — Eval-only

```bash
CKPT=.../best.ckpt ./scripts/submit_paper_cell.sh lm_eval
CKPT=.../best.ckpt ./scripts/submit_paper_cell.sh C4
CKPT=.../best.ckpt ./scripts/submit_paper_cell.sh decode_hierarchical
```

---

## 5. Checkpoint comparability

| Artifact | Valid for post-fix tables? |
|----------|----------------------------|
| Pre-fix four-arm bakeoff (Aug 2026) | **No** — uniform MASK, no chat template, pipeline fixes |
| Post-fix Phase 1 runs | **Yes** — B1/B2 neutral |
| ARPC on bakeoff ckpts | **No** — need `mixture_1_32` training |

---

## 6. Honest labeling (design-space instrument)

- Neutral four-arm ≠ Fast-dLLM v2 or BlockGen at default settings.
- `C2_fdllm` ≠ full recipe; use `C2_fdllm_full` for positive control.
- Complementary masks = sequential 2×B forwards (grad-equivalent approx).
- `sub_block_size` + `parallel_threshold_decode` = closer Fast-dLLM unmask; still not fused parallel kernel.
- BlockGen-inspired at 1.5B / 6k steps / lr 2e-5.
