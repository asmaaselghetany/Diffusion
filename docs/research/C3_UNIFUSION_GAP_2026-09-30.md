# C3 vs Unifusion — where we went wrong (2026-09-30)

Failed floor: `C3_fullseq` job **`2104831`** — GSM ancestral **1.8%** / ARPC **4.5%**.  
Hygiene: GenPPL **≈59**, H̄ **≈5.15**, **eos_rate=0**, **full_length_rate=1.0** (soup that never stops).

Paper: [Unifusion arXiv:2607.24507](https://arxiv.org/abs/2607.24507) (§3.2 + App C).

---

## What Unifusion actually does

| Ingredient | Paper | Our `C3_fullseq` `2104831` |
|------------|-------|----------------------------|
| AR→uniform **direct** | Yes (no mask stage) | Yes |
| **Shift** AR next-token → \(x_0\) | Required | **Yes** (`shift_loss_targets`) |
| **Causal→bidirectional anneal** | Required (~10k steps on GPT-2) | **No** (`intra_block_attn_anneal_steps=0`) — jumped to full bi |
| **Time conditioning** | Learned during adapt (AR had none) | **Missing** — `qwen-block` has no AdaLN / \(t\)-embed |
| Corruption | Linear uniform | USDM / Duo-style (BlockTrainer) |
| Objective | **GIDD** + native sampler | Duo/BlockGen uniform ELBO |
| Geometry | Full-seq **bi over \(x_t\)** | BlockTrainer dual-stream (`single_stream_train=false`) even at `bs=2048` |
| Data | FineWeb continual pretrain (~60B tok window in §4.1) | Nemotron **Instruct** SFT ~3B tok / 6k steps |
| Primary meter | GenPPL–entropy @ **16–256** steps | We scored **GSM/IFE** @ 32 steps (paper never claims Instruct CoT) |

---

## Smoking guns (ranked)

1. **Missing causal→bi anneal** — paper §3.2: start from AR causal pattern, open bi gradually. We already have lever `intra_block_anneal` (`U0_anneal_shift`) but **`C3_fullseq` only listed `[shift]`**. At `block_size=2048`, intra-block **is** the full sequence — anneal is exactly the paper curriculum. We never ran it.

2. **Dual-stream train at one block** — Unifusion is single noisy stream + bi attn. Dual-stream N2C / concat(\(x_t\),\(x_0\)) is a BlockGen/Fast-dLLM packing artifact. `U0_ss_shift` fixed this at bs=32 (38.1% ARPC); **C3 omitted ss**.

3. **No diffusion time conditioning** on Qwen-block — paper explicitly trains \(t\)-cond while keeping AR weights. Residual architecture gap (not fixed by levers alone).

4. **Wrong success meter for the paper recipe** — Unifusion sells GenPPL+H on GPT-2 CPT. Our C3 GenPPL≈59 / H≈5.15 is in a plausible ballpark vs their small-model ~98/5.26, but **FLR=1 + eos=0** = conversion soup; GSM collapse is expected if Instruct CoT was never the Unifusion claim.

5. **ARPC on C3** — no size-1 train mixture; ARPC 4.5% is not a BlockGen-faithful probe.

6. **Objective / schedule** — GIDD vs our USDM ELBO is a secondary fidelity gap (harder to close without a GIDD port).

---

## Fix (version bump — do not silently rewrite failed `C3_fullseq`)

| Cell | Preset | Levers |
|------|--------|--------|
| Historical fail | `C3_fullseq` | `[shift]` only — cite as negative |
| **Retry** | `C3_fullseq_v2` | `[shift, intra_block_anneal, single_stream_train, single_stream_decode]` · `BLOCK=2048` · `loader.num_workers=0` |

| Attempt | Job | Outcome |
|---------|-----|---------|
| v2 first | `2125144` | FAILED exit 143 — DataLoader/`torch_shm` (default `num_workers=2`) |
| v2 + nw=0 | `2125258` | FAILED — `.venv/bin/activate` missing after venv restore |
| v2 + activate | `2125335` | FAILED — missing `pygments` (`rich.syntax`) |
| v2 + deps | `2125360` | FAILED — HF hub HEAD on compute (no `HF_HUB_OFFLINE`; tf was 4.53) |
| v2 resubmit | **`2125372`** | **COMPLETED** 6k / 11h39 — eval `2127988` / hygiene `2127989` / lm_eval `2127990` pending |

Eval for Unifusion fidelity: GenPPL+H NFE sweep (**16/32/64/128/256**), not GSM-only. GSM/IFE still recorded as Instruct probe.

Residual: time-conditioning on Qwen — track separately; v2 does not invent AdaLN.

---

## Refs

- Paper HTML: https://ar5iv.labs.arxiv.org/html/2607.24507  
- Lever: `configs/levers/registry.yaml` (`intra_block_anneal`, `C3_fullseq_v2`)  
- Train curricula: `BlockTrainer._apply_unifusion_curricula`  
- Failed run: `Diffusion/outputs/block_qwen/ar2block_uniform_2104831/`
- v2 run: job **`2125372`**
