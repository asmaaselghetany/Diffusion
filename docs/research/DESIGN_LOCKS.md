# Design locks (closed — not open work)

Accepted decisions for the thesis path. These are **not** TODOs.
Do not reopen without a new cell ID / version bump.

| ID | Decision | Consequence |
|----|----------|-------------|
| **FAMILIES** | Three experiment families: **conversion** (`ar2block`), **native** (`block`), **transfer** (`xfer_*` on `ar2block`) | Registry refuses Fast-dLLM levers on `block` and native/BlockGen levers on `ar2block` unless preset is `xfer_*`. Never tag transfer as BlockGen. |
| **B4v1** | Hybrid trains with mask+Unif surrogate ELBO; **decode-as-masked** (prior = all MASK) | No hybrid reverse process in v1. Cell `B4_hybrid_p10` / `B4_hybrid_p50` (conversion family). |
| **C5 streams** | `C5_joint_ar` = block-clean dual-stream AR; `C5_causal_clean` = token-causal AR | Only the latter supports causal-NLD wording. Metrics: `val/nll` = diffusion; `val/joint_nll` = objective. |
| **DualCache** | `use_block_cache` is K/V-only replace_position **approximation** | Quality evals use `E_hierarchical` / truncate-only. Tok/s probes may use `E_dual_cache`. |
| **Systems tok/s** | Our BlockSampler ≠ Fast-dLLM fused / SGLang numbers | Always label; never claim parity. |
| **BlockGen home** | BlockGen recreate / B3 native cells are **`LINE=block` only** | Use `submit_blockgen_owt.sh` / `B3_*`. Jobs 1660577/1660931 discarded as BlockGen claims. |
| **P6 scheduling** | Native B3 + BlockGen OWT are **wired**; prefer interpreting C0/C2 before writing conversion results | Launch order is operator choice; keep family tables separate. |

See also: `PAPER_EXPERIMENTS.md`, `LEVERS.md`, `BLOCKGEN_LEVERS.md`, `configs/paper/cells.yaml`.
