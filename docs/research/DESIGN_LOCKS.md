# Design locks (closed — not open work)

Accepted decisions for the AR→block thesis path. These are **not** TODOs.
Do not reopen without a new cell ID / version bump.

| ID | Decision | Consequence |
|----|----------|-------------|
| **B4v1** | Hybrid trains with mask+Unif surrogate ELBO; **decode-as-masked** (prior = all MASK) | No hybrid reverse process in v1. Cell `B4_hybrid_p10` / `B4_hybrid_p50`. |
| **C5 streams** | `C5_joint_ar` = block-clean dual-stream AR; `C5_causal_clean` = token-causal AR | Only the latter supports causal-NLD wording. Metrics: `val/nll` = diffusion; `val/joint_nll` = objective. |
| **DualCache** | `use_block_cache` is K/V-only replace_position **approximation** | Quality evals use `E_hierarchical` / truncate-only. Tok/s probes may use `E_dual_cache`. |
| **Systems tok/s** | Our BlockSampler ≠ Fast-dLLM fused / SGLang numbers | Always label; never claim parity. |
| **P6 scheduling** | B3/B4/C5/E decode cells are **wired and ready** | Prefer interpreting C0/C2 first when writing results; launch order is operator choice, not a stub. |

See also: `PAPER_EXPERIMENTS.md`, `LEVERS.md`, `configs/paper/cells.yaml`.
