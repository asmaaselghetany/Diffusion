# Deep decode fix pass — 2026-09-23

Concrete bugs found and fixed after the greedy-soup smoking gun.

## Fixed

| Bug | Severity | Fix |
|---|---|---|
| HF unused embed slots (`151666…151935`) in Unif noise → empty `decode` holes (148 hits on hier_ss samples) | HIGH | `unused_embed_slot_ids` in `resolve_uniform_exclude_ids`; fast trailing-hole sample path |
| Auto-fix only set `hierarchical_kv`, left dual `x0[active]=xt` | HIGH | Also force `single_stream_decode` for uniform |
| `coerce` left `hubmatch` / `hierarchical` on uniform | HIGH | **Superseded 2026-09-24:** only `baseline` → `hierarchical`; never silent remap to UCC / `hierarchical_ss` |
| Uniform sub-blocks attended future Unif in same attn block | MEDIUM | `_truncated_active_end` returns `end` (no ceil) on uniform |
| N ancestral + near-noop final ≠ BlockGen last-of-N `α_s=1` | MEDIUM | Last of N is noise-removal; drop extra final |
| DualCache never invalidated on uniform redraws | MEDIUM | Always drop cache when not masked |
| Unif noise could plant EOS → `stop_on_eos` truncates CoT | HIGH | EOS/`im_end` banned from *noise redraws* only (`resolve_uniform_noise_redraw_exclude_ids`); still in `p(x0)` / V_eff |
| GenPPL first_chunk pad with token `0` (`!` on Qwen) | MEDIUM | Pad with EOS |
| GenPPL H̄ prefix strip dead after `skip_special_tokens` | MEDIUM | Strip specials-stripped prefix |
| Submit defaults (`nfe_sweep`, `revision_probe`, lm_eval auto) still baseline | MEDIUM | Default / coerce to `hierarchical` (thr=0.9 remask floor) |
| **Shared open-loop skeleton** (masked *and* uniform) stayed on full-seq dual | **CRITICAL** | Auto-force `hierarchical_kv`+`single_stream` whenever DualCache/sticky off; `allow_full_seq_decode` escape only. C0 ancestral GSM **2.27%** vs DualCache **~64%** proves DualCache was packing+commit, not "uniform-only" |

## Tests

| Suite | Result |
|---|---|
| `test_block_sampler` / `test_decode_profiles` / `test_shift_logits` / `test_uniform_simplex` / `test_arpc_*` / `test_gen_ppl_eos` / `test_uniform_elbo_v_eff` | **67 passed** |

## Not a code hole (still true)

- `1849335` dual-train + weak one-step `acc_moved` (~0.13 @ t=0.5) → fluent-wrong ancestral GSM ~7%
- C0 ~64% is DualCache commit, not ancestral — packing-only canary = `hierarchical_ss` ancestral on `1762534` → job **1972395** (`lm_eval_hier_ss_gsm_canary_20260923`)
- **Train crash (2026-09-23):** `PrunePeriodicCheckpoints` Hydra target shadowed by function re-export in `callbacks/__init__.py` → InstantiationException. Fixed; regression `test_prune_callback_hydra_target.py`.
- Resubmitted: U0_ss_pack **1973301**, U0_ss_shift (see queue), dual resume 1955203 (see queue). Still waiting: sticky 1969396/97, C0 canary 1972395. Hub 1968018 = separate (`wandb` missing / NCCL) — not auto-resubmitted.
