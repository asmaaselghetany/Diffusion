# UCC vs masked confidence — forensic (Phase C)

Same-data gap: C0 conf 56% vs U0 UCC 27%. Compared Hub-style paths in
`block_sampler.py`:

| | Masked conf (`_masked_step` / DualCache) | UCC (Hub-clean, 2026-09-25) |
|--|------------------------------|-------------------------------|
| Undecided | MASK | Unif noise (physics) |
| Propose | argmax @ T=0 (`greedy`) | argmax(`p_x0`) — **no Duo scribble** |
| Multi-commit | conf ≥ thr | conf ≥ thr |
| Force ≥1/row | always among masked | `sticky_min_conf=0` (always) |
| Token chosen | sampled (greedy→argmax) | argmax |
| Revise | none | **off** (was BlockGen-ish; not Hub) |
| DualCache K/V | yes (`use_block_cache`) | **MASK-only** — cannot twin |

**Bug / asymmetry fixed:** `sticky_min_conf=0.5` delayed force-max vs masked
Hub generate (always force-max one). Profiles now pin **`sticky_min_conf=0.0`**.

**Packing:**
- `uniform_commit` — B1 remask twin (`ss=false`, `sub=null`, thr=0.9) → bake **26.9%**
- `uniform_dual` — DualCache-*schedule* twin (`ss=true`, `sub=8`, thr=1.0); still Unif undecided, **not** MASK DualCache K/V
- `uniform_commit_ss` — thr=0.9 + Hub ss (old unfair twin)

**Deleted:** `uniform_dualcache` (MASK undecided on Unif weights) — OOD garbage.

**Hub-clean (2026-09-25):** skip Duo posterior on sticky; keep Unif on
non-commits; `revise=false`; `greedy=true` allowed under sticky.

Remaining same-data risks: Unif undecided state, L′=1 untrained ARPC,
T=1 vs quiet T=0.1 — Phase A/B. Cite post-attack floor from
`AR2BLOCK_GSM_TABLE.md` (`U0_ss_shift` 38.1% ARPC), not bake UC alone.
