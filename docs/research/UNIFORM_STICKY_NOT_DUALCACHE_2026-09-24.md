# Uniform pipeline residual — sticky ≠ DualCache (2026-09-24)

## Symptom

`uniform_dual` thr=1 lifted GSM flex **~6% → ~34%** on `1849335` (job
1969396) but:

- still ≪ C0 DualCache **~64%**
- **strict=0** is a red herring (C0 DualCache also strict=0; harness asks `\boxed{}`)
- samples: English CoT with **stutter / glued BPE** and systematically wrong algebra

## Smoking gun (code)

Masked DualCache: confidence-until loop with `_apply(t=1, dt=None)` until no MASK.

Old `uniform_dual`: fell through to **fixed-N mid-α ancestral** Unif redraws, with
sticky as a post-filter. At thr=1 almost nothing thresholds → **force-max ~1
token/step** from a diffuse `p_x0`, while uncommitted siblings stay Unif noise
that single-stream still attends.

## Fix

`_denoise_block`: when `uniform_confidence_sticky` + `unmask_threshold`, mirror
masked — confidence-until with noise-removal only until the window is frozen
(`U0-STICKY-CONF` in DESIGN_LOCKS).

Regression: `test_uniform_sticky_denoise_uses_confidence_until_not_ancestral`.

## Still residual (not this fix)

- Scored ckpts dual-stream / Unif(V)-era (`1849335`, `1955203`); need `U0_ss_pack`
- Uncommitted sites ≠ MASK (neutral); even α_s=1 commits can't fully erase that
- BlockGen TinyGSM recipe ≠ Instruct U0 (`TINYGSM-RECIPE`)

## A/B (closed)

Sticky→UCC confidence-until shipped; bake UC on `1955203` = **26.9%** GSM.
Hard-U `uniform_dual` control = **46.9%** (`2092157`). MASK DualCache twin
path deleted. Living scoreboard: `BAKEOFF_2026-09-24.md` · `AR2BLOCK_GSM_TABLE.md`.
