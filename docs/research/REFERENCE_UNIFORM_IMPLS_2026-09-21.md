# Reference uniform implementations vs our Qwen conversion (2026-09-21)

Compared against GitHub / vendored refs used by the thesis:

| Ref | Source | Uniform? |
|-----|--------|----------|
| **BlockGen** | `third_party/blockgen` = [jdeschena/blockgen](https://github.com/jdeschena/blockgen) | Yes (`UniformState`) |
| **Duo** | [s-sahoo/duo](https://github.com/s-sahoo/duo) `algo.py` | Yes (full-seq USDM) |
| **UDLM** | our `sampling/uniform.py` + `algo/udlm.yaml` (legacy) | Yes |
| **Unifusion** | paper only (no public LM code found) | Uniform + **shift** |
| **Fast-dLLM** | `third_party/Fast-dLLM` | Masked only |

BlockGen / Duo **never add a MASK token** on the uniform arm. Our conversion
**always keeps MASK** for the shared Qwen masked arm. Therefore the correct
paper alignment is **`Unif(V \\ {MASK})` with `V_eff = V-1` everywhere** —
not literal BlockGen `Unif(V)`.

---

## End-to-end contract (after this fix pass)

| Layer | BlockGen / Duo | Ours (fixed) |
|-------|----------------|--------------|
| FP draw | `Unif(V)` | `Unif(V\\{MASK})` via `sample_uniform_excluding_mask` |
| Prior | `randint(0,V)` | same helper |
| ELBO `V` | full `vocab_size` | **`V_eff`** + ban MASK logits (`_uniform_loss`) |
| Posterior coeffs | `V` | **`V_eff`** |
| Limiting | `1/V` | `1/V_eff` on non-MASK, 0 on MASK |
| Posterior sampler | **fast** (BlockGen default) | **fast** default (`naive` opt-in) |
| Logit prep | plain softmax | shift-align + ban MASK/PAD |
| Block geometry | `[prefix \| active block]` only | **hierarchical_kv** (auto if baseline) truncates `active_len` |
| Greedy | N/A / illegal for USDM | forced ancestral |
| float64 | sample scripts `true` | default `false` (parity pin optional) |

---

## Critical mismatches that caused U0 ≪ C0

1. **Full-seq Unif attention** on `DECODE_PROFILE=baseline` (BlockGen never sees future noise).
2. **`_uniform_step` skipped shift + MASK ban** (breaks U0+shift; allows MASK in draws).
3. **MASK-as-noise + `skip_special_tokens`** deleted tokens → salad CoT.
4. **ELBO used full `V` while FP moved to `V\\{MASK}`** (fixed now).
5. **Posterior used full `V` with limiting `1/(V-1)`** (aligned to `V_eff` now).

---

## Files touched

- `forward_process/utils.py` — `uniform_simplex_size`, `sample_uniform_excluding_mask`
- `forward_process/block_uniform.py`, `uniform.py`, `block_hybrid.py`
- `algorithms/block_trainer.py` — `_uniform_loss` + `prior_sample`
- `sampling/block_sampler.py` — V_eff, fast posterior, auto hierarchical
- `configs/sampling/block.yaml` — `posterior_sampler: fast`
- `evaluations/decode_profiles.py`, `block_qwen_lm_eval.py`
- `scripts/submit_family_eval.sh` — default profile `hierarchical`
- `tests/test_uniform_simplex.py`

---

## Still not bit-identical to BlockGen (accepted)

- Dual-stream train graph vs BlockGen DiT generate `forward(x0=prefix, xt=block)`.
- Hierarchical truncation mitigates; a true BlockGen single-stream generate path is future work.
- Existing U0 ckpts were trained under rare-MASK-in-V FP; decode now uses `V_eff`. Retrain after this pass for full train↔sample lock.
- Temperature: BlockGen TinyGSM paper mentions T=0.1; we expose `p_nucleus` / ARPC T only.

---

## Re-eval plan

Re-run all uniform Instruct cells with `DECODE_PROFILE=hierarchical`, `FORCE_GREEDY=0`, fixed sampler, after cancelling stale jobs that started under the partial fix.
