# Hard audit: uniform pipeline (2026-09-21) — **do not queue until locked**

Deep pass after the first MASK/`V_eff`/`hierarchical` fix round. Compared again
to BlockGen (`third_party/blockgen`), Duo (`s-sahoo/duo`), and our Qwen path.
Queue stays empty until the items below are accepted.

Companions: `DEBUG_U0_UNIFORM_SKELETON_2026-09-21.md`,
`REFERENCE_UNIFORM_IMPLS_2026-09-21.md`.

---

## Executive verdict

| Claim | Status |
|-------|--------|
| U0 ~6% GSM was partly **decode/skeleton**, not only “uniform weak” | Confirmed |
| First fix pass was **incomplete / self-inconsistent** | Confirmed — ELBO sum bug |
| Safe to re-queue Instruct eval on old U0 ckpts now? | **Decode-only**, with `hierarchical` |
| Safe to claim new train↔sample contract? | **No** until U0 retrain after ELBO lock |

---

## Critical findings (this pass)

### 1. ELBO `V_eff` coeffs + full-`V` `logsum` (was broken; now fixed)

After banning MASK and passing `V_eff=V−1` into `uniform_block_nll_per_token`,
the Duo term still did `x_bar_theta.log().sum(-1)` over width `V`. With
`p_MASK≈0`, the MASK slot equals `1−α` and injects extra `log(1−α)` into the
sum (~2–3% relative bias in probes).

**Fix:** `uniform_block_nll_per_token(..., mask_id=)` subtracts the dead MASK
log term when `width > V_eff`. Callers: `_uniform_loss`, `_hybrid_loss`.
Unit test: `tests/test_uniform_elbo_v_eff.py` (fixed ≡ compact).

### 2. lm-eval `baseline` undid uniform hierarchical (was broken; now fixed)

`BlockSampler` auto-enabled `hierarchical_kv` for uniform; then
`DECODE_PROFILE=baseline` pinned it **back to false** → full-seq Unif attention
again. Family submit defaults to `hierarchical`, but any baseline entry point
reopened the smoking gun.

**Fix:** after profile pins, if `forward_process_name==uniform` and scope is
still full-seq, force `hierarchical_kv=true` (`block_qwen_lm_eval.py`).

### 3. Old U0 ckpt ≠ current train contract

`1849335` trained under `Unif(V)` + ELBO `V` (internally consistent).
Current WT: `Unif(V\{MASK})` + ELBO `V_eff` (now algebraically consistent).
→ **Retrain U0** for a locked paper train cell. Re-eval of old weights only
supports **decode-protocol** claims.

---

## Verified OK

- Duo/BlockGen posterior algebra structure (fast + naive with `V_eff`)
- MASK-excluding FP / prior / init / ARPC redraw (`sample_uniform_excluding_mask`)
- Shift-align + MASK/PAD ban on `_uniform_step`
- Greedy refuse on uniform
- `active_len` truncation really drops future tokens on dual-stream Qwen
- Prefix continuation windows (`denoise_start`, restore)
- Family eval default profile `hierarchical`, greedy=0
- α-schedule / ignore_bos / dual-stream train wiring
- Queue empty (user cancelled)

---

## Remaining (accepted or lower priority)

| Item | Severity | Action |
|------|----------|--------|
| Dual-stream ≠ BlockGen `[prefix\|block]` packing | Medium | Accepted residual; hierarchical is the Qwen fix |
| Hub padded vocab in Unif support (~0.18%) | Medium | Document; optional remap later |
| `sub_block_size` on uniform would re-expose intra-block Unif | Medium | Keep null on hierarchical profile |
| `parameterization: subs` label on uniform yaml | Low | Cosmetic |
| True BlockGen generate path | Low / future | Separate work |
| Fast float64 BlockGen sample pin | Low | Optional parity |

---

## Test status (this session)

- `test_uniform_elbo_v_eff`, `test_uniform_simplex`, sampler/shift/decode profiles: **pass**
- `test_loss_symmetry` updated for Unifusion shift-on-uniform + MASK ban
- Queue: **empty** — do not submit until you sign off

---

## Before re-queue (checklist)

1. [x] ELBO MASK sum fix + unit test
2. [x] lm-eval cannot unpin uniform hierarchical
3. [x] Fast path respects `p_nucleus` when &lt; 1
4. [x] Forensic: U0 GSM 6% was baseline + hier=false + force_greedy=1
   → see `HARD_AUDIT_UNIFORM_GSM5_FORENSIC_2026-09-21.md`
5. [x] Profile coerce + generate() re-assert + ifeval/GenPPL defaults → hierarchical
6. [ ] Decide: **(A)** decode-only re-eval old U0 under hierarchical, or **(B)** retrain U0 first
7. [ ] If (A): cite as “old weights + fixed decode”, not new train contract
8. [ ] If (B): new `submit_lever U0` then family eval hierarchical

**Recommendation:** do **(A)** for one U0 + U0+shift smoke first (GSM only / short suite) to see if decode fixes move the needle; then **(B)** for the locked paper cell.
