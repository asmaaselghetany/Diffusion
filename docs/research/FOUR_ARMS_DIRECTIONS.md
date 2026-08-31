# Four-arm `block_qwen` bakeoff — status and directions

Scope of this note: **only** the four clean-graph arms and the follow-on
directions for turning that line into a **thesis chapter** (main baseline +
extras). No DepBench thesis, growblock, or other side tracks.

Canonical WandB (eval + samples + curves):  
https://wandb.ai/asmaaselghetany-elghitany/block_qwen_final

Local runs: `outputs/block_qwen/{ar2block,block}_{masked,uniform}_13976{0,1,2,3}/`

---

## 1. The four arms

Shared neutral recipe (`configs/experiment/block_qwen.yaml` + hooks **off**):

- Base: Qwen2.5-1.5B-Instruct (AR-init) or same arch from scratch  
- Data: Nemotron post-training SFT  
- Seq 2048, block 32, LR 2e-5, warmup 500, global batch 256, **6000** steps  
- Only intentional difference within a row: **corruption** (masked vs uniform)

| Arm | Init | Corruption | Run dir | gen-PPL (gpt2-large) |
|-----|------|------------|---------|----------------------|
| `ar2block_masked` | AR Instruct | masked | `…_139760` | ~311 |
| `ar2block_uniform` | AR Instruct | uniform | `…_139761` | ~264 |
| `block_masked` | scratch | masked | `…_139762` | ~794 |
| `block_uniform` | scratch | uniform | `…_139763` | ~331 |

**Design intent:** controlled 2×2 (init × corruption). Literature anchors:
Fast-dLLM-style AR→**masked** block; BlockGen-style scratch masked/**uniform**;
AR→**uniform** block still sparse. Neutral bakeoff deliberately omits Fast-dLLM
hooks (see `LEVERS.md`).

---

## 2. Honest readout (what these four runs show)

| Signal | Reading |
|--------|---------|
| Train / val NLL, ELBO@32 | AR-init looks “OK” (~1.6 NLL); scratch worse (~3) |
| Free-gen / gen-PPL | Soup; gen-PPL hundreds |
| Leftover `[MASK]` in free-gen | ~0% — not a trivial unmask bug |
| Pipeline / low-t sanity | Plumbing looks coherent → **correct-but-insufficient** more likely than catastrophic broken sampler |
| Uniform vs masked (gen-PPL) | Uniform slightly better under this recipe (same direction as BlockGen anecdotes) |
| Scratch vs AR | AR ≫ scratch at this budget — expected |

**Do not claim from these four alone:** fluent block dLLM, Fast-dLLM parity, or
a finished AR→uniform success story.

**Reasonable claim:** under a matched hooks-off skeleton, all four cells train,
but generation quality is not competitive; loss ≠ usable samples.

---

## 3. Soup: implementation vs missing recipe?

Working judgment (not proven to 100%):

| Hypothesis | Likelihood |
|------------|------------|
| Catastrophic sampler / mask wiring bug | **Low** (sanity + no mask leftovers + sane NLL) |
| Neutral hooks-off + smaller throughput than Fast-dLLM → soup | **High** |
| Need levers **and** scale, not one boolean | **High** |
| Soft mismatch vs Fast-dLLM paper details | **Low–medium** until a positive-control run |

**Decisive check:** one AR-masked run closer to Fast-dLLM (levers + token budget).
If quality jumps → soup was recipe. If still soup → hunt soft impl gaps.

---

## 4. Possible directions (only)

### D0 — Keep the four as baseline (no new claim)

Use current runs as **Fig.1 / Table.1**: matched skeleton, hooks off, generation fails.
Thesis contribution lives in D1–D3 below, not in these numbers alone.

### D1 — Scale-only (both arms)

Same four cells, **no hooks**; raise tokens (e.g. 2× batch and/or more steps) toward
Fast-dLLM throughput (~batch 256, ~6k steps, ~3B tokens in their writeup).

- **If** AR-masked improves a lot → budget was primary; levers study gets sharper.  
- **If** still soup → recipe (not steps alone) implicated.

Corruption-agnostic; preserves fair masked↔uniform compare.

### D2 — Recipe / lever transfer study (recommended thesis spine)

**Framing:** *Published block systems work because of recipe, not the skeleton alone —
and recipe pieces are not all corruption-symmetric.*

Pre-registered order (do not junkyard):

1. Scale (D1) on both corruptions.  
2. **Agnostic train levers** on both: `block_size_mixture` (± `stratified_gamma`).  
3. **Masked-only** track (label as asymmetric): `shift_loss_targets`, `complementary_masks`
   (± partial within-block if implemented). Code policy: refuse these on uniform.  
4. **Uniform-oriented** decode: mixture including block size 1 + `sampling.use_arpc`.

Same gates every time: gen-PPL, sample readability, optional fixed task.
Promote a lever only with go/no-go in `LEVERS.md`.

**Scientific questions:**

- Which levers are necessary for fluency?  
- Which transfer masked↔uniform?  
- Does uniform need a different stack than masked (BlockGen-style)?

### D3 — Fluency-gated 4-arm bakeoff

Only after **at least AR-masked** clears a fluency gate under a declared recipe:
re-run (or resume) the full 2×2 under **agnostic** levers only, and report masked vs
uniform fairly.

Without that gate, a matched four-arm bakeoff from soup is not viable as a thesis
section.

### D4 — Positive control (decision experiment)

Single arm: `ar2block_masked` + Fast-dLLM-like hooks + closer token budget.

- Success → pursue D2/D3; treat current four as intentional under-recied baseline.  
- Failure → debug soft implementation vs paper before more lever fishing.

### Explicitly out of scope for this note

DepBench / joint-k-tax thesis, growblock, harness-only stories, unrelated micros
except as go/no-go evidence for a lever row.

---

## 5. What applies to masked vs uniform

| Lever / change | Masked | Uniform |
|----------------|--------|---------|
| More batch / steps / tokens | yes | yes |
| `block_size_mixture`, `stratified_gamma` | yes | yes |
| `shift_loss_targets`, `complementary_masks` | yes | **no** (masked-only) |
| Partial within-block masking (if built) | yes | no (as Fast-dLLM) |
| `use_arpc` | no | yes (decode; best with size-1 in mixture) |

Details: `docs/research/LEVERS.md`, `docs/research/BLOCKGEN_LEVERS.md`, `docs/research/BLOCK_QWEN_TRAINING.md`.

---

## 6. Suggested default path

1. Freeze current four arms as **hooks-off baseline** (done).  
2. Run **D4** (or D1 then D4) before claiming anything about “implementation vs recipe.”  
3. If positive control works → **D2** as the thesis spine; use **D3** only once fluency exists.  
4. Do **not** turn on all Fast-dLLM knobs on masked only and call that the four-arm thesis baseline.

---

## 7. Pointers

| Doc | Role |
|-----|------|
| `docs/research/BLOCK_QWEN_TRAINING.md` | How to launch / eval the arms |
| `docs/research/LEVERS.md` | Lever registry + go/no-go |
| `docs/research/BLOCKGEN_LEVERS.md` | Mixture / ARPC micros |
| `tools/run_block_qwen_eval.py` | Post-train eval (samples, gen-PPL, DepBench, ELBO) |
