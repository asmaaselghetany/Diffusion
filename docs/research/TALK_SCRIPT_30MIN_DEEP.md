# Talk script — ~30–35 min (deep dive)

Casual speaking notes for `DLMA_BlockDiffusion_Progress.pptx`.  
Numbers: `PRESENTATION_SNAPSHOT_2026-09-18.md`. Say caveats out loud.

**How to use:** each block is ~what you say. Bracketed notes are for you, not for the audience.

---

## 0:00–3:00 — Open: what problem am I even solving?

Hey — thanks for the time. I’m going to go deep today, so I’ll move a bit slower on the knobs and recipes.

Here’s the practical problem. We have strong **autoregressive instruct** models — in our case Qwen2.5-1.5B-Instruct. Discrete **diffusion** language models are interesting because they can denoise in parallel inside a window, revise tokens, and (depending on the forward process) support different editability stories. But the literature is a mess of overlapping claims:

- Fast-dLLM: convert AR → **block** diffusion with a bunch of training hooks and a fancy decode stack.  
- BlockGen: often **scratch** block diffusion with mixture block sizes, ARPC, etc.  
- Unifusion / Duo-style uniform diffusion: different corruption, shift as an \(x_0\) interface, GenPPL hygiene.  
- Plus MDLM / GIDD / remasking stories about editable tokens.

If you just say “we did block diffusion like the papers,” you can’t tell whether a win came from **initialization**, **loss hooks**, **corruption**, **block geometry**, or **decode**. So the thesis contribution I’m defending today is not “we beat Hub on a leaderboard.” It’s:

> We treat AR→block as a **design space**, keep families on different lines, ablate components one factor at a time where it matters, and only then talk about quality, editability, and what’s still unknown.

Shared backbone for almost everything I’ll cite: Qwen2.5-1.5B-Instruct, Nemotron post-training SFT, sequence 2048, block size 32, AdamW 2e-5, global batch 256, **6000 steps** (~3B tokens), bf16. When I say “matched,” I mean that protocol unless I explicitly break it.

---

## 3:00–7:00 — Research questions (full wording)

### Conversion family (`LINE=ar2block`)

**RQ1 — conversion recipe.**  
Which conversion-axis choices — pretrained init, SFT token budget, Fast-dLLM-style dual-stream training hooks, optional joint AR objective — determine whether AR→block produces *usable* generation at instruct SFT scale?  
“Usable” means conditional tasks (GSM8K, IFEval, MMLU-style) plus non-collapsed free-gen, not just val loss going down.

**RQ2 — vs matched AR.**  
After the same Nemotron budget, does block conversion beat **matched causal AR SFT** (no blocks) on task accuracy and on parallel-decode throughput?  
That’s the fairness column. If AR SFT wins tasks and we only win speed, that’s still a story — but we have to measure both.

**RQ3 — confounding in literature.**  
Do published systems confound conversion choices with block-native choices — geometry, corruption, decode tricks? Can we separate them by keeping conversion on `ar2block` and native BlockGen-style work on `LINE=block`, with optional `xfer_*` as a *third* claim tag?

**RQ4 — metrics.**  
Does validation NLL/BPD track conditional generation and GenPPL, or mislead? We’re paranoid about that because diffusion likelihood and free-gen quality diverge a lot.

### Native family (`LINE=block`)

**RQ-N1.** At fixed (matched) budget, does scratch block diffusion produce non-soup text, and how does it compare to conversion **as a different paradigm** — not as a failed convert?

**RQ-N2.** On native only: do mixture / u-stratified / CE@1 / ARPC move quality?

### Extras under the conversion skeleton

**RQ-B1 / Claim K.** Holding the same AR→block skeleton, does masked vs uniform change gen quality or only likelihood ranking?  
Mechanism language is gated: taxonomy of failures first; causal editability only with inject/remask protocols. We are **not** claiming we discovered editable tokens — prior art exists; we ask whether it explains *our* local Instruct failures under matched C0 vs U0.

---

## 7:00–10:00 — Families, lines, and the “never mix” rule

Three families in the registry — this is operational, not vibes:

| Family | Line | Examples | Claim tag |
|--------|------|----------|-----------|
| Conversion | `ar2block` | C0, C2_*, C5, B1, B4 | AR→block |
| Native | `block` | N0, B2, B3, blockgen_owt | scratch / BlockGen |
| Transfer | `ar2block` + native knobs | `xfer_mixture`, `xfer_arpc` | `xfer_*` only |

Wrong family → job refused. That’s intentional.  
Calling `xfer_arpc` “BlockGen” in a talk is a claim error even if the code runs.

**Neutral conversion reference C0:** Instruct init, masked, block 32, **all fluency hooks off**, Nemotron 6k.  
Everything else is “C0 plus named change” or a different paradigm.

---

## 10:00–14:00 — Paradigms in depth

### Paradigm 1 — Native (scratch block)

Train block diffusion without Instruct conversion (or without relying on it). At our matched Nemotron 6k budget, scratch masked is roughly **GSM 0.5 / IFE 8 / MMLU 23**.  

What that means in the talk:  
- Conversion’s jump to GSM ~62 is **not** “diffusion is magic.” It’s largely **AR initialization + SFT budget**.  
- Outcome C in the pre-registered list: *initialization dominates* at matched budget.  
- Do **not** cite free-gen GenPPL≈402 from a collapsed scratch run — prefix/EOS failure, not a paradigm score.

Native BlockGen recreate (OWT scale, mixture, etc.) is a separate table. Different compute, different claim.

### Paradigm 2 — AR→block (main)

Load Instruct weights, train with block diffusion objective, fixed block 32.

**C0 baseline masked (hooks off):** GSM **62.1**, IFE **20.5**, MMLU **~39.8** (hubll), GenPPL **128** with H̄≈4.80.  
Hypothesis C1a/C1b from the plan: loss drops and generation becomes English-shaped, but neutral hooks-off does **not** automatically match Fast-dLLM fluency — so Tab 2 exists.

**Same paradigm, data variants with the “all five” component pack** (defined later):

- Math mix → GSM **66.8**, IFE 22.4, GenPPL 89 — best GSM.  
- Chat mix → GSM ~43 — same knobs, different data, big drop.  
- Mixed SFT → GSM ~60 — middle.

So: paradigm fixed, components fixed, **data still moves GSM by 20+ points**. Never present “recipe” without data.

**Also under AR→block but different corruption:** baseline uniform (B1) — same skeleton, hooks off. IFE 10.7 on the scoreboard is **asterisked** (greedy bug). GenPPL ~61 with higher entropy — hygiene, not fluency.

**Geometry cross-cut, still AR→block:** block-size mix masked (~63 GSM) — don’t call it a new paradigm.

### Paradigm 3 — AR→full-seq diffusion (C3 / Unifusion floor)

Full-sequence **diffusion** from AR Instruct init (`block_size=model.length`).
**“Full-seq” always means diffusion** — never causal AR SFT.

Matched causal AR SFT is a **separate** paradigm cell **`A_ar_sft`** (Tab-1
“stay AR” column), not C3 and not called full-seq. Floor already scored:
`ar_sft_1857323` GSM **70.8**.

Why C3 matters: fair Unifusion-style column vs AR→block (C0/U0). If C3 beats
block on tasks, geometry (full-seq vs block) is doing work — not “stay AR.”

### Paradigm 4 — Joint (diffusion + AR auxiliary)

Objective sketch: \(L \approx L_{\text{diff}} + \alpha L_{\text{AR}}\) with \(\alpha\) around 0.3 in our C5 cells.  
Important lock: logged `val/nll` for cross-cell comparison stays **diffusion-only**; `val/joint_nll` is the optimized mix. Don’t compare joint_nll to C0’s val/nll as if identical.

**Joint AR masked** (joint + causal clean): GSM **38.5** ↓, IFE **25.0** ↑, MMLU **42.0** ↑.  
**All five + joint:** GSM ~44, IFE ~21, MMLU ~30 — packing joint on top of Fast-dLLM pack is not automatically best.

Joint is a **paradigm** because it changes the training objective family, not just a mask schedule.

---

## 14:00–24:00 — Every component and every named recipe (slow section)

*[This is the densest part. Pause after each knob. Point at the ablation table.]*

Neutral C0 has these **off**. We only promote a lever to “shared recipe” after pre-registered go/no-go — and several have **not** been promoted; they stay Track-2 / cell-specific.

---

### Component 1 — Shift (`algo.shift_loss_targets`)

**Mechanics.**  
Without shift: at position \(i\), score the clean token \(x_0[i]\) (classic \(x_0\)-prediction at the same index).  
With shift: logits at \(i\) score clean token \(x_0[i+1]\) — same index shift Fast-dLLM uses for AR alignment; Unifusion discusses this as an \(x_0\) interface choice.

Implementation detail worth one sentence: when shift is on, sequences shorten by one in the loss (T→T−1); we had a real bug (TBUCKET-SHIFT-TRIM) when val t-bucket logging didn’t trim the pad mask — fixed. So shift isn’t only conceptual; it touches shapes everywhere.

**Decode coupling.**  
If you train with shift, decode should align logits the same way (`align_shift_logits` / shift mode in the sampler). Train/decode mismatch silently hurts.

**Why we include it.**  
Instruct AR already predicts next token. Block diffusion asks for bidirectional denoise inside a block. Shift is the soft bridge so the head isn’t forced to forget AR geometry overnight.

**Recipe: C2_shift / C0_shift (alias).**  
Single factor on masked AR→block.  
**Result:** GSM **64.0** (↑ from 62.1), IFE ~21.1, MMLU ~33.7 (↓), GenPPL **88** (↓ with H̄ still reasonable).  
**Line to say:** “Shift is our cleanest single-knob win: better math-ish GSM and better collapse hygiene, with a MMLU trade.”

**Recipe: U0_shift.**  
Same flag on **uniform** corruption (Unifusion-style naming).  
Early free-gen looked *worse* than U0; size-1 CE worse; generative suite still queued.  
**Line:** “Shift is not automatically portable to uniform in our block-Instruct setting — don’t universalize Tab 2.”

---

### Component 2 — Complementary masks (`algo.complementary_masks`)

**Mechanics.**  
Fast-dLLM v2 style: each training step uses a **pair** of masks \(m\) and \(\tilde{m}\) (fused batching: two sequential forwards of batch B, not a polarity bug).  
Old bug (COMP-POLARITY): we used to flip `~move_mask` in place — wrong vs Hub paired views; BPD blew up (~24). **Discard any readout from that era.** Current path: Hub fused 2B `m`/`~m`.

**Why.**  
More diverse mask coverage per optimizer step; Hub’s complementary objective.

**Recipe: C2_comp.**  
**Result:** GSM **53.3**, IFE **18.1**, GenPPL ~108 — **down** vs C0.  
**Line:** “Complementary alone hurt on our neutral skeleton. It’s not a free fluency button.”

**Policy.**  
Complementary is **masked-only** in our special-case policy — refused on pure uniform (doesn’t make sense the same way).

---

### Component 3 — Shift + complementary together (strict Tab-2 / `C2_fdllm`)

**Recipe name in docs:** `C2_fdllm` = shift + complementary only (strict Tab 2).  
Not yet “full Hub.”

**Result:** GSM **50.8**, IFE **17.0**, MMLU ~32, GenPPL ~107 — **worse than shift alone and worse than comp alone**.  

**This is the rhetorical climax of Tab 2:**  
> Interactions dominate. If I had only reported “Fast-dLLM hooks on,” I’d have concluded the hooks fail — but shift alone helps. That’s why we ban “full recipe” cells as the only ablation.

---

### Component 4 — Mask schedule (`algo.mask_schedule=fast_dllm`)

**Mechanics.**  
Default schedule: noise α from log-linear (or our α path) with sampling ε floor.  
Fast-dLLM schedule: sample \(t \sim U(0,1)\) then \(p_{\text{mask}} = (1-\varepsilon)t+\varepsilon\) (Hub).  

**Bug we fixed (FAST-DLLM-DOUBLE-EPS):** we used to floor \(t\) to \([\varepsilon,1]\) *and* apply the Hub map → minimum \(p \approx 2\varepsilon\). Now fast_dllm uses raw \(U(0,1)\) t. Mild but real C2 bias before the fix.

**Also:** under fast_dllm we turn antithetic sampling off to match Hub’s i.i.d. block times.

**Why.**  
So ELBO weights / α used at train and the Hub-like decode calendar aren’t fighting each other.

**Usually packaged inside `C2_fdllm_full`,** not a solo scoreboard row — say that when someone asks “where’s the schedule-only number?”

---

### Component 5 — Plain CE (`algo.loss_weighting=plain_ce`)

**Mechanics.**  
Masked ELBO path: continuous-time style coefficients involving \(\dot\alpha / (1-\alpha)\), etc.  
Plain CE: unweighted CE on mask sites only; mean denominator = **mask count**, not all valid tokens.

**Bug we fixed (PLAIN-CE-DENOM):** numerator zeroed clean sites but denominator used all valid tokens; with complementary doubling valid, effective loss ~½ Hub scale at same LR → C2 looked unfairly weak (~10 pp GSM). Fixed with `_plain_ce_token_count`.

**Why.**  
Hub Fast-dLLM training is CE-flavored on masks. ELBO weighting changes gradient scale. Plain CE is “Hub loss shape,” not a new forward process.

**Policy:** plain_ce is masked-only in our refuse list for uniform.

---

### Component 6 — Hub train parity pack

Not one bool — a small bundle, typically on `C2_fdllm_full` / math-mix:

1. **`hub_struct_attn_only`** — Hub overwrites attention with structural block-diffusion mask only; pads stay visible. We used to apply pad blocking → different softmax on packed MASK pads (HUB-STRUCT-ATTN).  
2. **`ignore_bos=false`** — Hub parity on BOS handling.  
3. Related: **SFT-ATTN-PROMPT fix** — we once passed assistant-only labels as `attention_mask`, so SDPA couldn’t attend to the user/system prompt. Hub keeps prompt visible and uses labels=-100 to drop CE on prompt. That bug hurt **both** C0 and C2 absolute levels.

**Line to say:**  
> Some of our “recipe” work is actually **bug parity with Hub**. Without naming those fixes, Tab 2 is uninterpretable historically.

---

### Recipe A — `C2_fdllm_full` / “all five” on math mix

**Pack:** shift + complementary + fast_dllm schedule + plain_ce + hub train parity.  
**Data:** math-oriented Nemotron mix (math mix cell).  

**Result:** GSM **66.8**, IFE 22.4, GenPPL 89, MMLU hubll ~30.6 (↓ vs C0’s ~39.8). HumanEval with chat template: baseline HE 34.1 → math-mix 36.0.

**How to narrate:**  
- Best GSM we have under conversion.  
- Not Pareto-optimal: MMLU can fall while GSM rises.  
- Hypothesis C2a partially supported: missing axis D *and* data explain a lot vs neutral C0 — but strict shift+comp alone did *not* improve, so “Fast-dLLM hooks” is too coarse a phrase. Prefer naming the pack.

**Same pack, chat mix / mixed SFT:** GSM 43 / ~60 — recipe×data interaction.

---

### Component 7 — Joint AR (`algo.joint_ar_alpha`)

**Mechanics.**  
Add α times an AR NLL on a clean stream to the diffusion loss.  
We log `val/ar_nll`, `val/diff_nll`, `val/joint_nll` separately.

**Why.**  
Keep next-token skill while learning block denoise (NLD-adjacent on the map).

**Design lock:** don’t use joint_nll as the only number when comparing to C0.

---

### Component 8 — Causal clean (`algo.causal_clean_stream`)

**Mechanics.**  
The AR auxiliary is computed with a **token-causal** pass on clean tokens — closer to true AR than a bidirectional “clean” encode.

**Recipe: C5 joint + causal clean (Joint paradigm row).**  
GSM 38.5 ↓, IFE 25 ↑, MMLU 42 ↑.  

**Recipe: all five + joint.**  
Still not best GSM; packing paradigms+components is another interaction surface.

**Line:**  
> Joint buys instruction/format/MMLU-ish behavior back; it taxes GSM on our runs. That’s a trade, not a bug — unless your title claim is GSM.

---

### Component 9 — Intra-block attention anneal

**Mechanics.**  
`intra_block_attn_anneal_steps`: over the first N training steps, intra-block attention interpolates from more causal → fully bidirectional.

**Why.**  
Curriculum: don’t destroy AR inductive bias on day 0; open the block gradually.

**Recipe: anneal-only on masked AR→block.**  
GSM ~61.1 (≈C0), IFE ~21.4, MMLU **42.7** (best in that ablation column), GenPPL ~134 (slightly worse hygiene).

**Uniform + anneal (U0_anneal):** queued — don’t invent numbers.

---

### Component 10 — Kernel anneal (hybrid only)

**Mechanics.**  
`kernel_anneal_steps`: on hybrid forward process, ramp \(p(\text{uniform})\) from 0 → target (e.g. 0.5) over training.

**Why.**  
Hybrid = MASK with prob \(1-p\), Unif(V\{MASK}) with prob \(p\). Starting at high \(p\) is a hard kernel shock.

**Recipe: B4 hybrid + anneals.** GenPPL hygiene ~106 on disk; full lm-eval queued.

**Policy:** kernel anneal refused unless `forward_process_name=hybrid`.

---

### Decode-side “components” (axis E — measure, don’t train)

These are **not** training paradigms, but they change reported scores:

| Decode knob | What it does | When we use it |
|-------------|--------------|----------------|
| `baseline` | Ancestral block sampler, greedy false, thr null | Default free-gen / uniform |
| `hubmatch` | hierarchical KV + single-stream + sub-block 8 + **thr=1** + greedy | Paper accuracy for **masked** |
| `dual_cache` | + DualCache splice | Speed/accuracy Hub-like |
| `NUM_STEPS` / NFE | 8/16/32/64 | Quality–compute curves |
| `ban_mask_pad_logits` | Safety; Hub often false for parity probes | Don’t silent-change between cells |

**Critical uniform rule:** confidence unmask is masked physics. On uniform, **greedy argmax of \(q(x_s\mid x_t)\)** preferentially keeps \(x_t\) when \(p_{x_0}\) is flat → locks Unif(V) prior → soup. Our IFE 10.7 used FORCE_GREEDY=1 on baseline profile — **invalid**. Sampler now warns and forces ancestral; rescore pending.

---

## 24:00–28:00 — Corruption, geometry, GIF, Claim K

### Corruption deep dive

**Masked / absorbing.**  
Corrupt toward MASK. Model learns \(p(x_0 \mid x_t)\) especially on mask sites (SUBS-style parameterization in our masked loss). Decode can commit high-confidence positions and remask/iterate — hubmatch thr=1 forces progress so you don’t stall.

**Uniform.**  
Corrupt by redrawing from Unif(V) (our block-uniform FP shares \(t\) within a block). Loss is DUO/UDLM-style uniform ELBO (`uniform_block_nll_per_token`), **not** SUBS mask CE — even though the config string still says `parameterization: subs` historically; the forward_process_name routes the loss.  
Hybrid loss even bans MASK from the uniform simplex and uses \(V-1\); pure uniform still draws including mask id rarely — minor compared to greedy bug.

**Hybrid (B4).**  
Third pole (GIDD-inspired). Decode-as-masked in B4v1 lock. Not the abstract lead.

**Matched skeleton comparison:** C0 IFE 20.5 vs U0 10.7* — star the star.

### Geometry

Fixed-32 is the conversion default.  
Mixture / weights / u-stratified / pure-noise-at-size-1 / CE@1 / ARPC live primarily on **native** BlockGen recreate; `xfer_*` tests whether those knobs help **conversion** without renaming the claim.  
Block-size mix on ar2block masked ≈ C0 GSM — geometry ≠ automatic win.

### GIF narration (~90–120s)

Left: masked hubmatch — MASK canvas, gold = active block, confidence commits.  
Right: uniform ancestral — no MASK skeleton; tokens rewrite inside the window; early fluent, later drift.  
Say unfairness once, then move: “geometry of uncertainty visualization, not bake-off.”

### Claim K

Only C0 vs U0. Label L1/L2/G1/G2 on fails. Go ≥30% local share on C0. Sheets under `editable_tokens/`. No causal slogan until taxonomy + inject/remask.

### GenPPL hygiene (Unifusion reporting rule)

Every free-gen number is **(PPL, H̄[, H̃])**.  
NFE sweep 8/16/32/64; dual first_chunk vs full; collapse panel.  
Uniform can look “better PPL” with higher entropy — less collapsed, not more fluent.  
Never cite GenPPL≈402, ARPC-as-fluency, xfer-as-U0.

---

## 28:00–32:00 — Status, takeaways, Q

### Status board (be boring and honest)

**Ready to cite:** scratch floor; C0; math/chat/mixed packs; Tab-2 ablations; joint masked; block-size mix masked; Claim K export sheets; decode viz; hygiene protocol.  

**Cite with asterisk:** U0 IFE 10.7; GenPPL rankings without H; GIF as score claim.  

**Pending:** ancestral U0 IFE; U0 GSM/MMLU; C3 AR SFT scores; U0_shift / U0_anneal / hybrid generative; continue-FT U0_from_C0 only if gated.  

**Never without external reproduction:** Hub leaderboard parity; “uniform is the better paradigm” as title.

### Closing takeaways (hit all five)

1. **Conversion works** under matched budget because init+SFT dominate scratch — 0.5 → ~62 → ~67 GSM with math pack.  
2. **Components are causal and interactive** — shift helps; complementary alone hurts; shift+comp can be worse than either; “Fast-dLLM hooks” is not one bit.  
3. **Paradigm ≠ component ≠ corruption ≠ geometry ≠ eval** — that’s the map vs Fast-dLLM/BlockGen/Unifusion mashups.  
4. **Uniform is a real axis** for Claim K, but decode/scoring bugs can fake “soup.”  
5. **Hygiene ≠ fluency** — always (PPL, H).

Thanks — I can go deeper on any single recipe: shift vs all-five, joint tradeoffs, complementary polarity history, or the uniform posterior greedy footgun.

---

## Pocket Q&A (extra depth if asked)

**Q: Why not only report C2_fdllm_full?**  
A: Because Tab 2 shows the minimal dual hook pack regresses; full pack + math data wins GSM — you need both single-factor and packed recipes or you mis-attribute.

**Q: Is plain CE “less principled” than ELBO?**  
A: For Hub parity and gradient scale, CE-on-masks is what that stack trains. We still have ELBO path for uniform/native comparisons; don’t mix meters silently (size-1 eval uses CE+pure-noise BlockGen-style).

**Q: What’s the difference between joint paradigm and shift?**  
A: Shift reindexes the diffusion target toward next-token. Joint adds a second loss term with its own stream. Orthogonal; combining is another experiment.

**Q: Why refuse complementary on uniform?**  
A: Complementary is a paired-mask construction for absorbing FP. Uniform redraw doesn’t have the same mask complementarity; we’d be inventing a different method.

**Q: Continue-FT masked→uniform?**  
A: Gated Unifusion-motivated probe (two-stage ≈ direct on their full-seq GenPPL). Only if direct U0 underperforms C0 on the metrics we care about — not the default path.

**Q: Throughput / DualCache?**  
A: Axis E supporting. hierarchical_kv + dual_cache approximate Hub progressive decode; still not identical fused kernels — label carefully in comparisons.
