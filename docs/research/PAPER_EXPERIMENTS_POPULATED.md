Axes, paradigms, components, and recipes (plain language)

This section explains every axis / paradigm / hook / recipe we use, assuming you have never seen them. No table of numbers here; those start after this guide.


The big picture in one breath

You start with a normal chat LLM (Qwen Instruct). That model writes text left to right, one token at a time. You want a block diffusion model instead: it works on chunks of tokens (blocks), and can denoise many positions inside a block with more parallel / revisable generation.

So the thesis is basically: what choices matter when you convert AR to block diffusion, and what is just noise from mixing up unrelated papers (Fast-dLLM, BlockGen, Unifusion, NLD).

To keep that honest, everything is factored into axes (big questions), paradigms (training path / objective family), components (train knobs / "hooks"), plus a few cross-cuts that are not hooks (noise, geometry, how you evaluate).


Families first (before axes)

Three families. Mixing them silently is how papers get confusing.

Conversion (LINE=ar2block)
Load a pretrained Instruct model and continue training it as block diffusion. Almost all of the "can we match Fast-dLLM?" story lives here. Reference cell: C0.

Native (LINE=block)
Train block diffusion from scratch (or BlockGen-style setups). This answers "is conversion special, or is block diffusion just good?" At our matched budget, scratch is weak (GSM ~0.5). That is important: conversion works largely because of AR init + SFT budget, not magic.

Transfer (xfer_*)
Take BlockGen-style geometry knobs (mixtures, ARPC, etc.) and stick them onto conversion. Useful experiment. Not a BlockGen recreate claim. Do not rank xfer against U0 as if they were the same recipe.


Axis A. Init and budget (and data mix)

Question: What are you converting from, for how long, and on what data?

Think of this as the fuel and starting point, not the engine design.

Init.
Usually: Qwen2.5-1.5B-Instruct weights already know English, chat format, some math/code. Starting from that is why conversion jumps from scratch GSM ~0.5 to C0 GSM ~62. The model is not learning language from noise; it is relearning how to generate under a diffusion objective.

Budget.
Canonical paper cells: ~6000 steps x global batch 256 ≈ ~3.15B tokens of Nemotron-style SFT. Longer or shorter budget is still axis A. Matched AR SFT (C3) is also axis A in spirit: same data/steps, but stay causal AR instead of becoming block diffusion. That answers "did we need blocks for accuracy, or only for parallel decode / editability?"

Data mix (the painful part of A).
Fast-dLLM cites a Nemotron post-training subset without publishing ratios. We use explicit caps, e.g.:
- math-heavy: lots of math (+ code)
- chat-heavy: chat / safety / science

Same training knobs, different mix -> Pareto front:
- math mix: GSM 66.8, MMLU ~30
- chat mix: MMLU 40.3, GSM ~43
- IFEval stuck ~20-25 either way (Hub ~44)

So axis A is why you can beat Hub on one metric and lose on others. Merges and continue-FT are still axis A experiments: trying to combine mixes without knowing Hub's secret blend.

In one sentence: Axis A is who you start as, how long you train, and what you feed it.


Axis D. Components / training recipe / "hooks"

Question: How do you adapt the AR backbone into block diffusion during training?

Neutral C0 = conversion with all fluency hooks off. Everything below is "C0 plus named change(s)."

These are train-time choices. They change the loss, masks, attention structure, or schedule. They are not "use DualCache at eval."


Component 1: Shift (shift_loss_targets)

Without shift: at position i, the model is trained to predict the clean token that belongs at i (classic x0 prediction).

With shift: logits at i are trained to predict the clean token at i+1. That is the AR next-token geometry Fast-dLLM uses, and Unifusion talks about as an x0 interface.

Why it matters: Instruct AR already thinks "next token." Block diffusion asks for bidirectional denoising inside a block. Shift is a soft bridge so the head is not forced to forget next-token geometry overnight.

Train/decode coupling: if you train with shift, decode should align logits the same way. Mismatch silently hurts.

Our result: shift alone -> GSM 64.0 (up from 62.1), GenPPL better, MMLU down a bit. Cleanest single-knob win on masked conversion. On uniform (U0_shift) it is not automatically good; do not universalize.


Component 2: Complementary masks (complementary_masks)

Masked diffusion: each step you hide some tokens with MASK. Complementary means each optimizer step uses a pair of masks m and ~m (Hub-style fused batching: two views), so you cover more mask patterns per step.

This only makes sense for absorbing / masked noise. Registry refuses it on pure uniform.

Our result: complementary alone hurt (GSM 53.3). Not a free fluency button. Old polarity bug made BPD explode; those runs are trash.


Component 3: Shift + complementary together (strict C2_fdllm)

People say "Fast-dLLM hooks" like it is one switch. Strict Tab-2 dual pack is shift+comp only.

Our result: worse than either alone (GSM 50.8). Interactions dominate. If you only reported "hooks on," you would wrongly conclude the hooks fail, while shift alone helps.


Component 4: Mask schedule (mask_schedule=fast_dllm)

This remaps diffusion time t into a mask rate (how much of the sequence is masked).

Hub-style: sample t ~ U(0,1), then p_mask = (1-eps)t + eps. We also turn off antithetic sampling under this schedule to match Hub's i.i.d. block times.

Why: so training's noise calendar matches the Hub-like decode story. Usually packaged inside the full recipe, not a solo scoreboard row. We fixed a double-eps bug that made minimum mask rate ~2eps.


Component 5: Plain CE (loss_weighting=plain_ce)

Two loss philosophies on masked training:

- ELBO-style weighting: continuous-time coefficients (alpha-dot stuff). Principled likelihood path, different gradient scale.
- Plain CE: unweighted cross-entropy on mask sites only, denominator = mask count (Hub CE flavor).

Why: Hub Fast-dLLM training is CE-on-masks shaped. Plain CE is "match Hub's loss shape," not a new forward process.

We had a nasty denom bug: numerator ignored clean sites but denom counted all tokens; with complementary that halved the effective loss scale -> C2 looked ~10 GSM points too weak. Fixed. Masked-only policy for plain_ce on uniform refuse list.


Component 6: Hub train parity (a small pack, not one bool)

Not a single clever idea. A bundle of "stop disagreeing with Hub's training wiring":

1. hub_struct_attn_only: attention uses Hub's structural block-diffusion mask; pads stay visible (we used to block pads differently).
2. ignore_bos=false: BOS handling matches Hub.
3. Related fix SFT-ATTN-PROMPT: we once fed assistant-only labels as attention_mask, so the model could not attend to the user prompt. Hub keeps prompt visible and uses labels=-100 to drop CE on prompt. That bug hurt both C0 and C2 absolute scores.

Honest line: some "recipe work" was really bug parity with Hub. Without naming those fixes, old tables are uninterpretable.


Recipe pack: C2_fdllm_full / "all five"

Pack: shift + complementary + fast_dllm schedule + plain_ce + hub train parity.
Plus data (axis A): math-oriented mix.

Result: GSM 66.8 (best), IFE ~22, MMLU ~30. Same pack on chat mix: GSM ~43. So "recipe" and "data" interact hard. Prefer naming the pack, not the vague phrase "Fast-dLLM hooks."


Component 7: Joint AR (joint_ar_alpha) (also a paradigm change)

Add alpha times an autoregressive NLL on a clean stream to the diffusion loss. We log ar_nll, diff_nll, joint_nll separately.

Why: keep next-token skill while learning block denoise (NLD-adjacent idea).

Important: do not compare val/joint_nll to C0's val/nll as if identical. Diffusion-only NLL stays the cross-cell comparable meter.

This is big enough that we treat Joint as its own paradigm row, not "just another tiny lever."


Component 8: Causal clean (causal_clean_stream)

How the AR auxiliary is computed: a token-causal pass on clean tokens (true AR-like), not a bidirectional clean encode.

Usually paired with joint. Our joint+causal_clean (chat mix): GSM down (38.5), IFE up (25), MMLU up (42). Buys format/instruction-ish behavior; taxes GSM. Trade, not a bug.

All-five + joint is another interaction surface; not automatically best.


Component 9: Intra-block attention anneal

Inside a block, attention can be causal or bidirectional. Anneal: over the first N training steps, interpolate from more causal to fully bidirectional.

Why: curriculum. Do not smash AR inductive bias on day 0; open the block gradually.

Our anneal-only run: GSM ~C0, MMLU 42.7 (best in that ablation column), GenPPL slightly worse. Uniform+anneal still queued for fair numbers.


Component 10: Kernel anneal (hybrid only)

Hybrid noise mixes MASK and uniform redraw with probability p. Kernel anneal ramps p(uniform) from 0 to target (e.g. 0.5) over training so you do not shock the model with a hard uniform kernel from step 0.

Refused unless forward process is hybrid.


Axis E. Decode evaluation (how you measure)

Question: How do you turn weights into the numbers you put in a table?

Same checkpoint can look smart or stupid depending on the sampler. That is why E is an axis: measure carefully, do not silently change decode between cells.

Main knobs:

baseline profile
Ancestral block sampling, typically no confidence threshold, greedy off. Default for free-gen and for uniform (see below).

hubmatch profile
Closer to Hub accuracy settings for masked: hierarchical KV, single-stream style decode, sub-block size 8, unmask threshold = 1, greedy. This is our paper accuracy protocol for masked conversion.

dual_cache
Adds DualCache-style caching/splice on top. Speed/accuracy Hub-like algorithmically; still not identical to Fast-dLLM fused CUDA kernels. Label tok/s carefully.

NFE / NUM_STEPS
How many denoising steps (8/16/32/64). Quality-compute curve. More steps is not automatically a fairer comparison unless matched.

Threshold / greedy
On masked: high confidence commit + remask is natural. thr=1 forces progress so you do not stall.

Critical uniform footgun:
On uniform, greedy argmax of the reverse transition can prefer keeping the current noisy token when the clean prediction is flat -> locks Unif(V) prior -> soup. Our U0 IFE 10.7 was under FORCE_GREEDY=1 and is invalid. Ancestral (no greedy) is the fair baseline for uniform.

Cross-stack check: exporting our weights into Hub's eval.py matched our BlockSampler roughly -> residual Hub gap is probably not "our sampler is secretly broken" (points back at axis A).

GenPPL + entropy: free-gen PPL without entropy is dangerous. Low PPL + high H can mean "less collapsed," not "more fluent." Hygiene meter, not English quality.


Axis C. Noise (forward process)

Question: What noise process do you train the diffusion against?

This is not a Fast-dLLM hook. It is the physics of the forward process.

Masked (absorbing) (our main track)
Corrupt toward a special MASK token. Model learns to fill masks (SUBS-style mask-site CE in our stack). Decode can commit confident positions and remask others. Hubmatch thr=1 fits this physics.

C0 lives here. Most Tab 2 ablations live here.

Uniform
Instead of masking, redraw tokens from a uniform distribution over the vocab (our block-uniform shares noise level t within a block). Loss is uniform ELBO style (uniform_block_nll_per_token), not mask CE, even if some config strings still say subs historically; forward_process_name routes the real loss.

Generation looks different: no MASK canvas; tokens rewrite inside the window; early text can look fluent then drift (our decode GIFs).

Matched floor to C0 is U0 (same math mix, hooks off). Old chat-only uniform runs are not fair.

Hybrid
Sometimes MASK, sometimes uniform redraw (GIDD-inspired third pole). Kernel anneal belongs here. Decode-as-masked is a lock for some B4 setups. Extra / appendix unless results force otherwise.


Axis B. Block geometry

Question: How do you chop the sequence into blocks, and do special sizes get special treatment?

Conversion default: fixed block size 32. Simple, matched across C0/C2.

BlockGen-native world adds more:
- mixtures of block sizes
- weighted size draws
- u-stratified schedules
- pure noise at size 1
- CE@1 special loss cases
- ARPC sampling tricks

Those are native-home ideas. xfer_* asks: do they help conversion if transferred? Different claim. Our block-size mix on masked conversion looked ~C0 GSM (~63). Geometry is not an automatic win and must not be renamed as "better conversion recipe."


Paradigms (training path / objective family)

Paradigms sit above components. Changing paradigm is bigger than flipping one bool.

Native
Train block diffusion without Instruct conversion (or without relying on it). Floor shows conversion's jump is mostly init+budget.

AR to block
The main story: Instruct to block diffusion. C0 / C2 / U0 live here.

AR to full sequence
Matched causal AR SFT (C3): same data/steps, no blocks. Fair "did we need diffusion?" column. Still waiting on scores.

Joint
Diffusion + AR auxiliary (joint_ar +/- causal_clean). Changes the objective family. Our numbers: better IFE/MMLU, worse GSM on the runs we have.


Named recipes as bundles of the above

| Name | Roughly means |
|------|----------------|
| C0 | AR to block, masked, block 32, hooks off, math-ish Nemotron budget. Neutral conversion floor. |
| C0_shift / C2_shift | C0 + shift only. |
| C0_comp | C0 + complementary only. |
| C2_fdllm | shift + complementary only (strict Tab 2). |
| C2_fdllm_full | all five Fast-dLLM-ish train knobs (+ hub parity). Still needs a mix specified. |
| C5 | Joint paradigm (+ causal clean). |
| C3 | Matched AR SFT (axis A / paradigm AR to full). |
| U0 | Same as C0 skeleton but uniform noise. |
| U0_shift / U0_anneal | U0 + that one lever. |
| U0_from_C0 | Continue-FT from masked C0 into uniform (gated two-stage). |
| B4 | Hybrid + anneals. |
| N0 | Native scratch floor. |
| xfer_* | Conversion + BlockGen geometry package. |


How to read interactions (so you do not get fooled)

1. Axis A can dwarf axis D. Same full pack, chat vs math: GSM 43 vs 67.
2. Hooks interact. Shift helps; comp hurts; both together can be worse than either.
3. Paradigm != component. Joint is not "another schedule."
4. Noise != recipe. Uniform U0 is not "C0 with Fast-dLLM off"; it is different physics.
5. Eval can fake failure. Greedy-on-uniform soup is an axis E bug, not proof uniform cannot work.
6. Hygiene != fluency. GenPPL without entropy lies.


Tiny mental model

- A = start + fuel + data diet
- D = how you teach the AR model to denoise (hooks / loss / attn curriculum)
- E = how you grade the exam
- C = what kind of noise you trained against
- B = how you cut the sequence into blocks


---

Thesis experiments working notes (populated)

This is the filled version of PAPER_EXPERIMENTS.md. Design stays in that file. This one is my "what do we actually have" dump so I can write the thesis without digging through Slurm every five minutes.

Live job mess lives in experiments/2026-09-07_lm_eval_nccl_timeout/EXPERIMENT.md.

Do not put pre-gate runs in any paper table. That means polarity-flip complementary, wrong plain_ce denom, SFT attention mask bug where the prompt was invisible, etc. Those runs exist. They are wrong. Move on.

How I score when I quote numbers unless I say otherwise: DECODE_PROFILE=hubmatch, FORCE_UNMASK_THRESHOLD=1, greedy, max_new_tokens=2048, scores from SUMMARY.json. IFEval = prompt-level strict. GSM8K = flexible-extract. For uniform/hybrid I need paper_gen (mmlu_generative, gsm8k, ifeval). Likelihood MMLU on Unif(V) is garbage, do not cite it.


What I can actually say

Families have to stay separate. Conversion is ar2block. Native scratch is block. Transfer is xfer_* (BlockGen geometry stuck onto conversion). Cross-wiring BlockGen knobs onto conversion and calling it a BlockGen recreate is not allowed. Registry already refuses most of that nonsense.

Axis D (training knobs / conversion recipe) matters. After we fixed plain_ce denom, SFT attention mask, and hub-struct attn, C2 math beats or matches C0 on GSM at the same budget. So "we forgot the Fast-dLLM hooks" was partly true for early broken runs, and is mostly closed now for GSM.

GSM can hit Hub 1.5B levels under conversion (~62 to 67 vs Hub ~63). Cool. IFEval and MMLU do not.

Axis E (decode) is probably not the residual Hub gap. We exported an earlier C2eb into Hub eval.py and got roughly the same GSM/IFEval as our BlockSampler (~38 / ~19). So the remaining gap looks like data/weights (axis A), not "our sampler is secretly broken compared to theirs."

Axis A is still open and annoying. The Fast-dLLM paper cites a Nemotron post-training subset without the mix ratios. Chat-heavy vs math-heavy mixes form a Pareto front (MMLU vs GSM). IFEval sits around 20 to 25 forever while Hub is ~44 to 47. Linear merges and continue-FT chat to math do not close all three at once.

Main write-up pair for conversion: C0 (hooks off) + C2 full math mix. Chat mix and merges are supporting evidence for the Pareto story.


What I cannot claim yet

Matched C3 AR SFT for Tab 1. Train is 1857323 after 1856100 CVD fail and 1857271 cancel. Still waiting.

Hub parity on IFEval / MMLU / code jointly. Not happening with current mixes.

Matched uniform U0 / U2 / xfer generative suites. Trains exist. Full paper_gen scores still landing. Old B1 1836729 was chat-only so it is not a fair floor against math-mix C0. Fair floor is U0 1849335.

U0_from_C0 continue-FT (1857519) is gated. Only run the story if direct U0 is clearly worse than C0. Do not invent a curriculum win early.

Longitudinal capability recovery figure (§7). Not assembled. Do not draw fake curves.


Snapshot (post-gate, hubmatch unless noted)

| Model / cell | Job / ckpt | MMLU | GSM8K | IFEval | Notes |
|--------------|------------|------|-------|--------|-------|
| Hub Fast-dLLM v2 1.5B (external) | our recheck + paper | 53.5 | ~62.8 | ~44.4 | target I keep missing jointly |
| C0 / Conv-Base | 1762534 | 39.8 | 62.1 | 20.5 | hooks off; hubmatch |
| C2 / Conv-Full math | 1763301 | 30.6 | 66.8 | 22.4 | main recipe row; best GSM |
| C2 / Conv-Full chat | 1773298 | 40.3 | 43.4 | 23.8 | same knobs, different mix |
| merge a=0.3 | 1836666 | 38.5 | 64.5 | 22.9 | best GSM merge |
| merge a=0.5 | 1836667 | 39.3 | 59.9 | 21.1 | |
| merge a=0.7 | 1836668 | 41.4 | 55.0 | 24.6 | best MMLU/IF merge |
| continue-FT chat→math | 1836800 | 38.1 | 59.8 | 23.3 | |
| C5_causal_clean | 1836723 | 42.0 | 38.5 | 25.0 | chat mix; IFE/MMLU up, GSM down |
| C2+C5 | 1836725 | 29.8 | 44.0 | 21.1 | packing everything does not win |
| native masked floor | 1836796 | 23.0 | 0.5 | 8.1 | scratch at matched budget is sad |
| U0 uniform≈C0 math mix | 1849335 | - | - | 10.7* | *FORCE_GREEDY=1 invalid; ancestral rescore pending |
| U2 uniform joint-AR | 1848844 | - | - | - | scoring; not BlockGen |
| xfer_blockgen_uniform_32 | 1849287 | - | - | - | transfer hooks + ARPC; not matched U0 |
| xfer_bg_mix_32 masked | 1855537 | 40.2 | 63.3 | 20.7 | C0 skeleton + 1+32 mix |
| xfer_bg_mix_32 uniform | 1855541 | - | - | - | U0 skeleton + 1+32 mix |

U0 GenPPL first_chunk ~61 with H~6.0 is hygiene only, not "better English."

Code (hubmatch, chat template) on same C0 / C2 math ckpts:

| Cell | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|-----------|------------|------|-------|
| C0 1762534 | 34.1 | 29.3 | 27.2 | 41.5 |
| C2 math 1763301 | 36.0 | 32.9 | 24.8 | 43.1 |
| Hub paper (approx.) | ~44 / ~40 | - | ~50 / ~41 | - |

Best mix for new uniform / transfer cells: chat+safety+science+math1M+code500k (same as masked C0 / C2-math). Chat-only uniform is not a matched floor. I already burned time on that mistake.


Research questions, provisional

RQ1 conversion recipe: partial yes. Axis D hooks improve GSM vs neutral C0 once train bugs are fixed. Remaining failure vs Hub is not "forgot shift/complementary." It is data mix / unpublished subset (axis A) plus IFEval/MMLU still being bad.

RQ2 vs matched AR: open. C3 wired, no post-gate three-way table yet.

RQ3 literature confounding: design answered (registry + families). Empirical native tables still thin.

RQ4 metrics: partly. Val BPD and free-gen soup are bad sole meters for Instruct conversion. Conditional lm-eval is the claim surface. GenPPL is diagnostic / hygiene only.

Native extras RQ-N1 / N2 / B1 / B2 / X: not result-ready for main tables.


Axes in plain words

| Axis | Name | Status | Evidence |
|------|------|--------|----------|
| A | Init and data budget | Open (main bottleneck) | Pareto chat vs math; Hub ratios unknown; merges/continue-FT tried |
| D | Conversion recipe / train knobs | Mostly closed for GSM | C2 math >= C0; Tab 2 masked filled; U0 levers queued |
| E | How we measure | Mostly closed for masked | hubmatch thr=1; uniform greedy bug patched; rescores in flight |
| B | Block geometry | Early | Native/OWT and xfer exist; do not smuggle into conversion Tab 1 |
| C | Noise | Partial | Main = masked; fair B1 = U0 vs C0; hybrid B4 landing |

C0 reference: Qwen2.5-1.5B-Instruct → block, masked, block 32, hooks off, Nemotron SFT ~6000×256 (~3.15B tok), Hub-keep vocab 151936.


Shared protocol for canonical rows

| Setting | Value |
|---------|-------|
| Backbone | Qwen2.5-1.5B-Instruct |
| Line | ar2block, load_pretrained=true |
| Data | Nemotron Post-Training SFT (sft_qwen); mix caps vary by cell |
| Seq / block | 2048 / 32 |
| Steps / GBS | 6000 / 256 (paper cells) |
| Train nodes | typically 8×4 after scale-up |
| Eval (masked accuracy) | hubmatch + thr=1 + greedy + 2048 new tokens |

Again: Fast-dLLM paper does not give mix ratios. Footnote mixes in Tab 1 or someone will compare chat C2 to math C0 and I will deserve the review comment.


Invalid / do not cite as main

| Class | Why |
|-------|-----|
| Pre PLAIN-CE-DENOM C2 | denom wrong, loss ~half Hub scale, C2 looked unfairly weak |
| Pre SFT-ATTN-PROMPT | prompt invisible in SDPA |
| Polarity-flip complementary | wrong paired masks |
| DualCache-era GSM ~2% / mid-word ! | window-local shift bug |
| Bare-BOS / open conversion_free soup | not the Instruct meter |
| Free-gen GenPPL ~402 | collapsed / prefix-EOS cells |
| U0 IFE 10.7 under FORCE_GREEDY=1 | asterisk or delete until ancestral lands |


Cell by cell (conversion)

C0 / neutral baseline 1762534: done. GSM 62.1, IFE 20.5, MMLU 39.8. Trainable, conditional English exists. Vs Hub we already match GSM-ish and lose hard on IFE/MMLU. Tab 1 neutral row.

C2 / Fast-dLLM positive control and ablations (masked):

| Cell | Status | MMLU | GSM8K | IFEval | GenPPL |
|------|--------|------|-------|--------|--------|
| C0_shift 1856101 | Done | 33.7 | 64.0 | 21.1 | 88 |
| C0_comp 1856103 | Done | 37.3 | 53.3 | 18.1 | 108 |
| C0_shift+comp 1856105 | Done | 32.0 | 50.8 | 17.0 | 107 |
| C0_anneal 1857473 | Done | 42.7 | 61.1 | 21.4 | 134 |
| C2_fdllm_full math 1763301 | Done | 30.6 | 66.8 | 22.4 | 89 |
| C2_fdllm_full chat 1773298 | Done | 40.3 | 43.4 | 23.8 | - |
| Earlier C2eb 1735919 in Hub gen | Done | - | 38.3 | 19.0 | - |

Shift alone helps GSM and hygiene. Complementary alone hurts. Shift+comp worse than either alone here. Anneal helps MMLU. Full pack + math is best GSM. Same pack + chat is a different world. Hub generator on earlier C2eb roughly matches our sampler.

Hypothesis C2a on GSM: supported for math mix. "Still soup under hubmatch chat-conditional GSM": rejected for that setting. Narrative I want: recipe mostly closed, distribution still open.

C3 matched AR SFT: missing for Tab 1. Wiring exists (ar_sft.sbatch / ar_sft_qwen.yaml). Leave blank. Do not invent numbers.

C4 decode / NFE: partial. hubmatch vs dual_cache vs baseline, thr probes exist. Full NFE grid {8,16,32,64} not assembled as a figure. Methods: hubmatch is accuracy protocol; tok/s is systems and not Fast-dLLM fused kernels.

C5 joint AR: 1836723 and 1836725 scored (see snapshot). Uniform C5 still thinner.


Extras / native

| Cell | Status | Draft |
|------|--------|-------|
| B1 fair floor | U0 1849335 scoring | not old chat-only 1836729 |
| B4 hybrid + anneals | 1857475 | GenPPL hygiene ~106 on disk; lm-eval queued |
| N0 / native masked | 1836795 / 1836796 | scratch floor is weak |
| B3 / blockgen_owt_uniform | early / separate scale | native track; do not dump into conversion Tab 1 |
| xfer_* | running / optional | different claim |


Shift as x0 interface (Unifusion vs Fast-dLLM)

Same index idea: logits at i supervise clean token i+1. Our algo.shift_loss_targets does that on the loss grid and decode can align_shift_logits. Same inductive bias, not a bit-identical port.

| | Unifusion | Our shift |
|--|-----------|-----------|
| Geometry | full-seq bi | block-causal |
| Where applied | init + train objective | train (decode align) |
| Masked | N/A (they go uniform) | Fast-dLLM mask-site CE |
| Uniform | yes (main) | now allowed (U0_shift) |

| Cell | = | Status |
|------|---|--------|
| C0 | hooks off, masked | Done 1762534 |
| C0_shift / C2_shift | C0 + shift_loss_targets | Scored 1856101 (GSM 64.0) |
| U0 | hooks off, uniform | IFE 10.7* greedy invalid; GSM queued |
| U0_shift | U0 + same shift | Queued 1857278 |

Table I want eventually: C0 vs C0_shift vs U0 vs U0_shift (GSM / IFEval / MMLU gen).


Axis A follow-ups I already ran because Hub gap would not die

| Probe | Status | Role |
|-------|--------|------|
| Chat↔math Pareto mixes | Done | mix tradeoff figure |
| Linear merge a=0.3/0.5/0.7 | Done 1836666-68 | last cheap combine bet; did not fix three-way |
| Continue-FT chat→math | Done 1836800 | sequential mix; also no Hub IFE miracle |


Editable tokens / Claim K (do not put in abstract)

I am not discovering editable tokens. Prior art exists (mask locks, uniform revises, remask/PC papers, GIDD-style stories). What is open is whether that mechanism explains our local Instruct fails under matched C0 vs U0.

| ID | Claim | Primary contrast |
|----|-------|------------------|
| K (kernel bridge) | Under matched AR→block convert, known revisability difference explains preferential help on local IFEval/code failures | C0 1762534 vs U0 1849335 only |
| R (revision ops) | Remask / absorb-ARPC / uniform-ARPC can also fix local errors when taxonomy says local | ancestral vs ARPC within an arm; secondary |

xfer / 1855537-41 is geometry-confounded. Not clean K evidence.

Phase 0 pins:

| Item | Pin |
|------|-----|
| Models (K) | C0 1762534, U0 1849335 (same math mix) |
| Suites | failed IFEval; optional failed HumanEval |
| Rubric | L1 local span chaos; L2 local lexical/syntax; G1 global constraint; G2 document format; mixed = primary + flag |
| Go threshold | local share (L1∪L2) / failures >= 30% on C0; do not move goalposts after labeling |
| N | pilot N=50 fails/model OK with bootstrap CI; prefer N>=100 |
| Causal language | scores + taxonomy = consistent with; causal only after Probe B |

| Decision | Rule |
|----------|------|
| Deny K for IFEval | C0 local share << 30% and U0 does not preferentially help the local slice |
| Keep K alive | local share >= 30% or U0 gains concentrate on L1/L2 |
| Axis A paragraph | global-dominant → editability still does not explain Hub IFEval |

| Probe | Measure | Role |
|-------|---------|------|
| A. Revision counts | mask ≈0 changes after commit; uniform redraw >0 | sanity only; never enough for K |
| B. Forced inject | mid-block flip one token; site recovery + local-span integrity | needed before causal K wording |
| C. ARPC/remask | ancestral vs absorb-ARPC (mask) | claim R only |
| D. NFE | local-error rate vs steps | supporting |

| Still true | Consequence |
|------------|-------------|
| Free-gen can be soup | conversion_free U0/xfer/U2: cite (PPL, H) as hygiene only, never fluency |
| xfer ≠ U0 recipe | xfer has BlockGen train geometry; do not rank as matched uniform |

U2 size-1 PPL ~459 was a meter bug (SIZE1-EVAL). Re-sweep ~9.9 after fix. See DESIGN_LOCKS.

This week if I am honest: label 50 C0 + 50 U0 IFE fails. If local share low, short denial of K for IFEval and stop selling the story. Sheets under docs/research/editable_tokens/.


Two-stage uniform (gated)

Unifusion found direct AR→uniform roughly equals AR→mask→uniform on their GenPPL setup. We still planned a gated try because our geometry differs.

| Cell | Init | Train | When |
|------|------|-------|------|
| U0 (canonical) | AR Instruct | uniform, hooks off, math mix | Done / scoring 1849335 |
| U0_from_C0 | resume C0 1762534 | continue-FT uniform, same mix | only if U0 paper_gen clearly worse than C0 |
| U0_from_C2 (optional) | resume C2 math 1763301 | same | only if U0_from_C0 helps |

Hypothesis: warm-start might ease optimization but should not beat matched-budget direct U0 on conditional metrics. If it does not help, drop two-stage and cite Unifusion + our null. Priority behind C3 and Tab 2.


Traffic light (claims)

| Claim | Light | Notes |
|-------|-------|-------|
| Conversion vs native design-space map | Green | families + registry |
| Recipe moves conditional gen | Green (GSM) | Tab 2 masked filled; U0 levers pending |
| Matched AR SFT vs block conversion | Red | need C3 |
| Val BPD mis-ranks conversion | Yellow | qualitative |
| Longitudinal capability recovery | Red | §7 not filled |
| Native BlockGen geometry moves quality | Red | pending / early |
| B1 masked vs uniform | Yellow | matched = U0 vs C0; chat-only B1 appendix |
| Two-stage AR→mask→uniform helps vs direct U0 | Gray | gated on weak U0 |
| Hub leaderboard parity | Red | never without external reproduction |

Lead finding if I have to pick one sentence: early recipe was incomplete because of wiring bugs, then recipe closed for GSM; remaining Hub gap looks like data/mix (A), not missing D levers.


Longitudinal §7

| Milestone schedule | Status |
|--------------------|--------|
| init→500→1k→2k→4k→6k bundles | Not assembled for canonical C0/C2 |
| Draft | Methods can mention intent; Results omit Fig 3 until it exists |


Figures / tables I can draft now

| ID | Content | Draft readiness |
|----|---------|-----------------|
| Fig 1 | Conversion vs native vs transfer map | Ready |
| Fig 2 | C0 train/val curves | Ready if I pull 1762534 logs |
| Fig 3 | Longitudinal recovery | Blocked |
| Fig 4 | NFE sweep | Optional / partial |
| Fig 5 | BPD vs GSM disagreement | Optional |
| Fig Mix | Chat↔math Pareto (MMLU vs GSM) | Ready (1773298 vs 1763301) |
| Tab 1 | C0, C2*, C3 | Partial (no C3) |
| Tab 2 | Lever ablation (masked) | Ready; GenPPL = hygiene only; U0 levers queued |
| Tab Hub | Hub vs our best rows | Ready (snapshot above) |
| Tab N1 / X1 / B1 | Native / transfer / noise | Blocked or queued |


Draft Tab 1 (partial)

| System | Line | Recipe | Mix | MMLU | GSM8K | IFEval |
|--------|------|--------|-----|------|-------|--------|
| Hub Fast-dLLM v2 1.5B | - | Hub | undisclosed subset | 53.5 | 62.8 | 44.4 |
| C0 | ar2block | hooks off | post-gate Nemotron | 39.8 | 62.1 | 20.5 |
| C2_fdllm_full | ar2block | shift+comp+fast_dllm+plain_ce | math1M/code500k | 30.6 | 66.8 | 22.4 |
| C2_fdllm_full | ar2block | same | chat/safety/science | 40.3 | 43.4 | 23.8 |
| C3 AR SFT | causal | - | matched | - | - | - |


Draft Tab 2 (masked AR→block; GenPPL = hygiene only)

| Cell | Components | MMLU | GSM8K | IFEval | GenPPL |
|------|------------|-----:|------:|-------:|-------:|
| C0 1762534 | off | 39.8† | 62.1 | 20.5 | 128 |
| C0_shift 1856101 | shift | 33.7 | 64.0 | 21.1 | 88 |
| C0_comp 1856103 | complementary | 37.3 | 53.3 | 18.1 | 108 |
| C0_shift+comp 1856105 | shift+comp | 32.0 | 50.8 | 17.0 | 107 |
| C0_anneal 1857473 | intra-block anneal | 42.7 | 61.1 | 21.4 | 134 |
| C2 full math 1763301 | full pack | 30.6† | 66.8 | 22.4 | 89 |

† hubll MMLU. Do not cite free-gen GenPPL ~402 on collapsed/prefix-EOS cells.


Implementation

Unchanged from plan: configs/paper/cells.yaml, submit_paper_cell.sh, submit_lever.sh, family firewall in configs/levers/registry.yaml. See PAPER_EXPERIMENTS.md section 9.


Priority vs reality

| Priority | Plan | Reality |
|----------|------|---------|
| P0 C0 | Done | Canonical 1762534 |
| P1 C2 | Done (full recipe) | 1763301 / 1773298; Tab 2 masked scored |
| P2 C3 | In flight | Train 1857323 (after 1856100 fail / 1857271 cancel) |
| P3 C4 | Partial | hubmatch locked; full NFE optional |
| P4 B1 / U0 | U0 scoring | Fair floor 1849335; ancestral IFE after greedy fix |
| P4b U0_from_C0 | Gated try | only if U0 weak |
| P4c U0_shift | In flight | 1857278 |
| P4d GenPPL hygiene | In flight / landed | see ledger |
| P5 N0/B2 | Queued / scored | 1836795/96 |
| P6 BlockGen | Early / separate | do not block conversion write-up |
| P7 B4/C5/xfer | Partly scored | C5 done; B4/xfer landing |
| P7b Tab 2 | Masked done | 1856101/03/05 + anneal 1857473; U0 levers queued |
| P8 Write-up | Start now | conversion spine + Pareto with this file |


Limitations I should not forget in the draft

Instruct SFT ~3B tokens. Hub subset ratios unknown so axis A ceiling is unclear.
Main conversion claims use masked.
IFEval / MMLU remain well below Hub under all mixes so far.
Our tok/s is not Fast-dLLM fused kernels (DESIGN_LOCKS).
Single backbone Qwen2.5-1.5B-Instruct.
Pending queue jobs may revise rows. Date-stamp any PDF.


Abstract-ish blurb if I need something short

We study AR→block conversion of a pretrained LM as a design space separate from native scratch block diffusion. Same codebase, hard family firewall. Fast-dLLM-style recipe (axis D) is needed for a fair comparison and improves GSM8K over a hooks-off baseline once training bugs are removed. GSM can reach Hub 1.5B levels (~62 to 67) but IFEval and MMLU do not, and chat vs math Nemotron mixes trade off on a Pareto front. Exporting our weights into the Hub generator matches our sampler, so the residual gap looks like data/mix (axis A) rather than decode porting. Matched AR SFT (C3), more uniform scores, and native BlockGen tables are still ongoing.


Where stuff lives

| Role | Path / job |
|------|------------|
| Plan (immutable design) | docs/research/PAPER_EXPERIMENTS.md |
| This populated copy | docs/research/PAPER_EXPERIMENTS_POPULATED.md |
| Live ledger | experiments/2026-09-07_lm_eval_nccl_timeout/EXPERIMENT.md |
| C0 hubmatch SUMMARY | outputs/block_qwen/ar2block_masked_1762534/lm_eval_m2048_hubmatch_t1/SUMMARY.json |
| C2 math SUMMARY | outputs/block_qwen/ar2block_masked_1763301/lm_eval_m2048_hubmatch_t1/SUMMARY.json |
| C2 chat SUMMARY | outputs/block_qwen/ar2block_masked_1773298/lm_eval_m2048_hubmatch_t1_chatpreserve/SUMMARY.json |
| Hub MMLU recheck | outputs/hub_fastdllm_eval/Fast_dLLM_v2_1.5B_t1_mmlu_recheck/SUMMARY.json |
| Our weights in Hub gen | outputs/hub_fastdllm_eval/our_ar2block_masked_1735919_t1/SUMMARY.json |

Updated 2026-09-18. Refresh the U0/U2/xfer lines when paper_gen SUMMARYs actually land.
