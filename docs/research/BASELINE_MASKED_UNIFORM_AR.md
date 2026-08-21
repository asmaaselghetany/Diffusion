# Baseline design: masked, uniform, and where AR comes from

This doc is only about the **training baseline**: why we compare masked vs uniform block diffusion, and whether to start from a pretrained AR model (Qwen) or train our own. For commands and hyperparameters see [BLOCK_QWEN_TRAINING.md](BLOCK_QWEN_TRAINING.md). For the wider DepBench / Commitment / GrowBlock program see [RESEARCH_LOGIC.md](RESEARCH_LOGIC.md).

---

## What the baseline is trying to answer

Block diffusion can corrupt tokens inside a semi-autoregressive block in (at least) two ways:

1. **Masked** (Fast-dLLM style): replace tokens with `[MASK]`; denoise by unmasking.
2. **Uniform** (BlockGen style): replace tokens with random vocabulary draws; denoise by revising.

Everything else (backbone family, data, steps, block size, sampler defaults) can be held fixed. Then a gap in ELBO, gen-PPL, or DepBench violation rate is attributable to **corruption type**, not to a pile of arm-specific training tricks.

That is the **neutral `block_qwen` baseline**:

| Arm | Launch | Only difference |
|-----|--------|-----------------|
| Masked | `algo=block_masked` | Per-block absorbing mask + SUBS ELBO |
| Uniform | `algo=block_uniform` | Per-block uniform corruption + uniform ELBO |

Shared: Qwen2.5-1.5B-Instruct init, Nemotron post-training SFT (`sft_qwen`; Alpaca only via `sft_qwen_alpaca`), seq 2048, block 32, same optimizer, same step budget, same noise schedule, same block sampler defaults. Optional paper hooks (`shift_loss_targets`, `complementary_masks`, `block_size_mixture`, `stratified_gamma`, `use_arpc`) stay **off** on both arms.

---

## Why masked and uniform in one codebase

**Masked** is the default story in Fast-dLLM and most block dLLM demos: parallel unmask inside a block, cache completed blocks, feel close to AR at the outer loop.

**Uniform** is BlockGen’s bet: tokens can be **revised** after they look decoded. That may matter for few-step sampling and for dependencies that cross block cuts (masked cannot un-commit a token once it is revealed).

We implement both in one `BlockTrainer` because:

- Fair comparison needs one training loop, one Qwen block mask path, one eval pipeline.
- DepBench can run on both checkpoints with the same probes and report `by_cross_block` splits.
- Downstream work (Commitment Agent) subclasses the same trainer instead of forking masked-only code.

We deliberately **do not** turn on Fast-dLLM-only or BlockGen-only extras in the shared experiment yaml. Those are documented as optional `HYDRA_OVERRIDES` for replication runs, not for the neutral compare.

---

## The second question: pretrained AR vs train our own

These are **not** the same decision. People say "train our own AR" and mean three different things.

### Option 1: Pretrained AR init (what we do today)

Start from **Qwen2.5-1.5B-Instruct** (or another public AR checkpoint). Run block SFT on top.

**Why papers do this**

- Fast-dLLM explicitly converts pretrained Qwen with ~1B SFT tokens instead of training block diffusion from scratch (they contrast with full-attention diffusion methods that need far more data).
- SDAR, Dream/DiffuLLaMA, REPR-ALIGN, BlockVLA: convert or align from an existing AR backbone because language representations are already there.
- [Mechanism Shift During Post-training AR to MDM](https://arxiv.org/html/2601.14758v3) studies what changes when you **post-train** a pretrained AR into diffusion, which assumes a strong AR starting point.

**Pros for us**

- Matches the field we cite (Fast-dLLM, BlockGen on top of capable LMs).
- Feasible on one 80GB GPU for 1.5B SFT-scale runs.
- Instruction-tuned init matters for anything agent-shaped later (Commitment, BFCL).

**Cons**

- You inherit whatever Qwen already learned; masked and uniform both share that init, but neither is "language from scratch."
- An external reviewer may ask for an AR baseline that received the **same** post-init training budget.

**Our current default:** Option 1 for `block_qwen` masked and uniform.

---

### Option 2: Matched AR SFT baseline (recommended add-on)

Same pretrained **starting** checkpoint as block arms, but train with **standard causal next-token SFT** on the same data, steps, and batch schedule. No `concat(xt, x0)`, no block corruption.

| | Masked / uniform | Matched AR SFT |
|--|------------------|----------------|
| Init | Qwen2.5-1.5B-Instruct | Same |
| Data | Nemotron (`sft_qwen`) | Same |
| Steps / LR / batch | Same | Same |
| Objective | Block diffusion ELBO | Causal LM loss |
| Decode | Block sampler | Standard AR generate |

This does **not** require training language from scratch. It answers: "given the same fine-tuning budget after download, how does plain AR compare to block masked vs block uniform?"

**Supporting research**

- [Autoregressive vs. Masked Diffusion: A Controlled Comparison](https://arxiv.org/abs/2603.22075) (same data, same compute, paradigm is the variable; they train **both** from scratch at small scale, but the *logic* is identical: matched AR reference).
- [Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/pdf/2507.15857) (paired scaling comparison; who wins depends on data repetition vs compute).

**Status in repo:** not wired as a third `block_qwen` arm yet; natural extension is `algo=ar_sft` or similar sharing `block_qwen.yaml`.

**Best for:** DepBench three-way table (masked / uniform / AR), gen-PPL, val metrics on the 1.5B Nemotron-SFT line.

---

### Option 3: Scratch AR and scratch block (small-scale paradigm study)

Train **both** AR and block diffusion from **random init** on a small corpus (e.g. TinyStories), matched tokens and steps. Typically smaller than 1.5B.

**Why anyone does this**

- Removes the confound of trillion-token pretraining.
- Cleanest story for "masked vs uniform vs AR as paradigms."
- [2603.22075](https://arxiv.org/abs/2603.22075) is the template: identical 50M tokens, 20k steps, release both checkpoints.

**Pros**

- Strong controlled-comparison citation.
- Cheap enough to actually finish on limited GPU budget.

**Cons**

- Not the same scale as the 1.5B Alpaca `block_qwen` story.
- Numbers do not transfer directly to instruction or DepBench at 2048 without re-running at that scale (expensive).

---

### Option 4: Full scratch pretrain at 1.5B+ (usually skip)

Random init, web-scale data, Chinchilla-optimal compute for AR **and** block diffusion.

**Supporting research:** SDAR pretrains paired 2B AR and 2B MDLM on **1T tokens** before SFT. That is the gold standard for fairness at scale.

**Why we skip it:** cluster-month budget, not a side experiment on one GPU. Fast-dLLM and BlockGen papers at 1.5B also assume **pretrained** Qwen, not scratch pretrain.

---

## Decision matrix (what to run when)

| Goal | Pretrained Qwen block SFT | + Matched AR SFT | + Scratch small-scale pair |
|------|-------------------------|------------------|----------------------------|
| Reproduce Fast-dLLM / BlockGen practice | Yes | Optional | No |
| Neutral masked vs uniform | Yes (in progress) | No | No |
| DepBench with AR reference column | Yes | **Yes** | Optional supplement |
| "Paradigm only" paper figure | No | No | **Yes** (TinyStories-scale) |
| Commitment / BFCL agents | Yes (Instruct init) | AR SFT on agent data | No |
| Claim "we trained our own LM" at 1.5B | No | Partial (SFT only) | No |

---

## Recommended stack for this project

**Now (primary line)**

1. **Masked + uniform** from **pretrained Qwen**, neutral hooks off (`block_qwen`). This is the main scientific compare.
2. Fix uniform validation memory (`eval_global_batch_size: 1` on both arms for symmetry) — done in `block_qwen.yaml`.

**Next (cheap, high citation value)**

3. **Matched AR SFT** arm: same init, same `block_qwen` recipe, causal LM only. Use for DepBench, gen-PPL, and any table that needs "AR with the same fine-tuning budget."

**Later (optional figure)**

4. **Scratch pair** on TinyStories (or similar) if you want a second "controlled comparison" figure independent of Qwen pretrain. Cite [2603.22075](https://arxiv.org/abs/2603.22075).

**Not planned unless budget changes**

5. Full scratch 1.5B pretrain for AR and block.

---

## How this connects to eval

| Metric | Masked | Uniform | Matched AR (planned) | Raw Qwen (no our SFT) |
|--------|--------|---------|----------------------|------------------------|
| Val ELBO / bpd | Block ELBO | Block ELBO (heavier) | Causal NLL | N/A |
| Gen-PPL | Block sample | Block sample | AR sample | Optional ceiling |
| DepBench | Infill + violations | Infill + violations | AR infill or generate | Weak baseline |
| Cross-block split | `by_cross_block` | `by_cross_block` | Same | Same |

Comparing to **untuned Qwen** alone is weak: it did not see our Alpaca steps. Comparing to **matched AR SFT** is the fair "same post-init budget" reference. Comparing to **scratch AR** is a different paper paragraph (paradigm at small scale).

---

## Literature quick reference

| Work | AR init | Comparison style |
|------|---------|------------------|
| [Fast-dLLM v2](https://arxiv.org/abs/2509.26328) | Pretrained Qwen SFT | Block masked; AR as decode baseline |
| [BlockGen](https://arxiv.org/abs/2606.02241) | Pretrained + block training | Masked vs uniform within block |
| [MDLM](https://arxiv.org/abs/2406.07524) | Often scratch or BERT-scale | Full-sequence masked diffusion |
| [AR vs MDLM controlled](https://arxiv.org/abs/2603.22075) | **Both scratch**, matched compute | Paradigm isolation |
| [Diffusion vs AR data-constrained](https://arxiv.org/pdf/2507.15857) | Matched scaling setup | When diffusion wins on repeated data |
| [SDAR](https://arxiv.org/html/2510.06303v3) | **Both scratch** 2B pretrain, then SFT | AR vs block diffusion at scale |
| [Mechanism shift AR→MDM](https://arxiv.org/html/2601.14758v3) | Paired pretrained AR vs converted MDM | Post-training, not scratch |
| [REPR-ALIGN](https://arxiv.org/html/2605.06885) | Pretrained AR frozen as teacher | AR representations transfer to DLM |

---

## One-paragraph summary

The baseline is **masked vs uniform block diffusion** with everything else shared and paper-specific hooks turned off. We start from **pretrained Qwen** because that matches Fast-dLLM and BlockGen practice and fits our hardware. Training "our own AR" should mean **matched AR SFT on the same data and steps** for eval fairness, or a **small scratch AR+block pair** for a clean paradigm study, not full 1.5B pretraining from random init unless the compute budget changes.
