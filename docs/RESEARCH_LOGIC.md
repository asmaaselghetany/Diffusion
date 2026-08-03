# How we got here: baseline logic and the three projects

This is the long version of *why* we built things the way we did. Not a command cheat sheet (see `BLOCK_QWEN_TRAINING.md` for that). More like: what problem each piece solves, what we considered, what we rejected, and what still depends on a GPU finishing.

---

## The question underneath everything

Block diffusion language models are getting serious: Fast-dLLM on Qwen, BlockGen on uniform corruption, papers reporting good ELBO and decent samples. But most of that work optimizes **average quality** on web text. It does not really ask:

- When you **fill in a hole** with locked context, does the model respect a dependency that spans a block cut?
- When you run a **multi-turn agent**, can prior tool calls stay frozen while you diffuse the next reply?
- Is a **fixed 32-token block** even the right unit, or are we slicing syntax in the wrong places?

Our stack is organized around those three questions. The baseline (`uni-d2` / `block_qwen`) is the shared engine. DepBench measures dependency failures. Commitment Agent tries to fix agent trajectories. GrowBlock asks whether the unit of diffusion should be learned at all.

---

## Repo map (one sentence each)

| Piece | What it is |
|-------|------------|
| **uni-d2** | Train Qwen block diffusion (masked or uniform), evaluate, checkpoint |
| **DepBench** | Controlled infill probes + violation metrics |
| **Commitment Agent** | Segment-aligned diffusion for tool/agent transcripts |
| **GrowBlock** | Kill experiment on segmentation; joint structure diffusion if warranted |

They share one venv and one `BlockTrainer` family. DepBench plugs into eval scripts. Commitment extends the trainer. GrowBlock reuses DepBench and segment machinery.

---

## Part 1: The baseline (`block_qwen`)

### Why we built our own path instead of cloning a paper repo wholesale

Fast-dLLM and BlockGen ship real systems, but they are not drop-in comparable:

- Fast-dLLM is masked-first, SFT-heavy, with training tricks (complementary masks, shifted labels, partial masking) that are **specific to their recipe**.
- BlockGen is uniform-first, with block-size mixtures and ARPC sampling that are **specific to their recipe**.
- Both papers report headline numbers on different data scales and hardware.

We wanted one codebase where **only the corruption type changes** between arms. Same Qwen init, same seq length, same optimizer, same steps, same sampler defaults. That is the neutral `block_qwen` experiment.

We also did **not** adopt BD3LM or BlockDiT for this line of work. Those are strong for research on alternate backbones and mask schedules, but our question is Qwen + block semi-AR diffusion in the Fast-dLLM / BlockGen style. Adding another architecture family would blur what we are comparing.

### What we actually took from each paper

**From Fast-dLLM (masked arm):**

- Block-wise causal attention on `concat(xt, x0)` (noised sequence beside clean sequence).
- Per-block absorbing corruption: one noise time `t` per block, tokens mask with probability `1 - alpha(t)`.
- SUBS parameterization for the reverse process (masked positions get log-probs; unmasked copied through).
- Semi-autoregressive decode: finish block *k*, cache it, diffuse inside block *k+1*.

**From BlockGen (uniform arm):**

- Same block outer loop, but corruption draws uniform random tokens over the vocabulary within each block.
- Uniform-state ELBO (full vocab statistics in the loss; more memory hungry at eval time).
- Optional extras we keep **off** for neutrality: block-size mixture, stratified gamma across GPUs, ARPC at sample time.

**From MDLM / UNI-D² infrastructure:**

- Hydra configs, Lightning training loop, log-linear noise, data loaders, checkpoint callbacks.
- The general discrete diffusion habit: one `TrainerBase`, switch forward process via config.

### The single trainer design (`BlockTrainer`)

One Lightning module, two forward processes. Switch with `algo=block_masked` or `algo=block_uniform`.

Why one trainer instead of two files?

- Every shared bug fix (Qwen mask injection, passthrough `_process_model_input`, init metrics) lands once.
- Neutral comparison stays honest: you are not maintaining divergent training loops.
- Downstream projects (Commitment) subclass `BlockTrainer` instead of forking masked code.

The actual difference inside `nll()`:

- **Masked:** gather log-probs at `x0` indices; cheap at validation batch 8.
- **Uniform:** materialize vocab-sized intermediate tensors; blows up GPU memory at eval batch 8 on 1.5B + seq 2048 (we learned this the hard way on jobs 126237/126240).

Training uses `batch_size: 1` with `global_batch_size: 128` (gradient accumulation). Validation uses `eval_global_batch_size: 1` on **both** masked and uniform for a fair paired compare (uniform OOMs at 8 on 1.5B + 2048).

### Neutral baseline: what is off on purpose

These config keys exist because the papers use them. We default them **false / empty / null** on **both** arms:

| Hook | Paper | Why off for neutral compare |
|------|-------|----------------------------|
| `shift_loss_targets` | Fast-dLLM | Masked-only training trick; not part of corruption type |
| `complementary_masks` | Fast-dLLM | Same |
| `block_size_mixture` | BlockGen | Changes training distribution, not just corruption |
| `stratified_gamma` | BlockGen | Multi-GPU variance trick |
| `use_arpc` | BlockGen | Decode-only; uniform arm advantage in their results |

You can turn them on via `HYDRA_OVERRIDES` at launch for a "replicate paper recipe" run. Do **not** put them as dot-keys in `block_qwen.yaml`; Hydra will park them at the config root and `BlockTrainer` will never see them.

### Choices we made for the current 1.5B SFT run

| Choice | Value | Reasoning |
|--------|-------|-----------|
| Base model | Qwen2.5-1.5B-Instruct | Matches Fast-dLLM scale; AR init matters for block SFT |
| Data | Nemotron post-training SFT (`sft_qwen`; default chat+safety+science) | Alpaca kept as `sft_qwen_alpaca` ablation only |
| Seq length | 2048 | Paper recipe |
| Block size | 32 | Paper recipe; matches semi-AR chunk in DepBench default sweeps |
| Steps | 7500 @ effective batch 128 | ~2B tokens on one GPU; paper uses 64 GPUs we do not have |
| Attention | SDPA + gradient checkpointing | Fits 1.5B + 4096 effective length (concat doubles seq) on 80GB |
| Val every | 500 steps | Faster feedback; eval batch 1 on both arms |

---

## Part 2: DepBench

### What paper metrics miss

Val ELBO and gen-PPL average over tokens. A model can look fine on average while systematically failing probes like "the verb must agree with the subject eight tokens back, and the hole sits across a block boundary."

DepBench is **infill-only**: context locked, hole to fill, ground truth known. Score = violated if not (exact match on hole **and** family constraint passes).

### Why `crosses_block_boundary` matters

Semi-AR block diffusion commits to earlier blocks before denoising later ones. Masked diffusion **cannot revise** a token once it is unmasked inside a block (BlockGen makes this explicit). So we tag each probe with whether the dependency edge crosses a block cut.

The cheap extra analysis people suggested: run DepBench on **masked and uniform** checkpoints from the same neutral training, split violation rate by that flag. If masked is worse specifically at boundaries, DepBench gets a **mechanism story**, not just a number. No new training, just eval slicing.

### Probe families (why these and not one task)

| Family | What it stresses |
|--------|------------------|
| arithmetic | Long carry chains (compositional, exact) |
| agreement | Syntax at distance |
| coreference | Antecedent must be preserved |
| csp | Small logic templates |
| naturalistic_* | Real-ish text with annotated deps |

Synthetic probes give clean metadata (distance, degree, cross-block). Naturalistic ones stress whether synthetic conclusions transfer.

### Design choice: infill backend = production sampler

We score with the same `BlockSampler` / Lightning checkpoint path you train with. Not a separate greedy decode hack. If the model fails DepBench, it fails the way you would actually use it.

---

## Part 3: Commitment Agent

### The agent problem in one paragraph

Tool agents are transcripts: user, assistant, tool call, tool result, assistant again. AR models are built for this: past tokens are past. Full-sequence bidirectional diffusion is attractive for parallel decoding but scary here: if you corrupt or attend backward into a tool call that already executed, you are not doing inference anymore, you are rewriting history.

### What Commitment Agent adds on top of the baseline

1. **Segments, not fixed token blocks.** Turns / tool regions get `segment_ids`. Only the `active_segment` is corrupted during training.
2. **Frozen prefix.** Segments before active stay clean in `xt`; loss does not apply on frozen tokens.
3. **Segment attention mask.** Same block-diffusion idea as Fast-dLLM, but boundaries follow semantics, not every 32 tokens.
4. **Commitment KL.** On committed tool-call tokens: teacher sees true future context; student sees permuted hypothetical next segment. KL penalizes instability of actions you already took.

Loss shape:

```
ELBO(active segment) + λ · KL_committed(teacher || student_hypothetical)
```

### Baselines inside the project (mechanism isolation)

We are not claiming one loss fixes everything. We wired comparisons:

| Trainer | Isolates |
|---------|----------|
| VanillaAgentTrainer | Full bidirectional diffusion on whole transcript |
| DiffuAgentTrainer | Format repair emphasis (JSON-ish) without commitment |
| CommitmentTrainer | Segment mask + commitment KL |
| AR reference | What Qwen-Instruct already does |

The ablation people care about: **segment mask alone vs +KL**. That tells you whether freezing segments is enough or the KL term is doing real work.

### Eval stack

BFCL for tool correctness, loop rate for repeated calls (a failure mode we have seen in diffusion agents), plus hooks toward broader agent boards. Training is code-complete; paper numbers need a real checkpoint run.

---

## Part 4: GrowBlock

### The hypothesis

Fixed blocks are a engineering convenience, not a linguistic fact. Clauses and sentences have variable length. If **oracle** boundaries (sentence ends, clause punctuation) dramatically beat fixed 32-token cuts on DepBench, especially on cross-segment dependencies, then learning boundaries jointly with tokens might be worth the complexity. If oracle barely helps, a joint structure model is probably theater.

### Phase 1: kill experiment (done in code, needs results)

Same checkpoint, same DepBench probes, **only segmentation changes**:

| Strategy | Meaning |
|----------|---------|
| `fixed` | Standard B-token blocks |
| `random` | Same segment sizes, shuffled cut positions (control for length) |
| `oracle_sentence` | Cuts at sentence ends |
| `oracle_clause` | Cuts at clause punctuation |

Verdict logic (legacy kill harness; Phase 2 novelty is scooped by DCDM):

- **STOP:** oracle ≈ fixed → ignore oracle cuts for bakeoff
- **CONDITIONAL_GO / GO:** oracle wins → reuse as a bakeoff column only; do not start learned-boundary training

### Phase 2: joint structure diffusion (archived scaffolding)

`BoundaryHead` and `JointStructureTrainer` remain as code, but Phase 2 is **not** an active paper track (DCDM owns learned chunks). Keep for possible bakeoff reuse of oracle cuts after a harness-clean kill rerun.

---

## How the pieces depend on each other

```
uni-d2 (train masked | uniform, neutral)
    ↓ checkpoints
DepBench (where do dependencies break?)
    ↓ cross-block splits, kill experiment input
Commitment Agent (freeze + future-stability; ledger fidelity primary)
    ↓ segment masks, attention machinery
GrowBlock (archived; oracle cuts may appear as a bakeoff column)
```

DepBench does not need Commitment. Commitment does not need GrowBlock. But they share vocabulary: segments, infill, violation rate, `crosses_block_boundary`, and History turn strata.

---

## Practical stuff we ran into (so you do not re-debug)

1. **Hydra dot-keys** like `algo.shift_loss_targets: true` at experiment root do not merge into `config.algo`. Use CLI overrides or set keys inside `configs/algo/block_*.yaml`.
2. **Uniform validation OOM** on 1.5B / 2048 / eval batch 8. Fix: `loader.eval_global_batch_size=1` on **both** arms (now default in `block_qwen.yaml`).
3. **Do not stack two training jobs on one GPU** (masked + uniform on the same node). Contention causes random OOM during attention, not just validation.
4. **7500 steps may not fit one 24h SLURM slot** — use `./scripts/resume_block_qwen.sh outputs/block_qwen/<run_dir>` to continue from `last.ckpt` (a bare resubmit creates a new job id and new folder).

---

## What we are claiming vs what we still need

| Claim | Status |
|-------|--------|
| Neutral masked vs uniform training recipe is wired | Yes |
| Masked 1.5B SFT run can train on one 80GB GPU | In progress |
| Uniform 1.5B SFT at same recipe | In progress |
| DepBench runs on Lightning ckpts | Yes |
| Masked vs uniform DepBench cross-block comparison | Waiting on both ckpts |
| Commitment training at scale | Code ready, needs GPU budget |
| GrowBlock kill verdict | Code ready, needs checkpoint + eval |
| GrowBlock joint training | Not wired; gated on kill |

---

## References (the papers we argue with)

- **Fast-dLLM v2** ([2509.26328](https://arxiv.org/abs/2509.26328)): masked block SFT on Qwen, complementary masks, shifted labels, sub-block decode.
- **BlockGen** ([2606.02241](https://arxiv.org/abs/2606.02241)): masked or uniform within block; mixture training; ARPC sampler.
- **MDLM** ([2406.07524](https://arxiv.org/abs/2406.07524)): SUBS masked diffusion ELBO; full-sequence, not block semi-AR.
- **Fast-dLLM repo**: [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
- **BlockGen repo**: [jdeschena/blockgen](https://github.com/jdeschena/blockgen)

---

## If you read one section before a meeting

**Baseline:** One trainer, two corruptions, all paper extras off, so masked vs uniform is a fair fight.

**DepBench:** Infill probes with dependency metadata; cross-block flag is the mechanistic hook.

**Commitment:** Segments frozen behind you, KL on tool calls so history does not wobble when future text changes.

**GrowBlock:** Oracle segmentation test first; joint diffusion only if cuts matter.

That is the logic. The rest is config keys and SLURM job IDs.
