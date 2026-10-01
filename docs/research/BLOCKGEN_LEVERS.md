# BlockGen recipe notes (native family only)

Clone: `third_party/blockgen` ([jdeschena/blockgen](https://github.com/jdeschena/blockgen), arXiv:2606.02241).

**Family:** BlockGen lives on **`LINE=block` (scratch)**. It is **not**
AR→block conversion. Geometry on conversion uses `xfer_*` presets and must not
be tagged BlockGen. Legacy `submit_blockgen_lever_micro.sh` is **retired** (it
hardcoded `ar2block_*.sbatch`).

## What BlockGen does (absorb + uniform)

| Piece | BlockGen | Our wiring |
|-------|----------|------------|
| Init | Scratch Block-DiT | `line=block` (`load_pretrained=false`) |
| Corruption | Absorb **or** uniform within block | `algo=block_masked` / `block_uniform` |
| Default `loss_type` | **`elbo`** | `loss_type: elbo` — already on |
| Special cases | CE at block size 1 | lever `ce_at_1` |
| Block sizes | `block_weights` over powers of 2 | `bg_weights_*` + optional list mixture |
| Stratified size draw | `u-stratified` | lever `u_stratified` |
| Pure noise @1 | yes | lever `pure_noise_1` |
| Decode | Ancestral + **ARPC** (needs size 1 in mix) | `arpc_blockgen` on **masked or uniform** |
| Generate packing | `[clean_prefix \| noisy_block]` | Conversion floor: `block_gen_logits` under `hierarchical` (both arms) |

**ARPC absorb:** propose clean block → score with \(L'=1\) AR NLL → remask
lowest-LL tokens with `[MASK]`. Pins: `arpc_mode=blockgen`,
`arpc_corruption_mode=ar_metric`, `arpc_ar_metric=nll` (matches
`third_party/blockgen/scripts/sample/*/blockgen_absorb_ar_then_arpc*.sh`).
Uniform uses the same scorer; re-noise is Unif(V) instead of MASK.

**Conversion decode bake-off (not native BlockGen claim):**

| Cell | Profile | Meaning |
|------|---------|---------|
| B1 | `hierarchical` | BlockGen packing + conf remask thr=0.9 (contract floor) |
| B2 | `hierarchical_arpc` | Same packing + ARPC at T=1 (decode probe OK without size-1 train) |
| Quiet lever | `hierarchical_quiet` | ss + T≈0.1 + ARPC (TinyGSM-ish; not B2) |

Conversion transfer train geometry: `xfer_bg_mix_32` (Fast-dLLM soft defaults)
/ `xfer_bg_mix_32_blockgen` (hard pure-noise + `x0_causal`) /
`xfer_bg_mix_32_arpc` / `xfer_bg_mix_32_arpc_blockgen` /
`xfer_bg_mix_32_blockgen_ss` (+ `_ss_shift`) / `xfer_bg_mix_32_blockgen_unifv`
(`uniform_simplex_mode=blockgen`).
Map: `FAIR_AR2BLOCK_MAP.md` · `BASELINE.md` · `AR2BLOCK_GSM_TABLE.md`.

### Side-by-side train options (bake later)

| Knob | Fast-dLLM default | BlockGen opt-in lever |
|------|-------------------|----------------------|
| Size-1 noise | `pure_noise_1` → `t=1` Bernoulli (`pure_noise_mode=soft`) | + `pure_noise_hard` → hard MASK/Unif fill |
| Clean-stream attn | block-causal x0→x0 | `x0_causal` → token-causal |
| Train packing | dual concat(xt,x0) | `single_stream_train` + `single_stream_decode` |
| Simplex | `Unif(V\E)` conversion | `unif_v_blockgen` → literal `Unif(V)` |

Queued `xfer_bg_mix_32` stays soft. To compare:
```bash
./scripts/submit_lever.sh --preset xfer_bg_mix_32 --arm uniform --paper
./scripts/submit_lever.sh --preset xfer_bg_mix_32_blockgen --arm uniform --paper
./scripts/submit_lever.sh --preset xfer_bg_mix_32_blockgen_ss --arm uniform --paper
./scripts/submit_lever.sh --preset xfer_bg_mix_32_blockgen_ss_shift --arm uniform --paper
./scripts/submit_lever.sh --preset xfer_bg_mix_32_blockgen_unifv --arm uniform --paper
```

### Name collision (do not conflate)

| Name | Actual axis |
|------|-------------|
| BlockGen “stratified” | **block-size** draw (`u-stratified`) |
| `algo.stratified_gamma` | **timestep** strat within a fixed block (ours; `B3_t_strat` on conversion) |

## Launch (registry)

```bash
# Native micros / paper cells
./scripts/submit_lever.sh --preset N0 --arm uniform --paper
./scripts/submit_lever.sh --preset B3_mixture --arm masked --micro
./scripts/submit_lever.sh --preset B3_arpc --arm uniform --paper
./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform --paper

# One-to-one OWT 1+16
PREFETCH_ONLY=1 ./scripts/submit_blockgen_owt.sh
./scripts/submit_blockgen_owt.sh

# Transfer only (NOT a BlockGen claim)
./scripts/submit_lever.sh --preset xfer_arpc --arm uniform --paper
```

Registry firewall: raw native levers on `--line ar2block` raise unless the
preset is `xfer_*`. Fast-dLLM levers refuse `--line block`.

## One-to-one OWT (our components)

Official: `third_party/blockgen/scripts/train/owt/blockgen_uniform_1_16.sh`.

| Official | Ours (`scripts/submit_blockgen_owt.sh`) |
|----------|------------------------------------------|
| `jdeschena/openwebtext` | `data=openwebtext-blockgen` |
| GPT-2 tokenizer | Qwen2.5-1.5B-Instruct |
| `small-block-dit` scratch | Qwen block, `line=block` |
| `algo=blockgen-uniform` | `algo=block_uniform` + `blockgen_owt_uniform` |
| 1+16, u-strat, pure_noise@1, CE@1 | matched levers (+ ARPC) |
| GBS 512, len 1024, lr 3e-4, 1M | same; `AUTO_RESUME` across 12h walls |

## Single-factor native map (was LEVER A–D)

| Preset | Knobs |
|--------|-------|
| `B3_mixture` | `block_size_mixture=[16,32]` |
| `B3_t_strat` | `stratified_gamma=0.5` (**conversion** track; not BlockGen) |
| `B3_weights_32` / `B3_u_stratified` | weighted 2^k (+ u-strat) |
| `B3_arpc` | mixture incl. 1 + ARPC (**uniform**, `line=block`) |

Always: report **`line × arm × data × budget`**. Never A+B (mixture + t-strat)
in one native micro unless pre-registered as a combo cell.
