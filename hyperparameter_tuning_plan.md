# 

**Document Version**: 1.0

**Date**: December 2024

**Target Models**: 180M and 350M Parameter Configurations

**Training Dataset**: OpenWebText (OWT)

**Training Framework**: Two-Stage Discrete Diffusion with EMA Teacher

---

## Executive Summary

This document outlines a systematic hyperparameter optimization study for the Latent JEPA discrete diffusion model. The architecture follows a three-component design: **Encoder** (tokens → latents), **Predictor** (noisy latents → clean latents), and **Readout Decoder** (latents → token logits). Training is conducted in two stages:

1. **Stage 1**: Joint training of encoder, predictor, and readout in latent space with EMA teacher
2. **Stage 2**: Frozen encoder/predictor with decoder-only fine-tuning

Each component requires independent hyperparameter grids evaluated against generative perplexity and output entropy metrics.

---

## Table of Contents

1. [Model Architecture Overview](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
2. [Evaluation Metrics](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
3. [180M Model Configuration Grid](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
4. [350M Model Configuration Grid](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
5. [Stage 1 Training Hyperparameters](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
6. [Stage 2 Training Hyperparameters](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
7. [Optimizer and Learning Rate Schedule](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
8. [EMA Teacher Configuration](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
9. [Experimental Protocol](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
10. [Compute Budget Estimation](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
11. [Ablation Study Design](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)
12. [Statistical Analysis Guidelines](https://www.notion.so/Hyperparameter-tuning-plan-2c5b1c9d5064800daa70dbad7bb067a1?pvs=21)

---

## 1. Model Architecture Overview

### 1.1 Component Descriptions

| Component | Function | Parameters (180M) | Parameters (350M) |
| --- | --- | --- | --- |
| Student Encoder | Token embeddings → Latent representations | ~60M | ~120M |
| Teacher Encoder | EMA copy of student (frozen) | ~60M | ~120M |
| Predictor | Denoising in latent space | ~25M | ~50M |
| Readout Decoder | Latent → Token logits | ~35M | ~60M |

### 1.2 Data Flow

```mermaid
Stage 1: x_0 → [Forward Process] → x_t → [Student Encoder] → z_t
                                            ↓
         x_0 → [Teacher Encoder] → z_0 ← [Predictor] ← z_t
                                            ↓
                             L_JEPA = MSE(ẑ_0, z_0) + VICReg(ẑ_0)
                                            ↓
                              [Readout] → logits → L_readout

Stage 2: x_t → [Frozen Encoder] → z_t → [Frozen Predictor] → ẑ_0
                                                              ↓
                                              [Trainable Readout] → logits
                                                              ↓
                                              L_CE(logits, x_0)[masked]
```

### 1.3 Training Dataset

| Property | Value | Notes |
| --- | --- | --- |
| **Dataset** | OpenWebText (OWT) | Open-source GPT-2 pretraining corpus |
| **Tokenizer** | GPT-2 BPE | vocab_size = 50,257 |
| **Sequence Length** | 256 (default) | Configurable: 128, 256, 512 |
| **Training Samples** | ~9B tokens | Full OWT corpus |
| **Validation** | 5% holdout | ~450M tokens |

**Rationale**: OWT provides a standardized benchmark for language modeling experiments, enabling direct comparison with prior work (MDLM, SEDD, D3PM). The GPT-2 tokenizer ensures compatibility with evaluation using GPT-2-large as the reference model.

---

## 2. Evaluation Metrics

### 2.1 Primary Metrics

| Metric | Description | Target Range | Computation |
| --- | --- | --- | --- |
| **Generative Perplexity** | GPT-2-large evaluated PPL on generated samples | Lower is better (< 30 ideal) | `exp(mean(NLL))` on 256-512 samples |
| **Output Entropy** | Shannon entropy of token distributions | 3.5-5.5 nats (balanced diversity) | `-Σ p(x) log p(x)` |

---

## 3. 180M Model Configuration Grid

### 3.1 Encoder Grid (180M)

**Baseline**: `latent_dim=512, hidden_size=512, n_heads=8, n_blocks=12`

| Parameter | Grid Values | Notes |
| --- | --- | --- |
| `latent_dim` | [256, **512**, 768] | Representation bottleneck width |
| `hidden_size` | [384, **512**, 768] | Internal transformer width |
| `n_heads` | [4, **8**, 12] | Must divide hidden_size cleanly |
| `n_blocks` | [6, 8, **12**, 16] | Encoder depth |
| `dropout` | [**0.0**, 0.05, 0.1] | I-JEPA typically uses 0.0 |
| `time_embed_dim` | [64, **128**, 256] | Time conditioning capacity |

**Grid Size**: 3 × 3 × 3 × 4 × 3 × 3 = **972 combinations**

**Recommended**: Fractional factorial (Latin hypercube) with **50-80 runs**

### 3.2 Predictor Grid (180M)

**Baseline**: `predictor_type="transformer", predictor_depth=6, predictor_hidden_size=384, predictor_n_heads=6`

| Parameter | Grid Values | Notes |
| --- | --- | --- |
| `predictor_type` | [**"transformer"**] | MLP cheaper but less expressive |
| `predictor_depth` | [3, **6**, 9, 12] | I-JEPA uses 6-12 for large encoders |
| `predictor_hidden_size` | [256, **384**, 512] | Narrow bottleneck (JEPA style) |
| `predictor_n_heads` | [4, **6**, 8] | Must divide predictor_hidden_size |
| `predictor_use_projections` | [**true**, false] | Input/output projections |

**Grid Size** (transformer only): 4 × 3 × 3 × 2 = **72 combinations**

**Recommended**: Full grid for transformer, sampled for MLP ablation

### 3.3 Readout Decoder Grid (180M)

**Baseline**: `readout_type="tiny_transformer", readout_hidden_size=512, readout_depth=2`

| Parameter | Grid Values | Notes |
| --- | --- | --- |
| `readout_type` | [**"tiny_transformer**"] | Architecture class |
| `readout_hidden_size` | [256, **512**, 768, 1024] | Decoder capacity |
| `readout_depth` | [1, **2**, 4, 6, 8] | Decoder depth |
| `readout_n_heads` | [4, **8**, 12] | For transformer types |
| `readout_bidirectional` | [**true**, false] | Causal vs bidirectional attention |
| `tie_readout_to_embedding` | [true, **false**] | Weight tying (tied_linear only) |

**Grid Size** (tiny_transformer focus): 4 × 5 × 3 × 2 = **120 combinations**

**Recommended**: Focus on `tiny_transformer` and `transformer` variants: **~60 runs**

---

## 4. 350M Model Configuration Grid

### 4.1 Encoder Grid (350M)

**Baseline (Option A)**: `latent_dim=768, hidden_size=768, n_heads=12, n_blocks=24`

**Baseline (Option B)**: `latent_dim=512, hidden_size=512, n_heads=8, n_blocks=32`

| Parameter | Grid Values (A) | Grid Values (B) | Notes |
| --- | --- | --- | --- |
| `latent_dim` | [512, **768**, 1024] | [**512**, 768] | Representation width |
| `hidden_size` | [512, **768**, 1024] | [**512**, 768, 1024] | Transformer width |
| `n_heads` | [8, **12**, 16] | [**8**, 12, 16] | Attention heads |
| `n_blocks` | [16, 20, **24**, 28] | [24, 28, **32**, 36] | Depth |
| `dropout` | [**0.0**, 0.05] | [**0.0**, 0.05] | Regularization |
| `time_embed_dim` | [**128**, 256] | [**128**, 256] | Time conditioning |

**Recommended**: Compare both baselines, then fine-tune best configuration: **~40 runs per baseline**

### 4.2 Predictor Grid (350M)

**Baseline**: `predictor_depth=12, predictor_hidden_size=384, predictor_n_heads=6`

| Parameter | Grid Values | Notes |
| --- | --- | --- |
| `predictor_depth` | [6, 9, **12**, 16] | Deeper for larger encoders |
| `predictor_hidden_size` | [**384**, 512, 768] | Bottleneck vs wider |
| `predictor_n_heads` | [**6**, 8, 12] | Must match hidden_size |
| `predictor_use_projections` | [**true**, false] | Required if hidden != latent_dim |

**Grid Size**: 4 × 3 × 3 × 2 = **72 combinations**

**Recommended**: **50 runs** with importance sampling

### 4.3 Readout Decoder Grid (350M)

**Baseline**: `readout_type="tiny_transformer", readout_hidden_size=768, readout_depth=4`

| Parameter | Grid Values | Notes |
| --- | --- | --- |
| `readout_type` | [**"tiny_transformer"**] | Architecture |
| `readout_hidden_size` | [512, **768**, 1024] | Match encoder or independent |
| `readout_depth` | [2, **4**, 6, 8] | Probe depth |
| `readout_n_heads` | [8, **12**, 16] | Attention capacity |
| `readout_bidirectional` | [**true**, false] | Attention masking |

**Grid Size**: 3 × 3 × 4 × 3 × 2 = **216 combinations**

**Recommended**: **60 runs** focused on transformer variants

---

## 5. Stage 1 Training Hyperparameters

### 5.1 Loss Configuration Grid

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `loss_norm` | [**"l2"**, "cosine"] | MSE vs cosine distance |
| `mask_only` | [**true**, false] | Loss on masked positions only |
| `normalize_targets` | [**true**, false] | LayerNorm on teacher targets |

### 5.2 VICReg Regularization Grid

| Parameter | Grid Values | Description |  |
| --- | --- | --- | --- |
| `redundancy` | [**"vicreg"**, "barlow", "none"] | Regularization type |  |
| `lambda_var` | [0.0, 0.05, **0.1**, 0.25, 0.5] | Variance regularization |  |
| `lambda_cov` | [0.0, 0.02, **0.04**, 0.1, 0.25] | Covariance regularization |  |
| `vicreg_eps` | [1e-5, **1e-3**, 1e-2] | Numerical stability |  |

**Key Interaction**: `(lambda_var, lambda_cov)` should be tuned jointly

**Recommended**: Grid search on `(lambda_var, lambda_cov)` with 5×5 = **25 combinations**

### 5.3 Noise Warmup Curriculum Grid

| Parameter | Grid Values | Description |
| --- | --- | --- |
|  |  |  |
|  |  |  |
| `sampling_eps` | [**1e-3**, 1e-2] | Timestep endpoint avoidance |

### 5.4 Stage 1 Full Factorial Design

**Critical Combinations** (cross-product of most impactful parameters):

```yaml
stage1_critical_grid:
  redundancy: ["vicreg", "none"]
  lambda_var: [0.0, 0.1, 0.25]
  lambda_cov: [0.0, 0.04, 0.1]
  loss_norm: ["l2", "cosine"]
  normalize_targets: [true, false]
  noise_warmup_end_step: [0, 10000]

```

**Total**: 2 × 3 × 3 × 2 × 2 × 2 = **144 combinations**

**Recommended**: Fractional factorial with **36-48 runs** (1/3 - 1/4 fraction)

---

## 6. Stage 2 Training Hyperparameters

### 6.1 Latent Source Configuration

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `latent_source` | [**"predicted"**, "teacher", "mixed"] | Source of input latents |
| `mixed_schedule` | ["constant", **"linear"**] | Teacher→predicted annealing |
| `mixed_warmup_steps` | [10000, 25000, **50000**, 100000] | Annealing duration |

### 6.2 Loss Configuration

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `time_sampling` | ["uniform", **"hazard"**] | Timestep distribution |
| `weight_by_hazard` | [**true**, false] | Hazard weighting for uniform sampling |
| `label_smoothing` | [**0.0**, 0.05, 0.1] | CE label smoothing |
| `sampling_eps` | [**1e-3**, 1e-2] | Endpoint avoidance |
|  |  |  |

### 6.3 Knowledge Distillation (Optional)

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `kd_weight` | [**0.0** (disabled), 0.1, 0.5, 1.0] | KD loss weight |
| `kd_temperature` | [1.0, **2.0**, 4.0] | Distillation temperature |

**Note**: KD requires a pretrained teacher LM; disabled by default.

### 6.4 Stage 2 Recommended Grid

**Critical Combinations**:

```yaml
stage2_critical_grid:
  latent_source: ["predicted", "mixed"]
  mixed_warmup_steps: [25000, 50000]  # Only if mixed
  time_sampling: ["uniform", "hazard"]
  weight_by_hazard: [true, false]  # Only if uniform
  label_smoothing: [0.0, 0.05, 0.1]

```

**Total**: ~36 meaningful combinations

**Recommended**: **24-30 runs** with stratified sampling

---

## 7. Optimizer and Learning Rate Schedule

### 7.1 Optimizer Configuration

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `optimizer` | [**AdamW**, Muon] | Optimizer type |
| `lr` | [1e-4, **3e-4**, 5e-4, 1e-3] | Base learning rate |
| `weight_decay` | [**0.0**, 0.01, 0.1] | L2 regularization |
| `betas` | [**(0.9, 0.999)**, (0.9, 0.98), (0.95, 0.999)] | Momentum parameters |

### 7.2 Learning Rate Schedule

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `warmup_steps` | [1000, **2000**, 5000, 10000] | Linear warmup duration |
| `start_factor` | [**1e-3**, 1e-2, 0.1] | Initial LR multiplier |
| `cosine_restarts` | [**false**, true] | Warm restarts |
| `num_cycles` | [1, **2**, 4] | Restart cycles (if enabled) |

### 7.3 Recommended Schedule Configurations

**Stage 1 (Long Training)**:

```yaml
lr: 3e-4
warmup_steps: 2000
start_factor: 1e-3
max_steps: 200000
cosine_restarts: false

```

**Stage 2 (Decoder Training)**:

```yaml
lr: 1e-4  # Lower for fine-tuning
warmup_steps: 1000
start_factor: 1e-3
max_steps: 50000
cosine_restarts: false

```

---

## 8. EMA Teacher Configuration

### 8.1 EMA Schedule Grid

| Parameter | Grid Values | Description |
| --- | --- | --- |
| `ema_decay` | [0.99, **0.996**, 0.999] | Initial momentum |
| `ema_warmup_steps` | [50000, 100000, **200000**] | Cosine warmup duration |
| `ema_final_decay` | [0.999, **0.9999**, 0.99999] | Final momentum |
| `copy_on_init` | [**true**, false] | Teacher initialization |

### 8.2 EMA Schedule Analysis

The EMA schedule follows a cosine warmup:

```
decay(t) = ema_final_decay - (ema_final_decay - ema_decay) * 0.5 * (1 + cos(π * t / warmup_steps))

```

**Recommended Configurations**:

| Configuration | ema_decay | ema_final_decay | warmup_steps | Use Case |
| --- | --- | --- | --- | --- |
| **Aggressive** | 0.99 | 0.9999 | 100000 | Fast adaptation |
| **Standard** | 0.996 | 0.9999 | 200000 | Balanced (default) |
| **Conservative** | 0.999 | 0.99999 | 300000 | Stable, slow adaptation |

---

## 9. Experimental Protocol

### 9.1 Study Phases

| Phase | Objective | Duration | Runs |
| --- | --- | --- | --- |
| **Phase 1A** | Encoder architecture sweep (180M) | 2 weeks | 50-80 |
| **Phase 1B** | Encoder architecture sweep (350M) | 2 weeks | 40-60 |
| **Phase 2A** | Predictor sweep (best encoder from 1A/1B) | 1 week | 50 |
| **Phase 2B** | Stage 1 training hyperparameters | 1 week | 40 |
| **Phase 3** | Readout decoder sweep (Stage 2) | 1 week | 60 |
| **Phase 4** | Cross-component validation | 1 week | 30 |
| **Phase 5** | Final ablations and statistical validation | 1 week | 20 |

### 9.2 Evaluation Checkpoints

Each experiment should evaluate at these checkpoints:

| Checkpoint | Steps | Evaluation |
| --- | --- | --- |
| **Early** | 10,000 | Latent metrics only (collapse detection) |
| **Quarter** | 50,000 | Full evaluation (PPL, entropy, latent) |
| **Half** | 100,000 | Full evaluation |
| **Final** | 200,000 | Full evaluation + sampling analysis |

### 9.3 Cross-Validation Protocol

For each hyperparameter configuration:

1. **Stage 1 Training** (200k steps)
    - Evaluate: latent metrics, reconstruction accuracy
    - Save: encoder/predictor checkpoint
2. **Stage 2 Training** (50k steps)
    - Load: frozen encoder/predictor from Stage 1
    - Sweep: readout decoder configurations
    - Evaluate: generative perplexity, output entropy
3. **Cross-Stage Evaluation**
    - Best Stage 1 config × Best Stage 2 config
    - Additional combinations for interaction analysis

### 9.4 Hyperparameter Interaction Analysis

**Critical Interactions to Test**:

| Interaction | Hypothesis | Test |
| --- | --- | --- |
| `latent_dim` × `predictor_hidden_size` | Bottleneck ratio matters | Grid search |
| `n_blocks` × `predictor_depth` | Encoder/predictor depth balance | Diagonal sweep |
| `lambda_var` × `lambda_cov` | VICReg regularization trade-off | Full 5×5 grid |
| `readout_depth` × `latent_source` | Decoder capacity × input quality | 4×3 grid |
| `hidden_size` × `readout_hidden_size` | Width matching | Compare matched vs mismatched |

---

## 10. Compute Budget Estimation

### 10.1 Single Run Estimates

| Model | Stage 1 (200k steps) | Stage 2 (50k steps) | Evaluation |
| --- | --- | --- | --- |
| 180M | ~12h (4×A100) | ~3h (4×A100) | ~30min |
| 350M | ~24h (4×A100) | ~6h (4×A100) | ~45min |

### 10.2 Total Budget

| Phase | Runs | GPU-Hours (180M) | GPU-Hours (350M) |
| --- | --- | --- | --- |
| Phase 1A/1B | 100-140 | 1,200-1,680 | 2,400-3,360 |
| Phase 2A | 50 | 600 | 1,200 |
| Phase 2B | 40 | 480 | 960 |
| Phase 3 | 60 | 180 | 360 |
| Phase 4 | 30 | 450 | 900 |
| Phase 5 | 20 | 300 | 600 |
| **Total** | 300-340 | **3,210-3,690** | **6,420-7,380** |

**Total Estimated**: ~10,000-11,000 GPU-hours (A100-80GB equivalent)

### 10.3 Resource Optimization Strategies

1. **Early Stopping**: Terminate runs with collapse (std < 0.1) or divergence (loss > 10×baseline) by step 10k
2. **Checkpoint Reuse**: Share Stage 1 checkpoints across Stage 2 decoder experiments
3. **Progressive Grid Refinement**: Coarse grid → Fine-tune best regions
4. **Asynchronous Scheduling**: Overlap evaluation with next run's warmup