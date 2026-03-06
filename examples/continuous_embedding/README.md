# Continuous Embedding Diffusion

This example demonstrates training a continuous Gaussian diffusion model on pretrained text embeddings from Qwen3-Embedding-0.6B.

## Overview

Unlike discrete diffusion models that operate on token IDs, continuous embedding diffusion:

1. **Encodes** text using a frozen pretrained embedder (Qwen3-Embedding-0.6B)
2. **Diffuses** in continuous embedding space using Gaussian noise
3. **Denoises** using a transformer-based denoiser
4. **Decodes** back to discrete tokens using a learned decoder

## Architecture

```
Text -> [Frozen Qwen3-Embedding] -> z_0 (clean embeddings)
                                      |
                                      v
                              [Gaussian Forward Process]
                                      |
                                      v
                                z_t (noisy embeddings)
                                      |
                                      v
                          [Trainable Denoiser Transformer]
                                      |
                                      v
                             z_hat_0 (predicted clean)
                                      |
                                      v
                          [Learned Decoder Transformer]
                                      |
                                      v
                              Token Logits -> Tokens
```

## Training Stages

### Stage 1: Joint Denoiser + Decoder Training

```bash
# Single GPU
./owt.sh

# Multi-GPU
DEVICES=4 BATCH_SIZE=128 ./owt.sh
```

Loss = MSE(z_hat_0, z_0) + lambda_ce * CE(logits, tokens)

### Stage 2: Decoder-Only Fine-tuning (Optional)

```bash
CHECKPOINT=/path/to/stage1/checkpoint.ckpt ./stage2.sh
```

Freezes the denoiser and trains only the decoder with teacher-forced z_0.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algo.parameterization` | `x0` | What denoiser predicts: `x0`, `epsilon`, or `v` |
| `algo.loss_type` | `simple` | `simple` (MSE) or `weighted` (ELBO-style) |
| `algo.lambda_ce` | `0.1` | Weight for cross-entropy decoder loss |
| `model.n_blocks` | `12` | Denoiser transformer depth |
| `model.hidden_size` | `1024` | Denoiser hidden dimension |
| `model.decoder_n_blocks` | `2` | Decoder transformer depth |

## Sampling

The model supports multiple sampling strategies:

- **DDPM**: Stochastic reverse process (default)
- **DDIM**: Deterministic/faster sampling
- **Ancestral**: Alternative stochastic sampler

```python
from discrete_diffusion.sampling import ContinuousEmbeddingSampler

sampler = ContinuousEmbeddingSampler(
    config,
    sampling_method="ddim",  # or "ddpm", "ancestral"
    eta=0.0,                 # 0=deterministic, 1=stochastic
)

tokens = sampler.generate(
    model,
    num_samples=16,
    num_steps=50,
)
```

## Requirements

- PyTorch 2.0+
- Transformers (for Qwen3-Embedding)
- ~8GB GPU memory for inference
- ~24GB GPU memory for training (with gradient checkpointing)

## Notes

- The Qwen3-Embedding model is frozen by default for stability
- Gradient checkpointing is enabled by default to reduce memory usage
- Generation quality depends heavily on the decoder; consider deeper decoders for better results
