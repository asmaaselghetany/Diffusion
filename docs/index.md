# UNI-D²

**Discrete diffusion models** operate in discrete state spaces (tokens, labels, programs, etc.) by scheduling noise and denoising transitions with carefully designed transition kernels instead of continuous Gaussian noise. 

This repository centralizes tooling, datasets, experiments, and evaluation pipelines so researchers have a reliable, extendible codebase for discrete diffusion variants in text and structured domains.

## Highlights

*   **Unified Entry Point:** Hydra + Lightning workflow for experimenting with MDLM, UDLM, BD3LM, FlexMDM, GIDD, SEDD, and PartitionMDLM.
*   **Comprehensive Sampling:** Helpers for absorbing, autoregressive, block, and flexible sampling strategies.
*   **Reproducibility:** Scripts to reproduce training recipes for datasets like LM1B, OpenWebText, Text8, and Tiny Shakespeare.

## Papers Implemented

1.  **[MDLM](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf)** – Sahoo et al. (NeurIPS 2024)
2.  **[UDLM](https://arxiv.org/pdf/2412.10193)** – Schiff et al. (arXiv 2024)
3.  **[FlexMDM](https://arxiv.org/pdf/2509.01025)** – Kim et al. (arXiv 2025)
4.  **[Block Diffusion](https://arxiv.org/pdf/2503.09573)** – Arriola et al. (arXiv 2025)
5.  **[GIDD](https://arxiv.org/pdf/2503.04482)** – von Rütte et al. (arXiv 2025)
6.  **[SEDD](https://arxiv.org/pdf/2310.16834)** – Lou et al. (arXiv 2023)
7.  **[PartitionMDLM](https://arxiv.org/pdf/2505.18883)** – Deschenaux et al. (arXiv 2025)

---

## Installation

```bash
pip install -e .
```

For systems with Flash Attention (CUDA 11.4+), install it after the editable install to boost throughput:

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Training
Run the Hydra-powered CLI. Here is a minimal example for training MDLM on OpenWebText:

```bash
PYTHONPATH=src python -u -m discrete_diffusion \
  data=owt \
  model=small \
  algo=mdlm \
  loader.batch_size=32 \
  trainer.devices=8 \
  hydra.run.dir=./outputs/owt/mdlm
```

For toy experiments, use the Tiny Shakespeare data config:

```bash
PYTHONPATH=src python -u -m discrete_diffusion \
  data=tiny_shakespeare \
  model=small \
  model.length=128 \
  algo=mdlm \
  loader.global_batch_size=64 \
  trainer.devices=1 \
  trainer.max_steps=2000
```

For SLURM baselines on Tiny Shakespeare (MDLM, latent JEPA, continuous embedding), use:

```bash
sbatch slurm_scripts/shakespeare_baselines/train_mdlm.sh
sbatch slurm_scripts/shakespeare_baselines/train_latent_jepa.sh
sbatch slurm_scripts/shakespeare_baselines/train_continuous_embedding.sh

sbatch --export=ALL,RUN_DIR=/path/to/mdlm_run \
  slurm_scripts/shakespeare_baselines/eval_mdlm_gen_ppl.sh
sbatch --export=ALL,RUN_DIR=/path/to/latent_jepa_run \
  slurm_scripts/shakespeare_baselines/eval_latent_jepa_gen_ppl.sh
sbatch --export=ALL,RUN_DIR=/path/to/continuous_run \
  slurm_scripts/shakespeare_baselines/eval_continuous_gen_ppl.sh
```

The wrapper/core script layout and unified evaluation flow are documented in
`slurm_scripts/shakespeare_baselines/README.md`.

### 2. Sampling
Once you have a checkpoint, use the generation script:

```bash
PYTHONPATH=src python -m discrete_diffusion.evaluations.generate_samples \
  checkpoint_path=outputs/owt/mdlm/checkpoints/last.ckpt \
  num_samples=16 \
  num_steps=2000
```

## Extending

Want to implement a new discrete diffusion method? Check out our [Extension Guides](extension_guides/README.md) to learn how to add custom algorithms, forward processes, noise schedules, and models.
