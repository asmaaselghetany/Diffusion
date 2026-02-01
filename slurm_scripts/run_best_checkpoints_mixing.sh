#!/bin/bash
# Submit stage-2 decoder training jobs with mixing study for multiple sweep checkpoints
# Uses cosine annealing teacher mixing with Bernoulli sampling and teacher corruption
# Output dirs use _mixing suffix to avoid overwriting existing checkpoints
set -euo pipefail

SWEEP_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/jepa_180m_full_study_20251223_202845"
WORKSPACE_BASE="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi"
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"

# Timestamp for unique run identification (prevents checkpoint collisions)
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# Mixing study hyperparameters (recommended defaults from plan)
TEACHER_MIX_WARM_FRAC="${TEACHER_MIX_WARM_FRAC:-0.1}"   # 7% warm window
TEACHER_MIX_TYPE="${TEACHER_MIX_TYPE:-bernoulli}"        # Bernoulli sampling
TEACHER_CORRUPTION_LAMBDA="${TEACHER_CORRUPTION_LAMBDA:-0.7}"  # Corruption lambda
TEACHER_ERROR_EMA_DECAY="${TEACHER_ERROR_EMA_DECAY:-0.99}"     # EMA decay for error sigma

# Decoder dropout override (empty = inherit from Stage 1 checkpoint)
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

CHECKPOINTS=(
  "phase1a_encoder/runs/stage1_180m_encoder_0017_latent_dim_256_hidden_size_768_n_heads_4_n_blocks_8"
  "phase1a_encoder/runs/stage1_180m_encoder_0046_latent_dim_512_hidden_size_384_n_heads_8_n_blocks_8"
  "phase2a_predictor/runs/stage1_180m_predictor_0022_predictor_depth_3_predictor_hidden_size_512_predictor_n_heads_4_predictor_use_projections_False"
  "phase2a_predictor/runs/stage1_180m_predictor_0036_predictor_depth_6_predictor_hidden_size_512_predictor_n_heads_4_predictor_use_projections_False"
  "phase2b_training/runs/stage1_180m_training_0000_loss_norm_cosine_mask_only_False_normalize_targets_True_redundancy_vicreg"
  "phase2b_training/runs/stage1_180m_training_0013_loss_norm_cosine_mask_only_False_normalize_targets_True_redundancy_vicreg"
  "phase2b_training/runs/stage1_180m_training_0033_loss_norm_l2_mask_only_True_normalize_targets_True_redundancy_vicreg"
)

# Extract config from .hydra/config.yaml (YAML format: "  key: value")
get_config() {
  local dir="$1" key="$2" default="$3"
  # Try simplified config.yaml first
  val=$(grep "^${key}:" "${dir}/config.yaml" 2>/dev/null | head -1 | awk '{print $2}')
  # Fall back to .hydra/config.yaml (indented YAML)
  if [[ -z "$val" ]]; then
    val=$(grep "^  ${key#model.}:" "${dir}/.hydra/config.yaml" 2>/dev/null | head -1 | awk '{print $2}')
  fi
  echo "${val:-$default}"
}

echo "=== Mixing Study Configuration ==="
echo "  timestamp: ${TIMESTAMP}"
echo "  warm_frac: ${TEACHER_MIX_WARM_FRAC}"
echo "  mix_type: ${TEACHER_MIX_TYPE}"
echo "  corruption_lambda: ${TEACHER_CORRUPTION_LAMBDA}"
echo "  error_ema_decay: ${TEACHER_ERROR_EMA_DECAY}"
echo "  decoder_dropout: ${DECODER_DROPOUT:-<inherit from stage1>}"
echo ""

echo "=== Validating checkpoints ==="
missing=0
for ckpt_rel in "${CHECKPOINTS[@]}"; do
  ckpt_path="${SWEEP_BASE}/${ckpt_rel}/checkpoints/last.ckpt"
  cfg_path="${SWEEP_BASE}/${ckpt_rel}/.hydra/config.yaml"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "MISSING: $ckpt_path"; missing=$((missing + 1))
  elif [[ ! -f "$cfg_path" ]]; then
    echo "MISSING CONFIG: $cfg_path"; missing=$((missing + 1))
  else
    echo "OK: $(basename "$ckpt_rel")"
  fi
done
[[ $missing -gt 0 ]] && { echo "ERROR: $missing checkpoints missing."; exit 1; }
echo "All checkpoints validated."

echo "=== Submitting mixing study jobs ==="
for ckpt_rel in "${CHECKPOINTS[@]}"; do
  CKPT_DIR="${SWEEP_BASE}/${ckpt_rel}"
  CKPT_PATH="${CKPT_DIR}/checkpoints/last.ckpt"
  RUN_NAME=$(basename "$ckpt_rel")
  # Use _mixing suffix + timestamp to avoid overwriting existing checkpoints
  OUTPUT_DIR="${WORKSPACE_BASE}/outputs/owt/stage2_decoder/${RUN_NAME}_mixing_${TIMESTAMP}"

  # Extract from .hydra/config.yaml under model: section
  HYDRA_CFG="${CKPT_DIR}/.hydra/config.yaml"
  LATENT_DIM=$(grep "^  latent_dim:" "$HYDRA_CFG" | awk '{print $2}')
  HIDDEN_SIZE=$(grep "^  hidden_size:" "$HYDRA_CFG" | awk '{print $2}')
  N_HEADS=$(grep "^  n_heads:" "$HYDRA_CFG" | head -1 | awk '{print $2}')
  N_BLOCKS=$(grep "^  n_blocks:" "$HYDRA_CFG" | awk '{print $2}')
  PREDICTOR_DEPTH=$(grep "^  predictor_depth:" "$HYDRA_CFG" | awk '{print $2}')
  PREDICTOR_HIDDEN=$(grep "^  predictor_hidden_size:" "$HYDRA_CFG" | awk '{print $2}')
  PREDICTOR_HEADS=$(grep "^  predictor_n_heads:" "$HYDRA_CFG" | awk '{print $2}')
  PREDICTOR_USE_PROJ=$(grep "^  predictor_use_projections:" "$HYDRA_CFG" | awk '{print $2}')
  READOUT_TYPE=$(grep "^  readout_type:" "$HYDRA_CFG" | awk '{print $2}')
  READOUT_DEPTH=$(grep "^  readout_depth:" "$HYDRA_CFG" | awk '{print $2}')
  READOUT_HIDDEN=$(grep "^  readout_hidden_size:" "$HYDRA_CFG" | awk '{print $2}')
  TIME_EMBED_DIM=$(grep "^  time_embed_dim:" "$HYDRA_CFG" | awk '{print $2}')
  # Use override if set, otherwise inherit from Stage 1
  if [[ -n "${DECODER_DROPOUT}" ]]; then
    DROPOUT="${DECODER_DROPOUT}"
  else
    DROPOUT=$(grep "^  dropout:" "$HYDRA_CFG" | head -1 | awk '{print $2}')
  fi

  echo "Submitting: $RUN_NAME (mixing study)"
  echo "  Output: $OUTPUT_DIR"
  echo "  latent=$LATENT_DIM hidden=$HIDDEN_SIZE heads=$N_HEADS blocks=$N_BLOCKS pred_depth=$PREDICTOR_DEPTH"

  sbatch --export=ALL \
    --job-name="s2m_${RUN_NAME:0:28}" \
    --output="${WORKSPACE_BASE}/logs/s2m_${RUN_NAME}_${TIMESTAMP}_%j.out" \
    --error="${WORKSPACE_BASE}/logs/s2m_${RUN_NAME}_${TIMESTAMP}_%j.err" \
    "${REPO_ROOT}/slurm_scripts/stage2_mixing_job_template.sh" \
    "$CKPT_PATH" "$OUTPUT_DIR" "$RUN_NAME" \
    "$LATENT_DIM" "$HIDDEN_SIZE" "$N_HEADS" "$N_BLOCKS" \
    "$PREDICTOR_DEPTH" "$PREDICTOR_HIDDEN" "$PREDICTOR_HEADS" \
    "$READOUT_TYPE" "$READOUT_DEPTH" "$READOUT_HIDDEN" \
    "$TIME_EMBED_DIM" "$DROPOUT" "$PREDICTOR_USE_PROJ" \
    "$TEACHER_MIX_WARM_FRAC" "$TEACHER_MIX_TYPE" \
    "$TEACHER_CORRUPTION_LAMBDA" "$TEACHER_ERROR_EMA_DECAY"
done

echo "=== All mixing study jobs submitted ==="


