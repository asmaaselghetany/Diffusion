#!/bin/bash
#SBATCH --job-name=ced_resume_all
#SBATCH --partition=gpu_p
##SBATCH --qos=gpu_normal
#SBATCH --qos=gpu_reservation
#SBATCH --reservation=haicu_stefan
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_resume_all_%j.out
#SBATCH --error=/lustre/groups/bauer/projects/discrete_diff_cdb/Discrete_Diffusion_RAE/JEDi/outputs/continuous_embedding/logs/ced_resume_all_%j.err
# ============================================================================
# Resume continuous embedding experiments from latest checkpoints
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Under sbatch, scripts are copied to a spool path; use submission dir when available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/slurm_scripts/continuous_embedding" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
OUTPUT_BASE="${REPO_ROOT}/outputs/continuous_embedding"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-5000}"
WANDB_MODE="${WANDB_MODE:-online}"

latest_run_dir() {
    local prefix="$1"
    ls -1dt "${OUTPUT_BASE}/${prefix}_"* 2>/dev/null | head -n 1 || true
}

submit_resume() {
    local script_name="$1"
    local run_dir="$2"
    local script_path="${REPO_ROOT}/slurm_scripts/continuous_embedding/${script_name}"

    if [[ -z "${run_dir}" ]]; then
        echo "SKIP ${script_name}: no matching run directory found"
        return
    fi
    if [[ ! -f "${script_path}" ]]; then
        echo "SKIP ${script_name}: script not found at ${script_path}"
        return
    fi

    local job_id
    job_id=$(sbatch --parsable \
        --export=ALL,RESUME=1,RESUME_RUN_DIR="${run_dir}",CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS}",WANDB_MODE="${WANDB_MODE}" \
        "${script_path}")

    echo "Submitted ${script_name} from ${run_dir} -> ${job_id}"
}

echo "=========================================="
echo "Resume Continuous Embedding Experiments"
echo "Output base: ${OUTPUT_BASE}"
echo "Checkpoint cadence: every ${CHECKPOINT_EVERY_N_STEPS} train steps"
echo "WandB mode: ${WANDB_MODE}"
echo "=========================================="

submit_resume "decoder_only_baseline.sh" "$(latest_run_dir decoder_only_baseline)"
submit_resume "denoiser_x0.sh" "$(latest_run_dir denoiser_x0)"
submit_resume "denoiser_epsilon.sh" "$(latest_run_dir denoiser_epsilon)"
submit_resume "denoiser_v.sh" "$(latest_run_dir denoiser_v)"
