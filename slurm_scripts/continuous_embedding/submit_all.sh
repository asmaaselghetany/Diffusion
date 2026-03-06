#!/bin/bash
# ============================================================================
# Submit all continuous embedding diffusion experiments
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Continuous Embedding Diffusion Experiments"
echo "=========================================="
echo ""
echo "Recommended order:"
echo "1. Quick test (verify pipeline)"
echo "2. Decoder-only baseline (upper bound)"
echo "3. x0 parameterization (most stable)"
echo "4. epsilon/v parameterization (compare)"
echo ""

# Function to submit with confirmation
submit_job() {
    local script=$1
    local name=$2
    
    echo "Submit ${name}? [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        JOB_ID=$(sbatch "${script}" | awk '{print $4}')
        echo "  Submitted: ${name} -> Job ID: ${JOB_ID}"
        echo "${JOB_ID}" >> submitted_jobs.txt
    else
        echo "  Skipped: ${name}"
    fi
}

# Create jobs list file
echo "# Submitted jobs - $(date)" > submitted_jobs.txt

echo ""
echo "=== Step 1: Quick Test (1 GPU, 1 hour) ==="
submit_job "quick_test.sh" "Quick Pipeline Test"

echo ""
echo "=== Step 2: Decoder Baseline (1 GPU, 24 hours) ==="
submit_job "decoder_only_baseline.sh" "Decoder-Only Baseline"

echo ""
echo "=== Step 3: Denoiser Experiments (2 GPUs each, 48 hours) ==="
submit_job "denoiser_x0.sh" "x0 Parameterization"
submit_job "denoiser_epsilon.sh" "Epsilon Parameterization"
submit_job "denoiser_v.sh" "V-Prediction Parameterization"

echo ""
echo "=========================================="
echo "Submission complete!"
echo "Check submitted_jobs.txt for job IDs"
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
