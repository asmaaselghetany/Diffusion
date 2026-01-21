#!/bin/bash
# validate_all_checkpoints.sh

PHASE_DIR="${1:-/hkfs/home/project/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/sweep_outputs/jepa_180m_full_study_20251223_202845/phase1a_encoder}"
REPO_ROOT="${REPO_ROOT:-/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa}"

echo "Validating checkpoints in: $PHASE_DIR"
echo ""

# Setup Python environment
cd "$REPO_ROOT"
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    source "${REPO_ROOT}/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

export PYTHONPATH="${REPO_ROOT}/src"

valid_count=0
corrupted_count=0
missing_count=0
error_count=0

# Find all checkpoint files
while IFS= read -r -d '' ckpt; do
    if [[ ! -f "$ckpt" ]]; then
        echo "✗ Missing: $ckpt"
        ((missing_count++))
        continue
    fi
    
    size=$(stat -f%z "$ckpt" 2>/dev/null || stat -c%s "$ckpt" 2>/dev/null)
    
    if [[ $size -eq 0 ]]; then
        echo "✗ Empty file: $ckpt"
        ((corrupted_count++))
        continue
    fi
    
    # Validate with Python - capture error for debugging
    error_msg=$(python3 -c "
import torch
import sys
try:
    # Use weights_only=False to allow custom classes like omegaconf.DictConfig
    checkpoint = torch.load('$ckpt', map_location='cpu', weights_only=False)
    # Basic validation: check if it's a dict with expected keys
    if isinstance(checkpoint, dict):
        # Check for common checkpoint keys
        has_state = 'state_dict' in checkpoint or 'model' in checkpoint or 'epoch' in checkpoint
        if has_state or len(checkpoint) > 0:
            sys.exit(0)
    sys.exit(0)  # Even if structure is unexpected, file loaded successfully
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
" 2>&1)
    
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ Valid: $ckpt ($size bytes)"
        ((valid_count++))
    else
        echo "✗ Corrupted: $ckpt ($size bytes)"
        # Show error for first few corrupted files for debugging
        if [[ $error_count -lt 3 ]]; then
            echo "  Error: ${error_msg}"
            ((error_count++))
        fi
        ((corrupted_count++))
    fi
done < <(find "$PHASE_DIR" -name "*.ckpt" -type f -print0)

echo ""
echo "Summary:"
echo "  Valid: $valid_count"
echo "  Corrupted: $corrupted_count"
echo "  Missing: $missing_count"