#!/usr/bin/env bash
# Idempotent Jupiter setup after cloning upstream Diffusion.
# Translates HAICORE paths and applies scifi Slurm defaults.
#
# Usage (login node, once after clone or git pull with upstream sbatch changes):
#   bash scripts/setup_juwels.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "Patching HAICORE / legacy paths -> Jupiter Diffusion-new..."
find scripts tools -type f \( -name '*.sbatch' -o -name '*.sh' -o -name '*.bash' \) -print0 \
  | xargs -0 sed -i \
    -e 's|/fast/project/HFMI_SynergyUnit/asmaa.elsayed|/e/project1/scifi/elsayed3|g' \
    -e 's|/p/project1/scifi|/e/project1/scifi|g' \
    -e 's|/p/scratch/scifi|/e/scratch/scifi|g' \
    -e 's|/e/project1/scifi/elsayed3/Diffusion[^-]|/e/project1/scifi/elsayed3/Diffusion-new|g'

echo "Done. Jupiter-specific files (do not overwrite on git pull):"
echo "  scripts/jupiter_paths.sh"
echo "  scripts/slurm/_common.sh"
echo "  scripts/_block_qwen_env.bash"
echo "  /e/project1/scifi/elsayed3/env.sh"
echo ""
echo "Next: bash scripts/setup_venv.sh"
