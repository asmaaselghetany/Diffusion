#!/usr/bin/env bash
# DEPRECATED — BlockGen micros must use submit_lever.sh with line=block.
#
# The old path hardcoded scripts/slurm/ar2block_*.sbatch, which silently ran
# BlockGen knobs as AR→block conversion. That is transfer (xfer_*), not BlockGen.
#
# Native BlockGen / B3:
#   ./scripts/submit_lever.sh --preset B3_mixture --arm masked --micro
#   ./scripts/submit_lever.sh --preset B3_arpc --arm uniform --micro
#   ./scripts/submit_lever.sh --preset blockgen_uniform --arm uniform --micro
# Full OWT 1+16:
#   ./scripts/submit_blockgen_owt.sh
# Geometry on conversion (explicit transfer, not a BlockGen claim):
#   ./scripts/submit_lever.sh --preset xfer_mixture --arm masked --micro

set -euo pipefail
echo "REFUSED: scripts/submit_blockgen_lever_micro.sh is retired (it wired BlockGen → ar2block)." >&2
echo "Use: ./scripts/submit_lever.sh --preset B3_*|blockgen_* --arm …  (line=block via preset)" >&2
echo "Or:  ./scripts/submit_blockgen_owt.sh" >&2
echo "Transfer-only: ./scripts/submit_lever.sh --preset xfer_* --arm …" >&2
exit 2
