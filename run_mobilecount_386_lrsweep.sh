#!/bin/bash
# MobileCount 386x260 LR sweep: lr=1e-6, 1e-5, 1e-4, 1e-3  (AdamW wd=1e-4, bs=6, 600 epochs)
# Usage: bash run_mobilecount_386_lrsweep.sh
set -e
cd /home/umrobotics/MobileCount

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for LR in 1e-6 1e-5 1e-4 1e-3; do
    echo ""
    echo "========================================"
    echo "[$(date)] MobileCount sweep: lr=${LR}"
    echo "========================================"
    bash "$SCRIPT_DIR/run_mobilecount_386_plateau.sh" "$LR"
done

echo ""
echo "[$(date)] MobileCount LR sweep complete."
