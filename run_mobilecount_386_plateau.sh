#!/bin/bash
# Run MobileCount on 386x260 with a single LR to 600 epochs (no plateau stop).
# Usage: bash run_mobilecount_386_plateau.sh <lr>
# Example: bash run_mobilecount_386_plateau.sh 1e-4
set -e
cd /home/umrobotics/MobileCount

PYTHON=/home/umrobotics/C-3-Framework/.venv/bin/python3
DATA_PATH=datasets/Tenebrio/386x260

LR=${1:?"Usage: $0 <lr>  e.g. $0 1e-4"}
LOG=/tmp/mobilecount_386_lr${LR}_bs6.log

echo "[$(date)] Starting MobileCount 386x260 lr=${LR} bs=6 wd=1e-4 (600 epochs)"

$PYTHON train.py \
    --lr "$LR" \
    --batch-size 6 \
    --weight-decay 1e-4 \
    --data-path "$DATA_PATH" \
    > "$LOG" 2>&1

echo "[$(date)] Training with lr=${LR} finished."

# Find the most-recently-modified 386x260 experiment directory
EXP_DIR=$(ls -td exp/*_386x260 2>/dev/null | head -1)
if [ -n "$EXP_DIR" ]; then
    echo "Saving training curves for ${EXP_DIR} ..."
    $PYTHON scripts/plot_training_curves.py "${EXP_DIR}"
else
    echo "WARNING: could not determine EXP_DIR; skipping plot."
fi

echo "[$(date)] Done: MobileCount 386x260 lr=${LR}"
