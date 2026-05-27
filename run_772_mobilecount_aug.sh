#!/bin/bash
set -e
cd /home/umrobotics/MobileCount

# MobileCount shares the Python environment with C-3-Framework
PYTHON=/home/umrobotics/C-3-Framework/.venv/bin/python3

echo "[$(date)] === MobileCount Aug Set 1: Full suite (lr=1e-4, bs=6) ==="
$PYTHON train.py --lr 1e-4 --batch-size 6 --aug-set 1 \
    2>&1 | tee /tmp/mobilecount_772_lr1e4_bs6_aug1.log

echo "[$(date)] === MobileCount Aug Set 2: Flips only (lr=1e-4, bs=6) ==="
$PYTHON train.py --lr 1e-4 --batch-size 6 --aug-set 2 \
    2>&1 | tee /tmp/mobilecount_772_lr1e4_bs6_aug2.log

echo "[$(date)] === MobileCount aug runs complete ==="
