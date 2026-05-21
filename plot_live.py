#!/usr/bin/env python3
"""Parse train_193x130.log and save an up-to-date loss curve PNG.

Run once for a snapshot, or loop (e.g. watch -n 30 python3 plot_live.py)
to keep it refreshed while training.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = Path("train_193x130_adamw.log")
OUT_PATH = Path("loss_curve_adamw_live.png")

ITER_RE = re.compile(r"\[ep (\d+)\]\[it \d+\]\[loss ([0-9.eE+\-]+)\]")
VAL_RE  = re.compile(r"\[mae [0-9.]+ mse [0-9.]+\], \[val loss ([0-9.eE+\-]+)\]")
# "train time:" marks end of an epoch; used to align val entries to epochs.
TRAIN_END_RE = re.compile(r"train time:")


def parse_log(path: Path):
    text = path.read_text(errors="replace")

    # --- train loss: average per-iteration losses per epoch ---
    epoch_iters = defaultdict(list)
    for m in ITER_RE.finditer(text):
        epoch_iters[int(m.group(1))].append(float(m.group(2)))

    train_epochs, train_losses = [], []
    for ep in sorted(epoch_iters):
        vals = epoch_iters[ep]
        train_epochs.append(ep)
        train_losses.append(sum(vals) / len(vals))

    # --- val loss: scan line-by-line, track which epoch each val belongs to ---
    # Epoch N (1-indexed) triggers a val block after its "train time:" line.
    # Validation runs when: epoch % VAL_FREQ == 0 or epoch > VAL_DENSE_START
    # (0-indexed epoch, so displayed ep = epoch+1).
    val_epochs, val_losses = [], []
    current_ep = None
    for line in text.splitlines():
        m_iter = ITER_RE.search(line)
        if m_iter:
            current_ep = int(m_iter.group(1))
        m_val = VAL_RE.search(line)
        if m_val and current_ep is not None:
            val_epochs.append(current_ep)
            val_losses.append(float(m_val.group(1)))

    return train_epochs, train_losses, val_epochs, val_losses


def plot(train_epochs, train_losses, val_epochs, val_losses, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    if train_epochs:
        ax.plot(train_epochs, train_losses,
                label="Train loss (epoch avg)", color="steelblue", linewidth=1.5)

    if val_epochs:
        ax.plot(val_epochs, val_losses,
                label="Val loss", color="darkorange",
                linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"MobileCount · Tenebrio 193×130 · 1e-4 Adam"
        + (f"  (ep {max(train_epochs)}/600)" if train_epochs else "")
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  "
          f"(train ep: {len(train_epochs)}, val ep: {len(val_epochs)})")


def main():
    if not LOG_PATH.exists():
        print(f"Log not found: {LOG_PATH}", file=sys.stderr)
        sys.exit(1)

    train_epochs, train_losses, val_epochs, val_losses = parse_log(LOG_PATH)
    plot(train_epochs, train_losses, val_epochs, val_losses, OUT_PATH)


if __name__ == "__main__":
    main()
