#!/usr/bin/env python3
"""
plot_losses.py — Visualize training and validation loss curves.

Usage:
    python plot_losses.py
"""

import json
import os
import matplotlib.pyplot as plt
from config import CHECKPOINT_DIR


def plot():
    history_path = os.path.join(CHECKPOINT_DIR, "loss_history.json")
    if not os.path.exists(history_path):
        print(f"No loss history found at {history_path}. Train the model first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Training loss
    if history.get("train"):
        steps = [e["step"] for e in history["train"]]
        losses = [e["loss"] for e in history["train"]]
        ax.plot(steps, losses, label="Train Loss", alpha=0.7, color="steelblue")

    # Validation loss
    if history.get("val"):
        steps = [e["step"] for e in history["val"]]
        losses = [e["loss"] for e in history["val"]]
        ax.plot(steps, losses, label="Val Loss", linewidth=2, color="orangered",
                marker="o", markersize=5)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (Cross-Entropy)", fontsize=12)
    ax.set_title("SmallGPT Training Progress", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Loss curve saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    plot()
