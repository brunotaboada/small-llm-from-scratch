#!/usr/bin/env python3
"""
plot_losses.py — Visualize training and validation loss curves.

=== WHAT DO LOSS CURVES TELL US? ===
The loss curve is the most important diagnostic tool during training:

  • **Decreasing train loss**: The model is learning! It's getting better at
    predicting next tokens on the training data.

  • **Decreasing val loss**: The model is generalizing! It's learning patterns
    that apply to unseen text, not just memorizing.

  • **Train loss decreases but val loss increases**: OVERFITTING! The model is
    memorizing the training data instead of learning general patterns.
    Fix: more data, more regularization (dropout), or stop training earlier.

  • **Loss plateau**: The model has learned as much as it can at this scale.
    Fix: increase model size, dataset size, or training steps.

  • **Loss spikes**: Training instability, often from learning rate too high.
    Fix: lower learning rate, increase warmup, or increase gradient clipping.

=== EXPECTED VALUES ===
  • Random model (untrained): loss ≈ ln(vocab_size) ≈ 8.3 for vocab_size=4000
    (The model has no idea which token comes next, so it assigns equal
    probability to all tokens. −ln(1/4000) = 8.29)
  • After 500 steps (our default): loss ≈ 4.0-5.0
    (Better than random, but still not great)
  • Well-trained small model: loss ≈ 2.5-3.5
    (Achievable with more steps and data)

=== USAGE ===
    python plot_losses.py
"""

import json
import os
import matplotlib.pyplot as plt
from config import CHECKPOINT_DIR


def plot():
    """
    Load loss history from training and create a publication-quality plot.

    Reads loss_history.json from the checkpoints directory (saved by train.py)
    and generates a PNG plot showing both training and validation loss curves.

    The plot is saved to checkpoints/loss_curve.png.
    """
    history_path = os.path.join(CHECKPOINT_DIR, "loss_history.json")
    if not os.path.exists(history_path):
        print(f"No loss history found at {history_path}. Train the model first.")
        return

    # Load the loss history (saved as JSON by train.py)
    with open(history_path) as f:
        history = json.load(f)
    # history = {
    #   "train": [{"step": 25, "loss": 7.5}, {"step": 50, "loss": 6.8}, ...],
    #   "val":   [{"step": 100, "loss": 6.2}, {"step": 200, "loss": 5.5}, ...]
    # }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training loss (logged every log_interval steps)
    if history.get("train"):
        steps = [e["step"] for e in history["train"]]
        losses = [e["loss"] for e in history["train"]]
        ax.plot(steps, losses, label="Train Loss", alpha=0.7, color="steelblue")

    # Plot validation loss (logged every eval_interval steps)
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
