#!/usr/bin/env python3
"""
train.py — Complete training pipeline for the small GPT language model.

This script handles:
  • Model and optimizer initialization
  • Learning rate schedule (linear warmup + cosine decay)
  • Training loop with gradient clipping
  • Periodic validation
  • Checkpoint saving & loading
  • Loss logging (prints + optional matplotlib plot)

Usage:
    python train.py                    # Train from scratch
    python train.py --resume latest    # Resume from latest checkpoint

The training objective is **causal language modelling** (next-token prediction):
    loss = CrossEntropy(predicted_logits, true_next_tokens)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import ModelConfig, TrainConfig, CHECKPOINT_DIR
from model import SmallGPT
from dataset import create_dataloaders
from train_tokenizer import load_tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Learning Rate Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Learning rate schedule: linear warmup → cosine decay.

    1. Warmup phase (steps 0..warmup_steps):
       LR increases linearly from 0 to learning_rate.
       This helps stabilize early training when gradients are noisy.

    2. Cosine decay phase (steps warmup_steps..max_steps):
       LR decreases following a cosine curve from learning_rate to ~0.
       This gradual reduction helps the model converge to a better minimum.

    This schedule is used by GPT-3 and many modern LLMs.
    """
    # Warmup phase
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step / max(1, cfg.warmup_steps))

    # Cosine decay phase
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    # Cosine annealing to 10% of peak LR
    min_lr = cfg.learning_rate * 0.1
    return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: torch.device, max_batches: int = 50) -> float:
    """
    Run validation and return average loss.

    We limit to `max_batches` to keep validation fast during training.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids, target_ids in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        _, loss = model(input_ids, targets=target_ids)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_losses: list,
    val_losses: list,
    cfg: TrainConfig,
):
    """Save a training checkpoint."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, path)
    # Also save a "latest" symlink for easy resuming
    latest_path = os.path.join(cfg.checkpoint_dir, "checkpoint_latest.pt")
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(path), latest_path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, list, list]:
    """Load a training checkpoint. Returns (step, train_losses, val_losses)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  📂 Resumed from {path} at step {ckpt['step']}")
    return ckpt["step"], ckpt.get("train_losses", []), ckpt.get("val_losses", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(resume: str | None = None):
    """Main training function."""

    # ── Configuration ─────────────────────────────────────────────────────
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Device setup
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = load_tokenizer()
    # Ensure model vocab matches tokenizer
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(tokenizer, train_cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Initializing Model")
    print("=" * 60)
    model = SmallGPT(model_cfg).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # AdamW (Adam with decoupled weight decay) is the standard optimizer
    # for transformer training.  We separate parameters into two groups:
    #   1. Parameters that should have weight decay (linear weights)
    #   2. Parameters that should NOT (biases, LayerNorm, embeddings)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate, betas=(0.9, 0.95))

    print(f"  Decay params:    {sum(p.numel() for p in decay_params):,}")
    print(f"  No-decay params: {sum(p.numel() for p in no_decay_params):,}")

    # ── Resume from checkpoint? ───────────────────────────────────────────
    start_step = 0
    train_losses = []
    val_losses = []

    if resume:
        if resume == "latest":
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, "checkpoint_latest.pt")
        else:
            ckpt_path = resume
        if os.path.exists(ckpt_path):
            start_step, train_losses, val_losses = load_checkpoint(
                ckpt_path, model, optimizer, device
            )
        else:
            print(f"  ⚠️  Checkpoint not found: {ckpt_path}. Training from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)
    print(f"  Max steps:     {train_cfg.max_steps:,}")
    print(f"  Batch size:    {train_cfg.batch_size}")
    print(f"  Seq length:    {train_cfg.max_seq_len}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print()

    model.train()
    step = start_step
    running_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    # We iterate over the dataloader in an infinite loop, counting steps
    # (not epochs).  This is common practice for LLM training.
    data_iter = iter(train_loader)

    pbar = tqdm(range(start_step, train_cfg.max_steps), desc="Training", ncols=100)
    for step in pbar:
        # Get next batch (restart dataloader if exhausted)
        try:
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, target_ids = next(data_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # ── Forward pass ──────────────────────────────────────────────
        _, loss = model(input_ids, targets=target_ids)

        # ── Backward pass ─────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # ── Update learning rate ──────────────────────────────────────
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Optimizer step ────────────────────────────────────────────
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        running_loss += loss.item()

        if (step + 1) % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (
                (step + 1 - start_step) * train_cfg.batch_size * train_cfg.max_seq_len
            ) / elapsed
            train_losses.append({"step": step + 1, "loss": avg_loss})
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr:.2e}",
                tok_s=f"{tokens_per_sec:.0f}",
            )
            running_loss = 0.0

        # ── Validation ────────────────────────────────────────────────
        if (step + 1) % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append({"step": step + 1, "loss": val_loss})
            print(f"\n  📊 Step {step+1}: val_loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"     🏆 New best val loss!")
            model.train()

        # ── Checkpointing ─────────────────────────────────────────────
        if (step + 1) % train_cfg.save_interval == 0:
            save_checkpoint(model, optimizer, step + 1, train_losses, val_losses, train_cfg)

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, step + 1, train_losses, val_losses, train_cfg)

    # ── Save loss history as JSON for plotting ────────────────────────────
    history = {"train": train_losses, "val": val_losses}
    history_path = os.path.join(train_cfg.checkpoint_dir, "loss_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  📈 Loss history saved to {history_path}")

    total_time = time.time() - start_time
    print(f"\n  ✅ Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"     Best val loss: {best_val_loss:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmallGPT")
    parser.add_argument("--resume", type=str, default=None,
                        help='Path to checkpoint or "latest" to resume training')
    args = parser.parse_args()
    train(resume=args.resume)
