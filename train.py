#!/usr/bin/env python3
"""
train.py — Complete training pipeline for the small GPT language model.

=== WHAT IS THIS FILE? ===
This script trains a SmallGPT model from scratch on text data. Training means
adjusting the model's ~1.2 million parameters so that it becomes good at
predicting the next token in a sequence.

=== WHAT HAPPENS DURING TRAINING? ===
The training loop repeats these steps thousands of times:

    1. SAMPLE a batch of text sequences from the dataset
    2. FORWARD PASS: feed tokens through the model → get predicted logits
    3. COMPUTE LOSS: compare predictions to actual next tokens (cross-entropy)
    4. BACKWARD PASS: compute gradients (how much each parameter contributed to the loss)
    5. UPDATE: adjust parameters to reduce the loss (using AdamW optimizer)

    After enough iterations, the model learns patterns like:
      "Once upon a" → "time" (high probability)
      "The cat sat on the" → "mat" (high probability)

=== KEY CONCEPTS ===
  • **Learning rate schedule**: Start low (warmup), ramp up, then slowly decrease.
    This helps training stability and convergence.
  • **Gradient clipping**: Cap gradient magnitudes to prevent "exploding gradients"
    that could destabilize training.
  • **Weight decay**: Penalize large weights to prevent overfitting.
  • **Checkpointing**: Save the model periodically so you can resume training
    or evaluate at specific points.

=== USAGE ===
    python train.py                    # Train from scratch
    python train.py --resume latest    # Resume from latest checkpoint

=== TRAINING OBJECTIVE ===
**Causal language modelling** (next-token prediction):
    Input:  [The, cat, sat, on, the]
    Target: [cat, sat, on, the, mat]

    Loss = CrossEntropy(predicted_logits, true_next_tokens)

    We want to minimize this loss — lower loss means the model assigns
    higher probability to the correct next token.
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
    Compute the learning rate for a given training step.

    Uses the popular "warmup + cosine decay" schedule:

    LR
    ↑
    │         ╭────╮
    │        ╱      ╲         ← cosine decay
    │       ╱        ╲
    │      ╱          ╲
    │     ╱            ╲
    │    ╱              ╲___  ← minimum LR (10% of peak)
    │   ╱ ← warmup
    │──╱
    └───────────────────────→ Steps
      0    warmup    max_steps

    === PHASE 1: LINEAR WARMUP (steps 0 → warmup_steps) ===
    LR increases linearly from 0 to the peak learning_rate.

    WHY WARMUP?
    At the start of training, the model weights are random and gradients are
    very noisy. A high learning rate would cause wild, unstable updates. By
    starting small and gradually increasing, we give the optimizer time to
    "find its footing" before taking larger steps.

    Example with warmup_steps=50, learning_rate=5e-4:
      Step 0:  LR = 0
      Step 10: LR = 10/50 × 5e-4 = 1e-4
      Step 25: LR = 25/50 × 5e-4 = 2.5e-4
      Step 50: LR = 50/50 × 5e-4 = 5e-4  (peak!)

    === PHASE 2: COSINE DECAY (steps warmup_steps → max_steps) ===
    LR decreases following a cosine curve from peak to min_lr.

    WHY COSINE DECAY?
    As training progresses, the model gets closer to a good solution.
    Smaller learning rates help the model fine-tune its parameters without
    "overshooting" the optimum. The cosine shape provides a smooth, gradual
    reduction.

    The minimum LR is 10% of the peak (not zero) to prevent the model from
    completely stalling at the end of training.

    This schedule is used by GPT-3, LLaMA, and many modern LLMs.

    Args:
        step: Current training step (0-indexed).
        cfg: Training configuration with warmup_steps, max_steps, learning_rate.

    Returns:
        The learning rate for this step (float).
    """
    # Phase 1: Linear warmup
    if step < cfg.warmup_steps:
        # Linearly interpolate from 0 to learning_rate
        return cfg.learning_rate * (step / max(1, cfg.warmup_steps))

    # Phase 2: Cosine decay
    # Calculate how far we are through the decay phase (0.0 → 1.0)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)  # Clamp to [0, 1]

    # Cosine annealing formula: decays from learning_rate to min_lr
    # cos(0) = 1, cos(π) = -1, so (1 + cos(π * progress)) goes from 2 → 0
    min_lr = cfg.learning_rate * 0.1  # Floor at 10% of peak
    return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()  # Disable gradient computation (saves memory, speeds up)
def evaluate(model: nn.Module, val_loader, device: torch.device, max_batches: int = 50) -> float:
    """
    Run validation and return average loss.

    === WHAT IS VALIDATION? ===
    Validation measures how well the model performs on data it HASN'T seen
    during training. This helps us detect overfitting — when the model
    memorizes training data but can't generalize to new text.

    If train_loss decreases but val_loss increases, the model is overfitting.
    If both decrease together, the model is learning genuine patterns.

    === WHY max_batches? ===
    We limit to `max_batches` to keep validation fast. During training, we
    run validation every `eval_interval` steps, so it shouldn't take too long.
    50 batches × 16 batch_size = 800 sequences is enough for a reliable estimate.

    Args:
        model: The SmallGPT model to evaluate.
        val_loader: DataLoader providing validation data.
        device: The device (CPU or GPU) to run on.
        max_batches: Maximum number of batches to evaluate (for speed).

    Returns:
        Average cross-entropy loss on the validation set (float).
        Lower = better. Random model ≈ 8.3, trained ≈ 3-5.
    """
    model.eval()  # Switch to evaluation mode (disables dropout)
    total_loss = 0.0
    n_batches = 0

    for input_ids, target_ids in val_loader:
        if n_batches >= max_batches:
            break
        # Move data to the same device as the model (CPU or GPU)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass only — no backward pass during validation
        _, loss = model(input_ids, targets=target_ids)
        total_loss += loss.item()  # .item() extracts the Python number from the tensor
        n_batches += 1

    model.train()  # Switch back to training mode (re-enables dropout)
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
    """
    Save a training checkpoint to disk.

    === WHAT IS A CHECKPOINT? ===
    A checkpoint is a snapshot of the complete training state: model weights,
    optimizer state (momentum, learning rate state), and training history.
    With a checkpoint, you can:
      1. Resume training from where you left off
      2. Load the trained model for text generation
      3. Compare models at different training stages

    The checkpoint is saved as a PyTorch .pt file containing:
      - model_state_dict: all model weights/biases
      - optimizer_state_dict: Adam's internal states (momentum, variance estimates)
      - step: current training step number
      - train_losses: list of training loss records
      - val_losses: list of validation loss records

    We also create a "checkpoint_latest.pt" symlink that always points to the
    most recent checkpoint, making it easy to resume training.

    Args:
        model: The model to save.
        optimizer: The optimizer (contains momentum/variance state).
        step: Current training step.
        train_losses: History of training losses.
        val_losses: History of validation losses.
        cfg: Training configuration (for checkpoint directory path).
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, path)

    # Create/update the "latest" symlink for easy resuming
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
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        path: Path to the checkpoint .pt file.
        model: The model to load weights into.
        optimizer: The optimizer to load state into.
        device: Device to load tensors to (CPU or GPU).

    Returns:
        Tuple of (step, train_losses, val_losses):
          - step: The training step the checkpoint was saved at.
          - train_losses: History of training losses up to that step.
          - val_losses: History of validation losses up to that step.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  📂 Resumed from {path} at step {ckpt['step']}")
    return ckpt["step"], ckpt.get("train_losses", []), ckpt.get("val_losses", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(resume: str | None = None):
    """
    Main training function — orchestrates the entire training pipeline.

    This function:
      1. Sets up configuration, tokenizer, data, model, and optimizer
      2. Optionally resumes from a checkpoint
      3. Runs the training loop for max_steps iterations
      4. Saves checkpoints and logs periodically
      5. Saves the final model and loss history

    Args:
        resume: Path to a checkpoint file, or "latest" to resume from the
                most recent checkpoint, or None to train from scratch.
    """

    # ── Configuration ─────────────────────────────────────────────────────
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # ── Device Setup ──────────────────────────────────────────────────────
    # Modern deep learning can run on CPU or GPU. GPUs are much faster
    # for matrix operations (the core of neural networks), but our model
    # is small enough to train on CPU in ~1-2 minutes.
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    # Load the BPE tokenizer we trained earlier. The model's vocabulary
    # must match the tokenizer's vocabulary.
    tokenizer = load_tokenizer()
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    # ── Data ──────────────────────────────────────────────────────────────
    # Create PyTorch DataLoaders that serve batches of (input, target) pairs.
    # The DataLoader handles: batching, shuffling, parallel loading.
    train_loader, val_loader = create_dataloaders(tokenizer, train_cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Initializing Model")
    print("=" * 60)
    model = SmallGPT(model_cfg).to(device)
    # .to(device) moves all model parameters to the target device (CPU/GPU).
    # All subsequent operations must use data on the same device.

    # ── Optimizer ─────────────────────────────────────────────────────────
    # === WHAT IS AN OPTIMIZER? ===
    # The optimizer adjusts model parameters to minimize the loss.
    # Given the gradient ∂loss/∂param for each parameter, the optimizer
    # decides HOW to update that parameter.
    #
    # === WHY AdamW? ===
    # AdamW (Adam with decoupled Weight decay) is the standard optimizer
    # for transformers. It combines:
    #   • Momentum: uses a running average of past gradients to smooth updates
    #   • Adaptive learning rates: scales updates by the inverse of gradient
    #     variance — parameters with noisy gradients get smaller updates
    #   • Weight decay: penalizes large weights to prevent overfitting
    #
    # === PARAMETER GROUPS ===
    # Not all parameters should have weight decay:
    #   • Weights in Linear layers: YES — regularize to prevent overfitting
    #   • Biases, LayerNorm params, Embeddings: NO — these are already
    #     constrained by their nature and decay would hurt performance
    #
    # We separate parameters into two groups with different weight_decay settings.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D params (biases, LayerNorm γ/β) and LayerNorm layers: no decay
        # 2D+ params (Linear weights, Embedding weights): apply decay
        if param.dim() < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_cfg.learning_rate, betas=(0.9, 0.95))
    # betas=(0.9, 0.95): controls momentum decay rates
    #   beta1=0.9: exponential decay for first moment (gradient mean)
    #   beta2=0.95: exponential decay for second moment (gradient variance)
    #   Lower beta2 (vs default 0.999) is standard for LLM training —
    #   it makes the optimizer more responsive to recent gradient magnitudes.

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

    model.train()  # Enable training mode (activates dropout)
    step = start_step
    running_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    # === STEP-BASED VS EPOCH-BASED TRAINING ===
    # Traditional ML trains for N "epochs" (full passes through the data).
    # LLM training counts "steps" (gradient updates) instead, because:
    #   1. Datasets are huge — one epoch could take days
    #   2. We want fine-grained control over training duration
    #   3. Learning rate schedules are defined over steps
    #
    # We iterate over the dataloader in an infinite loop, just counting steps.
    # When the dataloader runs out, we restart it (a new "epoch").
    data_iter = iter(train_loader)

    pbar = tqdm(range(start_step, train_cfg.max_steps), desc="Training", ncols=100)
    for step in pbar:
        # ── Get next batch ─────────────────────────────────────────────
        # Each batch contains:
        #   input_ids:  (batch_size, max_seq_len) — input token sequences
        #   target_ids: (batch_size, max_seq_len) — shifted by 1 position
        try:
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            # DataLoader exhausted — restart (beginning of new "epoch")
            data_iter = iter(train_loader)
            input_ids, target_ids = next(data_iter)

        # Move data to the same device as the model
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # ── Forward pass ──────────────────────────────────────────────
        # Feed tokens through the model and compute the loss.
        # The loss measures how well the model predicts the next token.
        #
        # Example: if input = [The, cat, sat] and target = [cat, sat, on]
        # The model predicts a probability distribution for each position:
        #   Position 0: P(cat|The) should be high
        #   Position 1: P(sat|The,cat) should be high
        #   Position 2: P(on|The,cat,sat) should be high
        # Loss = average cross-entropy across all positions and all batches
        _, loss = model(input_ids, targets=target_ids)

        # ── Backward pass ─────────────────────────────────────────────
        # This is where the magic of deep learning happens!
        #
        # 1. optimizer.zero_grad(): Clear old gradients from the previous step.
        #    (PyTorch accumulates gradients by default — we don't want that here)
        optimizer.zero_grad()

        # 2. loss.backward(): Compute gradients using backpropagation.
        #    For each parameter θ in the model, this computes ∂loss/∂θ.
        #    The gradient tells us: "if I increase θ by a tiny amount, how
        #    much does the loss change?" We want to move θ in the direction
        #    that DECREASES the loss.
        loss.backward()

        # 3. Gradient clipping: Cap the gradient magnitude to prevent
        #    "exploding gradients" — when gradients become huge, the model
        #    weights change too drastically in one step, destabilizing training.
        #
        #    max_norm=1.0 means: if the total gradient norm exceeds 1.0,
        #    scale all gradients down proportionally.
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # ── Update learning rate ──────────────────────────────────────
        # We manually set the learning rate each step according to our
        # warmup + cosine decay schedule.
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Optimizer step ────────────────────────────────────────────
        # 4. optimizer.step(): Update all parameters using the gradients.
        #    For AdamW, the update rule (simplified) is:
        #      θ_new = θ_old - lr × (momentum_adjusted_gradient + weight_decay × θ_old)
        #
        #    This moves each parameter in the direction that reduces the loss.
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        # Track the running average of the loss to smooth out noise.
        running_loss += loss.item()

        if (step + 1) % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            elapsed = time.time() - start_time
            # Calculate throughput: how many tokens are we processing per second?
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
        # Periodically check performance on held-out data to detect
        # overfitting (when train loss decreases but val loss increases).
        if (step + 1) % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append({"step": step + 1, "loss": val_loss})
            print(f"\n  📊 Step {step+1}: val_loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"     🏆 New best val loss!")
            model.train()  # Switch back to training mode

        # ── Checkpointing ─────────────────────────────────────────────
        # Save the model periodically so we don't lose progress if
        # training is interrupted.
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
