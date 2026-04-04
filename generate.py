#!/usr/bin/env python3
"""
generate.py — Text generation with the trained SmallGPT model.

This script implements autoregressive text generation with several
sampling strategies:

  1. **Greedy decoding**: Always pick the most probable next token.
     Fast but repetitive — the model tends to loop.

  2. **Temperature sampling**: Scale the logits by a temperature parameter
     before applying softmax.
       - temperature < 1.0 → more confident / less random
       - temperature = 1.0 → standard sampling
       - temperature > 1.0 → more creative / more random

  3. **Top-k sampling**: Only consider the top k most probable tokens,
     then sample from them.  This prevents rare/weird tokens from being
     selected.

  4. **Top-p (nucleus) sampling**: Instead of a fixed k, include tokens
     from the top of the distribution until their cumulative probability
     exceeds p.  This adapts dynamically — when the model is confident,
     fewer tokens are considered; when uncertain, more are included.

In practice, top-p with a small temperature is the most popular strategy
for open-ended generation.

Usage:
    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "The little dog" --temperature 0.5 --top_k 40
    python generate.py --interactive  # Interactive chat mode
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from config import ModelConfig, GenerationConfig, CHECKPOINT_DIR
from model import SmallGPT
from train_tokenizer import load_tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling Functions
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k filtering: set all logits outside the top-k to -infinity.

    This ensures that only the k most probable tokens have a non-zero
    probability after softmax.

    Args:
        logits: (vocab_size,) — raw model output for one position
        k:      Number of top tokens to keep
    """
    if k <= 0 or k >= logits.size(-1):
        return logits  # No filtering

    # Find the k-th largest value
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[..., -1]  # The k-th largest value

    # Set everything below the threshold to -inf
    logits[logits < threshold] = float("-inf")
    return logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (nucleus) filtering: keep the smallest set of tokens whose
    cumulative probability exceeds p.

    This is more adaptive than top-k:
      - When the model is confident (one token has high prob), only a
        few tokens pass the filter.
      - When the model is uncertain (flat distribution), many tokens
        pass the filter.

    Args:
        logits: (vocab_size,) — raw model output for one position
        p:      Cumulative probability threshold (e.g., 0.9)
    """
    if p >= 1.0:
        return logits  # No filtering

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative probability first exceeds p
    # We shift right by 1 so that the first token above the threshold is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p

    # Set filtered tokens to -inf
    sorted_logits[sorted_mask] = float("-inf")

    # Unsort to get back to original order
    logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)
    return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Generation Function
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model: SmallGPT,
    tokenizer,
    prompt: str,
    gen_cfg: GenerationConfig | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Generate text from a prompt using the trained model.

    The generation process is autoregressive:
      1. Encode the prompt into token IDs.
      2. Feed the tokens into the model to get logits for the next token.
      3. Sample a token from the logits using the chosen strategy.
      4. Append the new token and repeat from step 2.
      5. Stop when we hit <EOS> or reach max_new_tokens.

    Args:
        model:     The trained SmallGPT model
        tokenizer: The tokenizer instance
        prompt:    Input text to continue from
        gen_cfg:   Generation configuration
        device:    Torch device

    Returns:
        The generated text (including the prompt)
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ── Encode the prompt ─────────────────────────────────────────────────
    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids

    # Get special token IDs
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    max_seq_len = model.cfg.max_seq_len

    # Remove trailing <EOS> from the prompt encoding — we don't want the
    # model to think the sequence is already finished.  <BOS> at the start
    # is fine (it signals beginning of text).
    if prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]

    # ── Autoregressive generation loop ────────────────────────────────────
    generated_ids = prompt_ids

    for _ in range(gen_cfg.max_new_tokens):
        # Truncate to max_seq_len if needed (sliding window)
        if len(generated_ids) > max_seq_len:
            context_ids = generated_ids[-max_seq_len:]
        else:
            context_ids = generated_ids

        # Forward pass to get logits
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits, _ = model(x)

        # We only care about the logits for the LAST position
        # (the prediction for the next token)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # ── Apply sampling strategy ───────────────────────────────────
        if gen_cfg.greedy:
            # Greedy: always pick the most probable token
            next_token = next_token_logits.argmax().item()
        else:
            # Temperature scaling
            if gen_cfg.temperature != 1.0:
                next_token_logits = next_token_logits / gen_cfg.temperature

            # Top-k filtering
            if gen_cfg.top_k > 0:
                next_token_logits = top_k_filtering(next_token_logits, gen_cfg.top_k)

            # Top-p filtering
            if gen_cfg.top_p < 1.0:
                next_token_logits = top_p_filtering(next_token_logits, gen_cfg.top_p)

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        # Append the generated token
        generated_ids.append(next_token)

        # Stop if we hit <EOS>
        if next_token == eos_id:
            break

    # ── Decode back to text ───────────────────────────────────────────────
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ═══════════════════════════════════════════════════════════════════════════════
# Load Model Helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str | None = None, device: str = "auto") -> tuple[SmallGPT, object]:
    """
    Load a trained model and tokenizer.

    Args:
        checkpoint_path: Path to checkpoint file, or None for latest
        device: Device string ("auto", "cuda", "cpu")

    Returns:
        (model, tokenizer)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Initialize model
    model_cfg = ModelConfig()
    model_cfg.vocab_size = tokenizer.get_vocab_size()
    model = SmallGPT(model_cfg)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print("   Using untrained model (output will be random)")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "?")
        print(f"✅ Loaded checkpoint from step {step}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate text with SmallGPT")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to continue from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p / nucleus sampling (1.0 to disable)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (ignores temperature/top_k/top_p)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type prompts and see generations")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    if args.interactive:
        # ── Interactive mode ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  SmallGPT Interactive Generation")
        print("  Type a prompt and press Enter. Type 'quit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("📝 Prompt: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not prompt:
                continue

            text = generate(model, tokenizer, prompt, gen_cfg, device)
            print(f"\n📖 Generated:\n{text}\n")
            print("-" * 60)
    else:
        # ── Single generation ─────────────────────────────────────────
        print(f"\n📝 Prompt: {args.prompt}")
        print(f"   Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        print()

        text = generate(model, tokenizer, args.prompt, gen_cfg, device)
        print(f"📖 Generated text:\n{'='*60}\n{text}\n{'='*60}")


if __name__ == "__main__":
    main()
