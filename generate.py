#!/usr/bin/env python3
"""
generate.py — Text generation with the trained SmallGPT model.

=== WHAT IS TEXT GENERATION? ===
After training the model to predict next tokens, we can use it to GENERATE
new text by predicting one token at a time and feeding it back as input.
This is called "autoregressive generation" because each output becomes
the input for the next step.

=== HOW IT WORKS (Step by Step) ===

    Prompt: "Once upon a"
    
    Step 1: Feed [Once, upon, a] → model predicts "time" (highest probability)
    Step 2: Feed [Once, upon, a, time] → model predicts "," 
    Step 3: Feed [Once, upon, a, time, ,] → model predicts "there"
    ... and so on until we hit <EOS> or max length.

=== THE PROBLEM WITH GREEDY DECODING ===
Always picking the most probable token (greedy) tends to produce:
  • Repetitive text ("the the the the...")
  • Generic, boring output
  • Loops where the model gets stuck repeating phrases

This is why we need SAMPLING STRATEGIES.

=== SAMPLING STRATEGIES ===

1. **Temperature Sampling** — Controls randomness
   Before softmax, divide logits by temperature T:
     P(token) = softmax(logits / T)
   
   - T < 1.0 → sharper distribution → more confident/deterministic
     Example: logits [2.0, 1.0, 0.5] / 0.5 = [4.0, 2.0, 1.0]
              softmax → [0.84, 0.11, 0.04]  (strongly favors first token)
   
   - T = 1.0 → standard distribution (no change)
     Example: softmax([2.0, 1.0, 0.5]) → [0.51, 0.19, 0.11]
   
   - T > 1.0 → flatter distribution → more creative/random
     Example: logits [2.0, 1.0, 0.5] / 2.0 = [1.0, 0.5, 0.25]
              softmax → [0.41, 0.25, 0.19]  (more evenly distributed)

2. **Top-k Sampling** — Only consider the top k most probable tokens
   If k=3, only the 3 most likely tokens are candidates.
   This prevents rare/nonsensical tokens from being selected.
   
   Example: vocab has 4000 tokens, top-k=50
   → only the 50 most likely tokens can be generated
   → "xyzzy" (probability 0.0001%) has zero chance

3. **Top-p (Nucleus) Sampling** — Adaptive token filtering
   Include the smallest set of tokens whose cumulative probability ≥ p.
   
   Example with p=0.9:
     Token probs (sorted): [0.5, 0.2, 0.15, 0.08, 0.04, 0.02, 0.01]
     Cumulative:           [0.5, 0.7, 0.85, 0.93, ...]
     Keep first 4 tokens (cumulative reaches 0.93 > 0.9)
   
   This adapts to the model's confidence:
   - Confident prediction (one token has 95% probability) → only 1-2 tokens
   - Uncertain prediction (flat distribution) → many tokens
   
   Top-p is generally preferred over top-k because of this adaptivity.

In practice, **top-p (0.9) + temperature (0.7-0.8)** is the most popular
combination for open-ended text generation (used by ChatGPT, etc.).

=== USAGE ===
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
    Top-k filtering: zero out all logits outside the top-k most probable tokens.

    This is done by setting logits below the k-th highest to -infinity,
    so that after softmax they become zero probability.

    === CONCRETE EXAMPLE ===
    Suppose vocab_size=10 and k=3:
    
    Before filtering:
      logits = [1.2, 0.5, 3.1, -0.3, 2.8, 0.1, -1.0, 0.7, 1.5, 0.3]
    
    Top 3 values: 3.1 (idx 2), 2.8 (idx 4), 1.5 (idx 8)
    Threshold: 1.5 (the 3rd highest value)
    
    After filtering:
      logits = [-inf, -inf, 3.1, -inf, 2.8, -inf, -inf, -inf, 1.5, -inf]
    
    After softmax:
      probs = [0, 0, 0.52, 0, 0.35, 0, 0, 0, 0.13, 0]
    
    Now only the top 3 tokens can be sampled!

    Args:
        logits: (vocab_size,) — raw model output scores for one position.
                Higher logit = model thinks this token is more likely.
        k:      Number of top tokens to keep. All others get -inf.
                k=0 or k>=vocab_size means no filtering.

    Returns:
        Modified logits tensor with non-top-k values set to -inf.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits  # No filtering needed

    # Find the k-th largest value as the threshold
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[..., -1]  # The k-th largest value

    # Set everything below the threshold to -inf
    # After softmax, -inf → 0 probability
    logits[logits < threshold] = float("-inf")
    return logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (nucleus) filtering: keep the smallest set of tokens whose
    cumulative probability exceeds p, and zero out the rest.

    === WHY TOP-P IS BETTER THAN TOP-K ===
    Top-k always keeps exactly k tokens, regardless of the distribution:
      • Confident: [0.95, 0.03, 0.01, 0.005, ...] — k=50 includes many garbage tokens
      • Uncertain: [0.05, 0.04, 0.04, 0.03, ...]  — k=50 might miss valid tokens
    
    Top-p adapts to the distribution:
      • Confident: keeps ~1-3 tokens (cumulative quickly reaches p=0.9)
      • Uncertain: keeps many tokens (needs more to reach p=0.9)

    === STEP-BY-STEP EXAMPLE ===
    Suppose p=0.9 and we have 8 tokens:
    
    1. Softmax the logits to get probabilities:
       probs = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01]
       (already sorted descending for clarity)
    
    2. Compute cumulative probabilities:
       cumulative = [0.40, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99, 1.00]
    
    3. Find where cumulative first exceeds p=0.9:
       Index 3 has cumulative = 0.90 ≥ 0.9
       Keep tokens at indices 0, 1, 2, 3 (the first 4 tokens)
    
    4. Set all other logits to -inf:
       tokens 4, 5, 6, 7 get zeroed out after softmax.
    
    Result: only the top 4 tokens (covering 90% of probability mass) can be sampled.

    Args:
        logits: (vocab_size,) — raw model output scores for one position.
        p:      Cumulative probability threshold (e.g., 0.9).
                p=1.0 means no filtering. Lower p = fewer tokens kept.

    Returns:
        Modified logits tensor with low-probability tokens set to -inf.
    """
    if p >= 1.0:
        return logits  # No filtering

    # Sort logits in descending order (highest probability first)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Compute cumulative probabilities of the sorted tokens
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Create a mask for tokens to REMOVE:
    # We want to keep all tokens whose cumulative probability (EXCLUDING
    # the current token) is below p. This ensures the current token is
    # the one that pushes us over the threshold (so it gets kept).
    #
    # "cumulative_probs - current_prob >= p" means: even without this token,
    # we already have enough probability mass → remove this token.
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p

    # Set filtered tokens to -inf (they'll become 0 after softmax)
    sorted_logits[sorted_mask] = float("-inf")

    # Unsort: put the logits back in their original vocabulary order
    # (so that token IDs correspond to the right logit values)
    logits = sorted_logits.scatter(-1, sorted_indices.argsort(-1), sorted_logits)
    return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Generation Function
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()  # Disable gradients — we're not training, just generating
def generate(
    model: SmallGPT,
    tokenizer,
    prompt: str,
    gen_cfg: GenerationConfig | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Generate text from a prompt using the trained model.

    === THE AUTOREGRESSIVE GENERATION LOOP ===
    
    This is the core loop that makes language models generate text:
    
    1. Encode the prompt:      "Once upon a" → [2, 42, 100, 7]
    2. Feed into model:        model([2, 42, 100, 7]) → logits for position 4
    3. Sample next token:      logits → apply temperature/top-k/top-p → sample → 303
    4. Append:                 [2, 42, 100, 7, 303]
    5. Repeat from step 2:     model([2, 42, 100, 7, 303]) → logits for position 5
    6. Stop when:              we generate <EOS> or reach max_new_tokens
    7. Decode back to text:    [2, 42, 100, 7, 303, ...] → "Once upon a time, ..."

    === SLIDING WINDOW ===
    Our model has a maximum context length (max_seq_len=128). If the generated
    sequence exceeds this, we keep only the last 128 tokens as context.
    This means the model "forgets" the very beginning of long sequences.
    (Modern LLMs handle this with techniques like RoPE and longer contexts.)

    Args:
        model:     The trained SmallGPT model (in eval mode).
        tokenizer: The BPE tokenizer for encoding/decoding text.
        prompt:    Input text to continue from (e.g., "Once upon a time").
        gen_cfg:   Generation configuration (temperature, top_k, top_p, etc.).
                   If None, uses default GenerationConfig().
        device:    Torch device (CPU/GPU). If None, uses model's device.

    Returns:
        The complete generated text as a string (including the original prompt).

    Example:
        >>> text = generate(model, tokenizer, "The cat")
        >>> print(text)
        "The cat sat on the mat and looked at the bird..."
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()
    if device is None:
        device = next(model.parameters()).device

    model.eval()  # Disable dropout (we want deterministic behavior during generation)

    # ── Encode the prompt ─────────────────────────────────────────────────
    # Convert text to token IDs using the BPE tokenizer.
    # The tokenizer adds <BOS> at the start and <EOS> at the end.
    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids
    # Example: "Once upon a" → [2, 42, 100, 7, 3]
    #          where 2=<BOS>, 3=<EOS>

    # Get special token IDs for later comparison
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    max_seq_len = model.cfg.max_seq_len

    # Remove trailing <EOS> from the prompt encoding.
    # The tokenizer automatically adds <EOS>, but for generation we DON'T
    # want it — if the model sees <EOS>, it thinks the text is finished
    # and won't generate anything useful.
    # <BOS> at the start is fine — it signals "beginning of text".
    if prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]
    # Now: [2, 42, 100, 7] — <BOS> + prompt tokens, ready for continuation

    # ── Autoregressive generation loop ────────────────────────────────────
    generated_ids = prompt_ids  # Start with the prompt tokens

    for i in range(gen_cfg.max_new_tokens):
        # === Context Window Management ===
        # If the sequence is longer than the model's max context, use a
        # sliding window — keep only the last max_seq_len tokens.
        # The model can only "see" this window of context.
        if len(generated_ids) > max_seq_len:
            context_ids = generated_ids[-max_seq_len:]
        else:
            context_ids = generated_ids

        # === Forward Pass ===
        # Convert to tensor and add batch dimension: [ids] → [[ids]]
        # shape: (1, context_len)
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        # logits shape: (1, context_len, vocab_size)
        # logits[0, -1, :] gives the prediction for the NEXT token
        # (the model predicts the next token at each position, but we only
        # care about the last position for generation)

        # Extract logits for the last position only
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # === Apply Sampling Strategy ===
        if gen_cfg.greedy:
            # GREEDY: simply pick the token with the highest logit.
            # Deterministic but often repetitive.
            next_token = next_token_logits.argmax().item()
        else:
            # SAMPLING: introduce randomness for more diverse output.
            
            # Step 1: Temperature scaling
            # Divide logits by temperature before softmax.
            # This controls the "confidence" of the distribution.
            if gen_cfg.temperature != 1.0:
                next_token_logits = next_token_logits / gen_cfg.temperature

            # Step 2: Top-k filtering
            # Zero out all but the top-k most likely tokens.
            if gen_cfg.top_k > 0:
                next_token_logits = top_k_filtering(next_token_logits, gen_cfg.top_k)

            # Step 3: Top-p (nucleus) filtering
            # Zero out tokens until cumulative probability exceeds p.
            if gen_cfg.top_p < 1.0:
                next_token_logits = top_p_filtering(next_token_logits, gen_cfg.top_p)

            # Step 4: Sample from the filtered distribution
            # Convert logits to probabilities via softmax, then randomly
            # sample one token according to those probabilities.
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            # torch.multinomial: randomly picks an index with probability
            # proportional to the values in `probs`.
            # Example: probs = [0.5, 0.3, 0.2] → might return 0, 1, or 2
            #          with probabilities 50%, 30%, 20% respectively.

        # Append the generated token to our sequence
        generated_ids.append(next_token)

        # Stop if we generated the end-of-sequence token
        if next_token == eos_id:
            break

    # ── Decode back to text ───────────────────────────────────────────────
    # Convert token IDs back to human-readable text.
    # skip_special_tokens=True removes <BOS>, <EOS>, <PAD> from the output.
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ═══════════════════════════════════════════════════════════════════════════════
# Load Model Helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str | None = None, device: str = "auto") -> tuple[SmallGPT, object]:
    """
    Load a trained model and tokenizer from a checkpoint.

    This is the standard way to load a model for generation or evaluation.
    It handles:
      1. Loading the tokenizer (from tokenizer_model/ directory)
      2. Creating a SmallGPT model with the right configuration
      3. Loading trained weights from a checkpoint file
      4. Moving the model to the right device (CPU/GPU)

    Args:
        checkpoint_path: Path to a checkpoint .pt file.
                         If None, loads the latest checkpoint.
        device: Device string: "auto" (picks GPU if available), "cuda", or "cpu".

    Returns:
        (model, tokenizer) tuple:
          - model: SmallGPT instance with trained weights, in eval mode.
          - tokenizer: The BPE tokenizer for encoding/decoding text.

    Example:
        >>> model, tokenizer = load_model()
        >>> text = generate(model, tokenizer, "Hello world")
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load the BPE tokenizer
    tokenizer = load_tokenizer()

    # Initialize model with matching vocabulary size
    model_cfg = ModelConfig()
    model_cfg.vocab_size = tokenizer.get_vocab_size()
    model = SmallGPT(model_cfg)

    # Load trained weights from checkpoint
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
    model.eval()  # Set to evaluation mode (disables dropout)
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Command-line interface for text generation.

    Supports single-shot generation and interactive mode.

    Examples:
        # Single generation with default settings
        python generate.py --prompt "Once upon a time"

        # More creative generation (higher temperature)
        python generate.py --prompt "The story begins" --temperature 1.2

        # More focused generation (lower temperature, fewer candidates)
        python generate.py --prompt "The answer is" --temperature 0.3 --top_k 10

        # Interactive mode — type prompts and see generations
        python generate.py --interactive
    """
    parser = argparse.ArgumentParser(description="Generate text with SmallGPT")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to continue from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=focused, 1.0=normal, 2.0=creative)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling: only consider top k tokens (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p / nucleus sampling: cumulative prob threshold (1.0=disabled)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (ignores temperature/top_k/top_p)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type prompts and see generations")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device

    # Create generation configuration from CLI arguments
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    if args.interactive:
        # ── Interactive mode ──────────────────────────────────────────
        # Loop: read prompt from user → generate text → print → repeat
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
