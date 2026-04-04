#!/usr/bin/env python3
"""
config.py — Central configuration for the small LLM project.

=== WHAT IS THIS FILE? ===
All hyperparameters and paths for the entire project are defined here in one
place. This makes it easy to experiment: just change a number here and it
propagates to all scripts (model, training, tokenizer, generation).

=== WHY USE DATACLASSES? ===
Python dataclasses give us structured configuration objects with:
  • Type hints (so you know what each setting expects)
  • Default values (sensible defaults that work out of the box)
  • Easy modification (just change a field)
  • String representation (print the config to see all settings)

=== HYPERPARAMETER RELATIONSHIPS ===
Some settings are connected and should be changed together:
  • vocab_size in TokenizerConfig and ModelConfig must match
  • max_seq_len in ModelConfig and TrainConfig must match
  • d_ff is typically 4 × d_model
  • d_model must be divisible by n_heads
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Project Paths ──────────────────────────────────────────────────────────────
# These are computed relative to this file's location, so the project works
# regardless of where you clone it.
PROJECT_ROOT = Path(__file__).resolve().parent   # e.g., /home/user/small_llm_from_scratch
DATA_DIR = PROJECT_ROOT / "data"                  # Raw text data (.txt files)
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_model"  # Saved BPE tokenizer
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"     # Model checkpoints and logs


# ── Tokenizer Config ──────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    """
    Configuration for the BPE (Byte-Pair Encoding) tokenizer.

    The tokenizer converts raw text into integer token IDs that the model
    can process. BPE creates a vocabulary of sub-word tokens by iteratively
    merging the most frequent character pairs.

    Key trade-off: vocab_size
      • Larger vocab → common words get single tokens (efficient) but
        embedding table is bigger (more parameters, slower to train)
      • Smaller vocab → more sub-word splitting (longer sequences) but
        fewer parameters and faster training
      
    For our educational model, 4000 tokens is a good balance.
    (GPT-2 uses 50,257 tokens; LLaMA uses 32,000)
    """
    vocab_size: int = 4000          # BPE vocabulary size (keep small for fast training)
    min_frequency: int = 2          # A token pair must appear at least this many times
                                    # to be merged. Higher = smaller, more common vocab.
    special_tokens: list = field(default_factory=lambda: [
        "<PAD>",   # Padding token  (id=0) — fills shorter sequences to equal length
        "<UNK>",   # Unknown token  (id=1) — replaces tokens not in vocabulary
        "<BOS>",   # Beginning of sequence (id=2) — signals start of text
        "<EOS>",   # End of sequence (id=3) — signals end of text
    ])
    save_path: str = str(TOKENIZER_DIR / "tokenizer.json")


# ── Model Config ───────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    Configuration for the SmallGPT transformer decoder model.

    === ARCHITECTURE OVERVIEW ===
    Token embedding + Positional embedding
    → N transformer decoder blocks, each containing:
        • Multi-Head Causal Self-Attention (n_heads parallel attention computations)
        • Feed-Forward Network (2-layer MLP with GELU activation)
        • Layer Normalization (pre-norm style, like GPT-2)
        • Residual connections (skip connections for gradient flow)
    → Final LayerNorm → Linear head → logits over vocabulary

    === PARAMETER BUDGET ===
    With the default settings below, the model has ~1.2M parameters:
      • Token embeddings: vocab_size × d_model = 4000 × 128 = 512K
      • Position embeddings: max_seq_len × d_model = 128 × 128 = 16K
      • Per transformer block: ~200K
        - Attention QKV projection: d_model × 3 × d_model = 128 × 384 = 49K
        - Attention output projection: d_model × d_model = 128 × 128 = 16K
        - FFN fc1: d_model × d_ff = 128 × 512 = 66K
        - FFN fc2: d_ff × d_model = 512 × 128 = 66K
        - LayerNorm params: 2 × d_model × 2 = 512
      • 4 blocks: ~800K
      • LM head: tied with token embeddings (no extra params)
      • Total: ~1.2M (tiny! trains in ~1 minute on CPU)

    === SCALING GUIDE ===
    To make the model more capable (but slower to train):
      • d_model: 128 → 256 → 512 → 768 (GPT-2 small)
      • n_layers: 4 → 6 → 12 (GPT-2 small has 12)
      • n_heads: 4 → 8 → 12
      • d_ff: typically 4 × d_model
    Rule of thumb: doubling d_model roughly 4× the parameters.
    """
    vocab_size: int = 4000          # Must match tokenizer vocab_size
    max_seq_len: int = 128          # Maximum context window (in tokens)
                                    # The model can only "see" this many tokens at once.
                                    # GPT-2: 1024, GPT-3: 2048, GPT-4: 8K-128K
    n_layers: int = 4               # Number of stacked transformer blocks
                                    # More layers = deeper understanding but slower
    n_heads: int = 4                # Number of parallel attention heads
                                    # More heads = more diverse attention patterns
                                    # d_model must be divisible by n_heads
    d_model: int = 128              # Embedding / hidden dimension
                                    # This is the "width" of the model — how many
                                    # numbers represent each token. Larger = more
                                    # expressive but more parameters.
    d_ff: int = 512                 # Feed-forward inner dimension (4 × d_model)
                                    # The FFN expands to this size, applies GELU,
                                    # then compresses back to d_model.
    dropout: float = 0.1            # Dropout rate (fraction of neurons randomly
                                    # disabled during training for regularization).
                                    # 0.1 = 10% dropout. Set to 0 for generation.
    pad_token_id: int = 0           # Must match <PAD> id from tokenizer


# ── Training Config ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """
    Configuration for the training pipeline.

    === KEY HYPERPARAMETERS EXPLAINED ===

    batch_size (16):
      Number of sequences processed simultaneously in each training step.
      Larger batch = more stable gradients but needs more memory.
      Each step processes: batch_size × max_seq_len = 16 × 128 = 2048 tokens.

    learning_rate (5e-4 = 0.0005):
      Controls how much weights change per update. Too high = unstable,
      too low = slow learning. 5e-4 is a common starting point for small LLMs.
      (GPT-3 used 6e-5 for its largest model.)

    weight_decay (0.01):
      L2 regularization strength. Penalizes large weights to prevent
      overfitting. 0.01 is standard for transformer training.

    warmup_steps (50):
      Number of steps for learning rate warmup. During warmup, LR increases
      linearly from 0 to the peak. This stabilizes early training.

    max_steps (500):
      Total number of gradient updates. With batch_size=16, max_seq_len=128:
      Total tokens seen = 500 × 16 × 128 = 1,024,000 tokens.
      This is enough to learn basic patterns but not for high quality output.
      Increase to 2000-5000 for better results.

    grad_clip (1.0):
      Maximum gradient norm. If gradients exceed this, they're scaled down.
      Prevents "exploding gradients" that can destabilize training.
    """
    # Data paths
    train_path: str = str(DATA_DIR / "train.txt")
    val_path: str = str(DATA_DIR / "val.txt")
    max_seq_len: int = 128          # Should match ModelConfig.max_seq_len

    # Optimization hyperparameters
    batch_size: int = 16            # Sequences per training step
    learning_rate: float = 5e-4     # Peak learning rate (after warmup)
    weight_decay: float = 0.01      # L2 regularization strength
    warmup_steps: int = 50          # Steps for LR warmup (0 → peak)
    max_steps: int = 500            # Total training steps (~1-2 min on CPU)
    grad_clip: float = 1.0          # Max gradient norm for clipping

    # Logging & checkpointing
    log_interval: int = 25          # Print loss every N steps
    eval_interval: int = 100        # Run validation every N steps
    save_interval: int = 250        # Save checkpoint every N steps
    checkpoint_dir: str = str(CHECKPOINT_DIR)

    # Device selection
    device: str = "auto"            # "auto" picks cuda if available, else cpu


# ── Generation Config ──────────────────────────────────────────────────────────
@dataclass
class GenerationConfig:
    """
    Configuration for text generation (inference time).

    === SAMPLING STRATEGY CHEAT SHEET ===

    For creative writing (stories, poetry):
      temperature=0.8-1.0, top_k=50, top_p=0.9

    For more focused/factual output:
      temperature=0.3-0.5, top_k=10, top_p=0.8

    For deterministic/reproducible output:
      greedy=True (ignores all other sampling settings)

    For maximum creativity (may be incoherent):
      temperature=1.5, top_k=0, top_p=1.0
    """
    max_new_tokens: int = 200       # Maximum tokens to generate after the prompt
    temperature: float = 0.8        # Sampling temperature:
                                    #   <1.0 = more focused/confident
                                    #   =1.0 = standard (no scaling)
                                    #   >1.0 = more creative/random
    top_k: int = 50                 # Top-k filtering: keep top k tokens (0=disabled)
    top_p: float = 0.9              # Nucleus / top-p filtering: cumulative probability
                                    # threshold (1.0=disabled, 0.9=common)
    greedy: bool = False            # If True, always pick the highest-probability token
                                    # (deterministic, but often repetitive)
