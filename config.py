"""
config.py — Central configuration for the small LLM project.

All hyperparameters and paths are defined here so they can be easily
adjusted without touching the rest of the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Project Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_model"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# ── Tokenizer Config ──────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    vocab_size: int = 4000          # BPE vocabulary size (keep small for fast training)
    min_frequency: int = 2          # Minimum token frequency to be included
    special_tokens: list = field(default_factory=lambda: [
        "<PAD>",   # Padding token  (id=0)
        "<UNK>",   # Unknown token  (id=1)
        "<BOS>",   # Beginning of sequence (id=2)
        "<EOS>",   # End of sequence (id=3)
    ])
    save_path: str = str(TOKENIZER_DIR / "tokenizer.json")


# ── Model Config ───────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    A small GPT-style transformer decoder.

    Architecture overview:
      - Token embedding  + Positional embedding
      - N transformer decoder blocks, each containing:
            • Multi-Head Causal Self-Attention
            • Feed-Forward Network (FFN) with GELU activation
            • Layer Normalization (pre-norm style, like GPT-2)
            • Residual connections
      - Final LayerNorm → linear head → logits over vocabulary

    Total parameters ≈ 1.2 M  (tiny, trains in ~1 min on CPU)
    """
    vocab_size: int = 4000          # Must match tokenizer vocab_size
    max_seq_len: int = 128          # Maximum context window (in tokens)
    n_layers: int = 4               # Number of transformer blocks
    n_heads: int = 4                # Number of attention heads
    d_model: int = 128              # Embedding / hidden dimension
    d_ff: int = 512                 # Feed-forward inner dimension (4 × d_model)
    dropout: float = 0.1            # Dropout rate
    pad_token_id: int = 0           # Must match <PAD> id from tokenizer


# ── Training Config ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Data
    train_path: str = str(DATA_DIR / "train.txt")
    val_path: str = str(DATA_DIR / "val.txt")
    max_seq_len: int = 128          # Should match ModelConfig.max_seq_len

    # Optimization
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 500            # Total training steps (~1-2 min on CPU)
    grad_clip: float = 1.0          # Gradient clipping max-norm

    # Logging & checkpointing
    log_interval: int = 25          # Print loss every N steps
    eval_interval: int = 100        # Run validation every N steps
    save_interval: int = 250        # Save checkpoint every N steps
    checkpoint_dir: str = str(CHECKPOINT_DIR)

    # Device
    device: str = "auto"            # "auto" picks cuda if available, else cpu


# ── Generation Config ──────────────────────────────────────────────────────────
@dataclass
class GenerationConfig:
    max_new_tokens: int = 200       # Maximum tokens to generate
    temperature: float = 0.8        # Sampling temperature (1.0 = neutral)
    top_k: int = 50                 # Top-k filtering (0 = disabled)
    top_p: float = 0.9              # Nucleus / top-p filtering (1.0 = disabled)
    greedy: bool = False            # If True, always pick most-likely token
