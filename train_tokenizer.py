#!/usr/bin/env python3
"""
train_tokenizer.py — Train a Byte-Pair Encoding (BPE) tokenizer on the dataset.

BPE is one of the most popular sub-word tokenization algorithms used in modern
language models (GPT-2, RoBERTa, etc.).  It works by:

  1. Starting with individual characters as the initial vocabulary.
  2. Iteratively merging the most frequent adjacent pair of tokens into a new
     token until the desired vocabulary size is reached.

This gives a good balance between character-level flexibility (can handle any
word, even unseen ones) and word-level efficiency (common words get single tokens).

We use the `tokenizers` library from Hugging Face, which implements BPE in Rust
for speed, but the concepts are the same as a pure-Python implementation.

Usage:
    python train_tokenizer.py
"""

import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from config import TokenizerConfig, DATA_DIR, TOKENIZER_DIR


def train_tokenizer(cfg: TokenizerConfig | None = None) -> Tokenizer:
    """Train a BPE tokenizer on the training data and save it."""
    if cfg is None:
        cfg = TokenizerConfig()

    print("=" * 60)
    print("  Training BPE Tokenizer")
    print("=" * 60)
    print(f"  Vocab size : {cfg.vocab_size}")
    print(f"  Min freq   : {cfg.min_frequency}")
    print(f"  Save path  : {cfg.save_path}")
    print()

    # ── Step 1: Initialize a blank BPE tokenizer ──────────────────────────
    # The BPE model starts empty; the trainer will learn merge rules from data.
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # ── Step 2: Set up the pre-tokenizer ──────────────────────────────────
    # Before BPE, we split text on whitespace so that BPE merges happen
    # *within* words, not across word boundaries.
    tokenizer.pre_tokenizer = Whitespace()

    # ── Step 3: Configure the BPE trainer ─────────────────────────────────
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=cfg.special_tokens,  # These get ids 0, 1, 2, 3
        show_progress=True,
    )

    # ── Step 4: Train on the dataset ──────────────────────────────────────
    train_file = str(DATA_DIR / "train.txt")
    assert os.path.exists(train_file), f"Training data not found at {train_file}"
    print(f"  Training on: {train_file}")
    tokenizer.train([train_file], trainer)

    # ── Step 5: Add post-processing to wrap sequences with <BOS> / <EOS> ─
    # This automatically prepends <BOS> and appends <EOS> to every encoded
    # sequence, which is standard practice for language models.
    bos_id = cfg.special_tokens.index("<BOS>")
    eos_id = cfg.special_tokens.index("<EOS>")
    tokenizer.post_processor = TemplateProcessing(
        single=f"<BOS>:0 $A:0 <EOS>:0",
        special_tokens=[
            ("<BOS>", bos_id),
            ("<EOS>", eos_id),
        ],
    )

    # ── Step 6: Enable padding ────────────────────────────────────────────
    pad_id = cfg.special_tokens.index("<PAD>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<PAD>")

    # ── Step 7: Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    tokenizer.save(cfg.save_path)
    print(f"\n  ✅ Tokenizer saved to {cfg.save_path}")
    print(f"  Final vocab size: {tokenizer.get_vocab_size()}")

    # ── Quick demo ────────────────────────────────────────────────────────
    demo_text = "Once upon a time, there was a little girl named Lily."
    encoded = tokenizer.encode(demo_text)
    print(f"\n  Demo encoding:")
    print(f"    Text   : {demo_text}")
    print(f"    Tokens : {encoded.tokens[:20]}...")
    print(f"    IDs    : {encoded.ids[:20]}...")
    decoded = tokenizer.decode(encoded.ids)
    print(f"    Decoded: {decoded}")

    return tokenizer


def load_tokenizer(path: str | None = None) -> Tokenizer:
    """Load a previously trained tokenizer from disk."""
    if path is None:
        path = TokenizerConfig().save_path
    assert os.path.exists(path), f"Tokenizer not found at {path}. Run train_tokenizer.py first."
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    train_tokenizer()
