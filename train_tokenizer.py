#!/usr/bin/env python3
"""
train_tokenizer.py — Train a Byte-Pair Encoding (BPE) tokenizer on the dataset.

=== WHAT IS TOKENIZATION? ===
Neural networks work with numbers, not text. Tokenization is the process of
converting text into a sequence of integers (token IDs) that the model can process.

    "Hello world!" → [42, 100, 7]  (token IDs)

The tokenizer also works in reverse:
    [42, 100, 7] → "Hello world!"  (decoding)

=== WHY NOT JUST USE CHARACTERS? ===
We could tokenize character-by-character: "Hello" → ['H', 'e', 'l', 'l', 'o']
But this has problems:
  • Sequences become very long (slow to process)
  • Each character carries very little meaning
  • The model has to learn to combine characters into words from scratch

=== WHY NOT JUST USE WORDS? ===
We could tokenize word-by-word: "Hello world" → ["Hello", "world"]
But this also has problems:
  • Vocabulary would be enormous (millions of unique words)
  • Can't handle typos, new words, or rare words ("OOV" problem)
  • Different forms of the same word ("run", "running", "ran") get separate tokens

=== BPE: THE BEST OF BOTH WORLDS ===
Byte-Pair Encoding (BPE) creates a vocabulary of SUB-WORD tokens:
  • Common words get single tokens: "the" → ["the"]
  • Rare words are split into pieces: "brontosaurus" → ["br", "onto", "saur", "us"]
  • Can handle ANY text (falls back to characters for unknown patterns)

How BPE builds its vocabulary:
  1. Start with all individual characters: {a, b, c, d, ...}
  2. Count all adjacent pairs in the training text
  3. Merge the most frequent pair into a new token
     Example: 'h' + 'e' → 'he' (appears very often)
  4. Repeat steps 2-3 until vocab reaches desired size
  5. Result: frequent character sequences become single tokens

After training, the vocabulary might look like:
  Characters: a, b, c, ..., z
  Common pairs: th, he, in, er, an, ...
  Common words: the, and, was, for, ...
  Sub-words: ing, tion, ment, ...

Example tokenization:
  "understanding" → ["under", "stand", "ing"]
  "the" → ["the"]
  "xylophone" → ["xy", "lo", "phone"]  (rare word split into pieces)

We use the `tokenizers` library from Hugging Face, which implements BPE in Rust
for speed, but the algorithm is the same as described above.

=== USAGE ===
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
    """
    Train a BPE tokenizer on the training data and save it to disk.

    This function:
      1. Creates a blank BPE tokenizer
      2. Configures pre-tokenization (split on whitespace)
      3. Trains BPE merges on the training text
      4. Adds post-processing (<BOS> and <EOS> wrapping)
      5. Enables padding (for batching variable-length sequences)
      6. Saves the trained tokenizer to disk

    Args:
        cfg: Tokenizer configuration. If None, uses default TokenizerConfig().
             Key settings: vocab_size=4000, min_frequency=2.

    Returns:
        The trained Tokenizer instance.

    Example:
        >>> tokenizer = train_tokenizer()
        >>> encoded = tokenizer.encode("Hello world")
        >>> print(encoded.tokens)  # ['<BOS>', 'Hello', 'world', '<EOS>']
        >>> print(encoded.ids)     # [2, 42, 100, 3]
    """
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
    # The BPE model starts with an empty vocabulary. The trainer (below) will
    # learn merge rules from the training data.
    # unk_token="<UNK>" tells the tokenizer what to output for characters or
    # sequences it hasn't seen during training.
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # ── Step 2: Set up the pre-tokenizer ──────────────────────────────────
    # BEFORE applying BPE, we split text on whitespace. This ensures that
    # BPE merges happen WITHIN words, not across word boundaries.
    #
    # Without pre-tokenization: "the cat" might merge into "thec" "at"
    # With whitespace pre-tokenization: "the" and "cat" are processed separately
    tokenizer.pre_tokenizer = Whitespace()

    # ── Step 3: Configure the BPE trainer ─────────────────────────────────
    # The trainer controls the BPE learning process:
    #   • vocab_size: stop merging when vocabulary reaches this size
    #   • min_frequency: only merge pairs that appear at least this many times
    #   • special_tokens: reserve these tokens at the start of the vocabulary
    #     They get IDs 0, 1, 2, 3 (in order), which we reference elsewhere.
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=cfg.special_tokens,  # ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        show_progress=True,
    )

    # ── Step 4: Train on the dataset ──────────────────────────────────────
    # Feed the training text file to the BPE algorithm.
    # It will:
    #   1. Read all text and split into characters
    #   2. Count all adjacent character pairs
    #   3. Merge the most frequent pair
    #   4. Repeat until vocab_size is reached
    train_file = str(DATA_DIR / "train.txt")
    assert os.path.exists(train_file), f"Training data not found at {train_file}"
    print(f"  Training on: {train_file}")
    tokenizer.train([train_file], trainer)

    # ── Step 5: Add post-processing ───────────────────────────────────────
    # Automatically wrap every encoded sequence with <BOS> and <EOS>:
    #   "Hello world" → [<BOS>, Hello, world, <EOS>]
    #
    # This is standard for language models:
    #   <BOS> tells the model "this is the start of a new text"
    #   <EOS> tells the model "this text is finished"
    bos_id = cfg.special_tokens.index("<BOS>")  # = 2
    eos_id = cfg.special_tokens.index("<EOS>")  # = 3
    tokenizer.post_processor = TemplateProcessing(
        single=f"<BOS>:0 $A:0 <EOS>:0",
        # This template means: prepend <BOS>, then the actual tokens ($A),
        # then append <EOS>. The ":0" specifies type_id=0 (for single sequence).
        special_tokens=[
            ("<BOS>", bos_id),
            ("<EOS>", eos_id),
        ],
    )

    # ── Step 6: Enable padding ────────────────────────────────────────────
    # When batching sequences of different lengths, we need to pad shorter
    # ones to the same length. This tells the tokenizer to use <PAD> (id=0)
    # for padding.
    #
    # Example batch (max_len=5):
    #   [<BOS>, The, cat, <EOS>, <PAD>]  ← padded
    #   [<BOS>, Hello, world, today, <EOS>]  ← full length
    pad_id = cfg.special_tokens.index("<PAD>")  # = 0
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<PAD>")

    # ── Step 7: Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    tokenizer.save(cfg.save_path)
    print(f"\n  ✅ Tokenizer saved to {cfg.save_path}")
    print(f"  Final vocab size: {tokenizer.get_vocab_size()}")

    # ── Quick demo ────────────────────────────────────────────────────────
    # Show how the trained tokenizer works on a sample sentence.
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
    """
    Load a previously trained tokenizer from disk.

    Args:
        path: Path to the tokenizer JSON file.
              If None, uses the default path from TokenizerConfig.

    Returns:
        A Tokenizer instance ready for encoding/decoding text.

    Raises:
        AssertionError: If the tokenizer file doesn't exist.
            Fix: run `python train_tokenizer.py` first.

    Example:
        >>> tokenizer = load_tokenizer()
        >>> encoded = tokenizer.encode("Hello world")
        >>> print(encoded.ids)  # [2, 42, 100, 3]
    """
    if path is None:
        path = TokenizerConfig().save_path
    assert os.path.exists(path), f"Tokenizer not found at {path}. Run train_tokenizer.py first."
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    train_tokenizer()
