#!/usr/bin/env python3
"""
dataset.py — Data loading and preprocessing for language model training.

=== WHAT IS THIS FILE? ===
This file handles the pipeline from raw text files to PyTorch tensors that
the model can consume during training. The key steps are:

  1. READ raw text from .txt files
  2. TOKENIZE the text into integer token IDs using BPE
  3. CHUNK the long token stream into fixed-length sequences
  4. CREATE (input, target) pairs for next-token prediction

=== THE NEXT-TOKEN PREDICTION SETUP ===
For causal language modeling, we need input-target pairs where the target
is the input shifted by one position:

  Raw tokens:  [The, cat, sat, on, the, mat]

  input  = [The, cat, sat, on, the]    ← what the model sees
  target = [cat, sat, on, the, mat]    ← what the model should predict

At each position, the model predicts the next token:
  Position 0: sees [The]           → should predict "cat"
  Position 1: sees [The, cat]      → should predict "sat"
  Position 2: sees [The, cat, sat] → should predict "on"
  ... and so on.

=== THE "PACKED" APPROACH ===
Instead of treating each sentence separately, we concatenate ALL text into
one long stream of tokens and slide a window over it. This is the standard
approach used by GPT-2, GPT-3, etc.

Advantages:
  • No wasted padding tokens (every position is useful)
  • The model sees cross-sentence context (learns paragraph flow)
  • Simple implementation

Disadvantage:
  • Sequences may start/end mid-sentence (not always clean boundaries)

=== EXAMPLE (with max_seq_len=4) ===
Suppose our text tokenizes to: [10, 20, 30, 40, 50, 60, 70, 80]

We extract overlapping chunks of length max_seq_len + 1 = 5:
  Chunk 0: [10, 20, 30, 40, 50] → input=[10,20,30,40] target=[20,30,40,50]
  Chunk 1: [20, 30, 40, 50, 60] → input=[20,30,40,50] target=[30,40,50,60]
  Chunk 2: [30, 40, 50, 60, 70] → input=[30,40,50,60] target=[40,50,60,70]
  Chunk 3: [40, 50, 60, 70, 80] → input=[40,50,60,70] target=[50,60,70,80]

Total: 4 training examples from 8 tokens.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from config import TrainConfig, TokenizerConfig


class TextDataset(Dataset):
    """
    A PyTorch Dataset that serves (input, target) pairs for language modeling.

    This dataset:
      1. Reads a text file into memory
      2. Tokenizes the entire text into a flat array of token IDs
      3. Returns fixed-length chunks as (input, target) pairs on demand

    The __getitem__ method is called by the DataLoader during training to
    fetch individual samples. The DataLoader then batches multiple samples
    together.

    === MEMORY USAGE ===
    The entire tokenized text is stored in memory as a 1D tensor. For our
    small dataset (~1MB text → ~300K tokens → ~1.2MB as int64 tensor), this
    is perfectly fine. For larger datasets (GB+), you'd need memory-mapping
    or streaming approaches.
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
    ):
        """
        Load and tokenize a text file.

        Args:
            text_path:   Path to the .txt file containing training/validation text.
            tokenizer:   A trained BPE tokenizer instance (from train_tokenizer.py).
            max_seq_len: Context window size in tokens. Each training example will
                         have this many tokens as input and this many as target.
                         Must match the model's max_seq_len.
                         Default: 256 (but our config uses 128).

        After initialization, self.token_ids is a 1D tensor of ALL token IDs
        from the file, e.g., tensor([2, 42, 100, 7, 303, 2, 55, 12, ...]).
        """
        super().__init__()
        self.max_seq_len = max_seq_len

        print(f"  Loading and tokenizing: {text_path}")

        # ── Step 1: Read the raw text ─────────────────────────────────────
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        # text is now one big string, e.g., "Once upon a time...\n\nThe cat..."

        # ── Step 2: Tokenize the entire text ──────────────────────────────
        # We process in chunks of 100K characters to avoid memory issues
        # with very large files. The BPE tokenizer converts each chunk to
        # token IDs, and we concatenate all IDs into one flat list.
        #
        # Note: We use the tokenizer as-is, including post-processing that
        # adds <BOS>/<EOS> around each chunk. For bulk training data, you
        # might want to disable this, but for our small dataset it's fine.
        chunk_size = 100_000  # characters per chunk
        all_ids = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            encoded = tokenizer.encode(chunk)
            all_ids.extend(encoded.ids)
            # encoded.ids is a list like [2, 42, 100, 7, 303, 3, ...]

        # Convert to a PyTorch tensor for efficient indexing
        self.token_ids = torch.tensor(all_ids, dtype=torch.long)
        # dtype=torch.long (int64) because token IDs are integers that will
        # be used for embedding lookup (nn.Embedding requires LongTensor).

        print(f"    Total tokens: {len(self.token_ids):,}")
        print(f"    Sequences:    {len(self):,}  (seq_len={max_seq_len})")

    def __len__(self) -> int:
        """
        Number of (input, target) pairs we can extract from the token stream.

        Each pair needs max_seq_len + 1 consecutive tokens (input is the first
        max_seq_len tokens, target is the last max_seq_len tokens, overlapping
        by max_seq_len - 1).

        Example: 1000 tokens, max_seq_len=128 → 1000 - 128 = 872 sequences.

        Returns:
            Number of available training sequences (int).
        """
        return max(0, len(self.token_ids) - self.max_seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair by index.

        This extracts a window of max_seq_len + 1 tokens starting at position idx,
        then splits into input (first max_seq_len) and target (last max_seq_len).

        Args:
            idx: Starting position in the token stream (0-indexed).

        Returns:
            input_ids:  (max_seq_len,) — the input token sequence
            target_ids: (max_seq_len,) — the target (shifted by 1 position)

        Example (max_seq_len=4, idx=0):
            token_ids = [10, 20, 30, 40, 50, 60, 70, 80]
            chunk     = token_ids[0:5] = [10, 20, 30, 40, 50]
            input_ids = [10, 20, 30, 40]   ← chunk[:-1]
            target_ids = [20, 30, 40, 50]  ← chunk[1:]

        The model should learn:
            Given [10]         → predict 20
            Given [10, 20]     → predict 30
            Given [10, 20, 30] → predict 40
            Given [10, 20, 30, 40] → predict 50
        """
        # Extract max_seq_len + 1 consecutive tokens
        chunk = self.token_ids[idx : idx + self.max_seq_len + 1]
        # Split into input and target (offset by 1)
        input_ids = chunk[:-1]   # [t_0 .. t_{n-1}]
        target_ids = chunk[1:]   # [t_1 .. t_n]
        return input_ids, target_ids


def create_dataloaders(
    tokenizer: Tokenizer,
    train_cfg: TrainConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    === WHAT IS A DATALOADER? ===
    A DataLoader wraps a Dataset and provides:
      • Batching: groups multiple samples into a batch tensor
      • Shuffling: randomizes the order of samples each epoch (training only)
      • Parallel loading: uses multiple CPU workers to prepare data while
        the GPU is busy training
      • Memory pinning: pre-copies data to GPU-accessible memory for faster
        GPU transfers

    Args:
        tokenizer: Trained BPE tokenizer for encoding the text data.
        train_cfg: Training configuration (batch_size, data paths, etc.).
                   If None, uses default TrainConfig().

    Returns:
        (train_loader, val_loader) tuple of DataLoader instances.
        Each batch from the loader contains:
          input_ids:  (batch_size, max_seq_len) — input token sequences
          target_ids: (batch_size, max_seq_len) — shifted target sequences

    Example:
        >>> train_loader, val_loader = create_dataloaders(tokenizer)
        >>> for input_ids, target_ids in train_loader:
        ...     print(input_ids.shape)   # (16, 128)
        ...     print(target_ids.shape)  # (16, 128)
        ...     break
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    print("\n" + "=" * 60)
    print("  Creating Datasets & DataLoaders")
    print("=" * 60)

    # Create Dataset objects (tokenize the text files)
    train_dataset = TextDataset(
        train_cfg.train_path,
        tokenizer,
        train_cfg.max_seq_len,
    )
    val_dataset = TextDataset(
        train_cfg.val_path,
        tokenizer,
        train_cfg.max_seq_len,
    )

    # Wrap in DataLoaders for batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,           # Randomize order each epoch (important for training!
                                # Without shuffling, the model sees data in the same
                                # order every epoch, which can lead to overfitting.)
        num_workers=2,          # Use 2 CPU processes for parallel data loading
        pin_memory=True,        # Pre-copy data to GPU-accessible memory (faster)
        drop_last=True,         # Drop the last incomplete batch (ensures uniform
                                # batch sizes, which is important for some optimizations)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,          # Don't shuffle validation data (order doesn't matter
                                # for evaluation, and consistent order aids debugging)
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  Train batches per epoch: {len(train_loader):,}")
    print(f"  Val batches per epoch:   {len(val_loader):,}")

    return train_loader, val_loader


if __name__ == "__main__":
    # ── Quick test ────────────────────────────────────────────────────────
    # Run this file directly to verify the dataset works correctly.
    from train_tokenizer import load_tokenizer

    tok = load_tokenizer()
    train_loader, val_loader = create_dataloaders(tok)

    # Grab one batch and inspect it
    batch = next(iter(train_loader))
    input_ids, target_ids = batch
    print(f"\n  Batch input shape:  {input_ids.shape}   (batch_size, max_seq_len)")
    print(f"  Batch target shape: {target_ids.shape}  (batch_size, max_seq_len)")
    print(f"\n  Sample input[:10]:  {input_ids[0, :10].tolist()}")
    print(f"  Sample target[:10]: {target_ids[0, :10].tolist()}")
    print(f"\n  Notice: target is input shifted by 1 position!")
    print(f"    input[1:10]  = {input_ids[0, 1:10].tolist()}")
    print(f"    target[0:9]  = {target_ids[0, 0:9].tolist()}")
    print(f"    (These should be identical ↑)")
    print("  ✅ Dataset works!")
