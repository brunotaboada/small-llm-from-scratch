#!/usr/bin/env python3
"""
dataset.py — Data loading and preprocessing for language model training.

For language modelling, we need to:
  1. Read raw text from files.
  2. Tokenize the text into integer token IDs.
  3. Chunk the token stream into fixed-length sequences.
  4. Create (input, target) pairs where target = input shifted by 1 position.

The standard approach for causal LM training:
  - input  = [t_0, t_1, t_2, ..., t_{n-1}]
  - target = [t_1, t_2, t_3, ..., t_n]
  The model learns to predict each next token given all previous tokens.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from config import TrainConfig, TokenizerConfig


class TextDataset(Dataset):
    """
    A simple dataset that:
      1. Reads a text file
      2. Tokenizes the entire text into a flat array of token IDs
      3. Returns overlapping fixed-length chunks as (input, target) pairs

    This is the standard "packed" approach used by GPT-2 and similar models,
    where we concatenate all text and slide a window over it.
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
    ):
        """
        Args:
            text_path:   Path to the .txt file
            tokenizer:   A trained tokenizer instance
            max_seq_len: Context window size (in tokens)
        """
        super().__init__()
        self.max_seq_len = max_seq_len

        print(f"  Loading and tokenizing: {text_path}")

        # ── Step 1: Read the raw text ─────────────────────────────────────
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # ── Step 2: Tokenize the entire text ──────────────────────────────
        # We split into manageable chunks to avoid memory issues with very
        # large files, then concatenate all token IDs.
        # Disable post-processing for bulk encoding (we don't want <BOS>/<EOS>
        # around every chunk — we'll handle sequence boundaries ourselves).
        chunk_size = 100_000  # characters per chunk
        all_ids = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            encoded = tokenizer.encode(chunk)
            all_ids.extend(encoded.ids)

        self.token_ids = torch.tensor(all_ids, dtype=torch.long)
        print(f"    Total tokens: {len(self.token_ids):,}")
        print(f"    Sequences:    {len(self):,}  (seq_len={max_seq_len})")

    def __len__(self) -> int:
        # Number of complete (input+target) sequences we can extract.
        # We need max_seq_len + 1 tokens for each (input, target) pair.
        return max(0, len(self.token_ids) - self.max_seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids:  (max_seq_len,) — the input sequence
            target_ids: (max_seq_len,) — the target (shifted by 1)
        """
        chunk = self.token_ids[idx : idx + self.max_seq_len + 1]
        input_ids = chunk[:-1]   # [t_0 .. t_{n-1}]
        target_ids = chunk[1:]   # [t_1 .. t_n]
        return input_ids, target_ids


def create_dataloaders(
    tokenizer: Tokenizer,
    train_cfg: TrainConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    print("\n" + "=" * 60)
    print("  Creating Datasets & DataLoaders")
    print("=" * 60)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,           # Shuffle for training
        num_workers=2,
        pin_memory=True,
        drop_last=True,         # Drop incomplete last batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  Train batches per epoch: {len(train_loader):,}")
    print(f"  Val batches per epoch:   {len(val_loader):,}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    from train_tokenizer import load_tokenizer

    tok = load_tokenizer()
    train_loader, val_loader = create_dataloaders(tok)

    # Grab one batch
    batch = next(iter(train_loader))
    input_ids, target_ids = batch
    print(f"\n  Batch input shape:  {input_ids.shape}")
    print(f"  Batch target shape: {target_ids.shape}")
    print(f"  Sample input[:10]:  {input_ids[0, :10].tolist()}")
    print(f"  Sample target[:10]: {target_ids[0, :10].tolist()}")
    print("  ✅ Dataset works!")
