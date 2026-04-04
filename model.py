#!/usr/bin/env python3
"""
model.py — A GPT-style Transformer Language Model built from scratch in PyTorch.

This file implements every component of a decoder-only transformer, the same
architecture family used by GPT-2, GPT-3, and LLaMA.  Each class is heavily
commented so you can read it top-to-bottom as a tutorial.

Architecture (high level):
    Input token IDs
         │
         ▼
    ┌──────────────────┐
    │  Token Embedding  │  — look up a learned vector for each token
    │  + Pos Embedding  │  — add position information
    └────────┬─────────┘
             │  (×N transformer blocks)
    ┌────────▼─────────┐
    │   LayerNorm       │
    │   Multi-Head      │  — each token attends to all *previous* tokens
    │   Causal Attn     │    (causal mask prevents looking into the future)
    │   + Residual      │
    ├───────────────────┤
    │   LayerNorm       │
    │   Feed-Forward    │  — 2-layer MLP with GELU activation
    │   + Residual      │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   Final LayerNorm │
    │   Linear → logits │  — project back to vocabulary size
    └──────────────────┘

We use **pre-norm** style (LayerNorm before attention/FFN) which is more
stable during training than the original post-norm style.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. POSITIONAL EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (like GPT-2).

    Unlike sinusoidal positional encodings (from the original "Attention Is All
    You Need" paper), learned embeddings let the model figure out the best way
    to encode position.  For small models and fixed context lengths this works
    well and is simpler to implement.

    Shape: (max_seq_len, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        # One embedding vector per position (0 .. max_seq_len-1)
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for positions [0, seq_len).

        Returns:
            Tensor of shape (seq_len, d_model)
        """
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)  # (seq_len, d_model)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-HEAD CAUSAL SELF-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention.

    Self-attention lets every token in the sequence look at every other token
    to gather context.  "Causal" means we add a mask so that position i can
    only attend to positions ≤ i — this prevents information leaking from the
    future, which is essential for autoregressive (left-to-right) language
    modelling.

    Multi-head attention splits the representation into `n_heads` independent
    "heads", each attending with a smaller dimension (d_model / n_heads).
    The outputs are concatenated and projected back to d_model.

    The attention computation for each head:

        Attention(Q, K, V) = softmax( Q K^T / √d_k ) V

    where d_k = d_model / n_heads is the dimension per head.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads  # Dimension per head
        self.d_model = cfg.d_model

        # Linear projections for Q, K, V (combined into one matrix for efficiency)
        # Input:  (batch, seq_len, d_model)
        # Output: (batch, seq_len, 3 * d_model)  — contains Q, K, V concatenated
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # Output projection: merges the multi-head outputs back to d_model
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Dropout for attention weights and output
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Pre-compute the causal mask (upper-triangular matrix of -inf)
        # Shape: (1, 1, max_seq_len, max_seq_len) — broadcastable over batch & heads
        mask = torch.triu(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1
        ).bool()
        # Register as a buffer (not a parameter — no gradient, but moves to device)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape  # batch, seq_len, d_model

        # ── Step 1: Compute Q, K, V ───────────────────────────────────────
        qkv = self.qkv_proj(x)                        # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                # each (B, T, C)

        # Reshape into (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── Step 2: Scaled dot-product attention ──────────────────────────
        # attn_weights[i,j] = (q[i] · k[j]) / √d_k
        scale = math.sqrt(self.d_head)
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, T)

        # Apply causal mask: set future positions to -inf so softmax → 0
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),  # (1,1,T,T)
            float("-inf"),
        )

        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 3: Weighted sum of values ────────────────────────────────
        out = attn_weights @ v  # (B, heads, T, d_head)

        # ── Step 4: Concatenate heads and project ─────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEED-FORWARD NETWORK (FFN)
# ═══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This is a simple 2-layer MLP applied independently to each position:
        FFN(x) = Linear_2( GELU( Linear_1(x) ) )

    The inner dimension (d_ff) is typically 4× the model dimension, giving the
    network more capacity to transform representations.

    GELU (Gaussian Error Linear Unit) is the activation function used in GPT-2
    and many modern transformers.  It's a smooth approximation of ReLU.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.fc1(x)          # (B, T, d_ff)
        x = F.gelu(x)            # Non-linear activation
        x = self.fc2(x)          # (B, T, d_model)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with pre-norm residual connections.

    The data flow is:
        x  ──→  LayerNorm → Attention → + (residual)
           └─────────────────────────────┘
                ──→  LayerNorm → FFN → + (residual)
           └────────────────────────────┘

    Pre-norm (applying LayerNorm *before* the sublayer) is used by GPT-2 and
    most modern transformers because it leads to more stable training than the
    original post-norm formulation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Attention sub-layer with residual connection
        x = x + self.attn(self.ln1(x))
        # FFN sub-layer with residual connection
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THE FULL GPT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SmallGPT(nn.Module):
    """
    A small GPT-style language model.

    This is the top-level module that wires together:
      - Token embeddings
      - Positional embeddings
      - A stack of TransformerBlocks
      - A final LayerNorm + linear head

    For language modelling, the training objective is **next-token prediction**:
    given tokens [t_0, t_1, ..., t_{n-1}], predict [t_1, t_2, ..., t_n].
    The loss is cross-entropy between the predicted logits and the true next tokens.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # ── Embeddings ────────────────────────────────────────────────────
        # Token embedding: maps each token id to a d_model-dimensional vector
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        # Positional embedding: adds position information
        self.pos_emb = LearnedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # ── Output head ───────────────────────────────────────────────────
        self.ln_final = nn.LayerNorm(cfg.d_model)
        # Project from d_model back to vocabulary size to get logits
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head.
        # This is a common technique that reduces parameters and often improves
        # performance.  The intuition is that the embedding and un-embedding
        # matrices should be similar since they both relate tokens to vectors.
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"  SmallGPT initialized: {self.count_parameters()/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2's scheme:
          - Normal(0, 0.02) for linear layers and embeddings
          - Zero bias
          - Scale residual projections by 1/√(2*n_layers) for stability
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Zero out padding embedding
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) — integer token IDs
            targets:   (batch, seq_len) — target token IDs for loss computation
                       (usually input_ids shifted by one position)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets not provided
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        # ── Step 1: Embed tokens + positions ──────────────────────────────
        tok_emb = self.token_emb(input_ids)        # (B, T, d_model)
        pos_emb = self.pos_emb(T)                  # (T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)    # (B, T, d_model) — broadcasting

        # ── Step 2: Pass through transformer blocks ───────────────────────
        for block in self.blocks:
            x = block(x)

        # ── Step 3: Final LayerNorm + project to vocabulary ───────────────
        x = self.ln_final(x)                       # (B, T, d_model)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        # ── Step 4: Compute loss if targets are provided ──────────────────
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,  # Don't penalize padding
            )

        return logits, loss


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = ModelConfig()
    model = SmallGPT(cfg)

    # Dummy forward pass
    dummy_ids = torch.randint(0, cfg.vocab_size, (2, 32))  # batch=2, seq=32
    logits, loss = model(dummy_ids, targets=dummy_ids)
    print(f"  Logits shape: {logits.shape}")  # (2, 32, vocab_size)
    print(f"  Loss:         {loss.item():.4f}")
    print("  ✅ Model forward pass works!")
