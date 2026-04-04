#!/usr/bin/env python3
"""
model.py — A GPT-style Transformer Language Model built from scratch in PyTorch.

=== WHAT IS THIS FILE? ===
This file implements every component of a decoder-only transformer, the same
architecture family used by GPT-2, GPT-3, LLaMA, and ChatGPT.  Each class is
heavily commented so you can read it top-to-bottom as a tutorial.

=== WHY "DECODER-ONLY"? ===
The original Transformer paper ("Attention Is All You Need", 2017) had an
encoder AND a decoder. Modern language models use ONLY the decoder part because:
  • We just want to predict the next token (no separate input/output language)
  • The causal (masked) attention in the decoder naturally supports left-to-right
    text generation
  • It's simpler and scales better

=== ARCHITECTURE (high level) ===

    Input token IDs    e.g. [42, 100, 7, 303]
         │
         ▼
    ┌──────────────────┐
    │  Token Embedding  │  — look up a learned vector for each token
    │  + Pos Embedding  │  — add position information (so the model knows word order)
    └────────┬─────────┘
             │  (×N transformer blocks — in our case N=4)
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
    │   Linear → logits │  — project back to vocabulary size (one score per word)
    └──────────────────┘

We use **pre-norm** style (LayerNorm before attention/FFN) which is more
stable during training than the original post-norm style.

=== CONCRETE EXAMPLE (our config) ===
With d_model=128, n_heads=4, n_layers=4, vocab_size=4000:
  • Input:  batch of token IDs, shape (batch_size, seq_len) e.g. (16, 128)
  • Each token gets a 128-dim embedding vector → (16, 128, 128)
  • After 4 transformer blocks, still (16, 128, 128)
  • Linear head projects to vocab → (16, 128, 4000) = logits (scores for each word)
  • Total parameters: ~1.2 million (tiny! GPT-3 has 175 billion)
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

    === WHY DO WE NEED POSITIONAL EMBEDDINGS? ===
    Unlike RNNs which process tokens one-by-one (so position is implicit),
    transformers process all tokens in parallel. Without positional info,
    the model can't distinguish "the cat sat on the mat" from "mat the the
    on sat cat" — the same tokens would produce the same embeddings!

    Positional embeddings solve this by adding a unique vector for each
    position (0, 1, 2, ...). The model LEARNS these vectors during training,
    so it can figure out the best way to encode "this is position 3".

    === ALTERNATIVES ===
    - **Sinusoidal** (original Transformer): fixed sin/cos functions. Works
      but can't be tuned by the model.
    - **Rotary (RoPE)** (LLaMA, modern models): encodes relative positions
      by rotating Q and K vectors. Better for long contexts.
    - **ALiBi** (BLOOM): adds a linear bias to attention scores.
    We use learned embeddings because they're the simplest to understand.

    === SHAPE ===
    The embedding table has shape (max_seq_len, d_model).
    For max_seq_len=128, d_model=128: a 128×128 lookup table.

    Example:
        Position 0 → [0.12, -0.03, 0.55, ..., 0.01]  (128 numbers)
        Position 1 → [-0.05, 0.22, 0.11, ..., -0.33]  (128 numbers)
        ...
        Position 127 → [0.08, 0.14, -0.02, ..., 0.19]  (128 numbers)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        """
        Args:
            max_seq_len: Maximum sequence length the model can handle.
                         For our model: 128 tokens.
            d_model:     Dimensionality of embeddings.
                         For our model: 128 dimensions.
        """
        super().__init__()
        # One embedding vector per position (0 .. max_seq_len-1)
        # nn.Embedding is just a lookup table: given an index, return a vector
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for positions [0, seq_len).

        Args:
            seq_len: Number of positions to generate embeddings for.

        Returns:
            Tensor of shape (seq_len, d_model) — one vector per position.

        Example:
            If seq_len=4 and d_model=128:
            Returns shape (4, 128) — 4 position vectors of 128 dims each.
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        # Look up each position's embedding vector
        return self.embedding(positions)  # (seq_len, d_model)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-HEAD CAUSAL SELF-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention — THE core innovation of transformers.

    ═══════════════════════════════════════════════════════════════════════════
    INTUITION: THE LIBRARY ANALOGY
    ═══════════════════════════════════════════════════════════════════════════

    Imagine you're in a library trying to understand a word in a sentence.
    Self-attention is like asking: "Which other words in this sentence should
    I look at to understand this word better?"

    For example, in "The cat sat on the mat because it was tired":
      • To understand "it", the model should attend to "cat" (not "mat")
      • To understand "tired", it should attend to "cat" and "it"

    ═══════════════════════════════════════════════════════════════════════════
    WHAT ARE Q, K, V? (Query, Key, Value)
    ═══════════════════════════════════════════════════════════════════════════

    Think of it like a search engine:
      • **Query (Q)**: "What am I looking for?" — Each token generates a query
        vector that represents what information it needs.
      • **Key (K)**: "What do I contain?" — Each token generates a key vector
        that advertises what information it has.
      • **Value (V)**: "Here's my actual content." — Each token generates a
        value vector that contains the actual information to be passed along.

    The attention score between token i and token j is:
        score(i,j) = Q_i · K_j  (dot product)
    High score = token i's query matches token j's key = i should attend to j.

    Concrete tiny example (imagine d_head=3 for simplicity):
        Token "cat" → Q = [0.5, 0.1, 0.8]    ("I'm looking for subjects")
        Token "sat" → K = [0.1, 0.9, 0.2]    ("I'm a verb")
        Token "it"  → K = [0.6, 0.2, 0.7]    ("I'm a pronoun/subject")

        score(cat, sat) = 0.5*0.1 + 0.1*0.9 + 0.8*0.2 = 0.30  (low)
        score(cat, it)  = 0.5*0.6 + 0.1*0.2 + 0.8*0.7 = 0.88  (high!)

    So "cat" pays more attention to "it" than "sat" — makes sense!

    ═══════════════════════════════════════════════════════════════════════════
    WHY DIVIDE BY √d_head? (Scaled Dot-Product Attention)
    ═══════════════════════════════════════════════════════════════════════════

    The dot product Q·K grows larger as the dimension increases. If d_head=32,
    the dot product of two random vectors has variance ~32, meaning scores
    could be huge numbers like ±50.

    When we apply softmax to huge numbers, it becomes nearly one-hot:
        softmax([50, 1, 2]) ≈ [1.0, 0.0, 0.0]  — only attends to one token!

    Dividing by √d_head normalizes the variance back to ~1:
        softmax([50/√32, 1/√32, 2/√32]) = softmax([8.8, 0.18, 0.35])
    This keeps the softmax output smooth, allowing the model to attend to
    multiple tokens with different weights.

    ═══════════════════════════════════════════════════════════════════════════
    HOW DOES THE CAUSAL MASK WORK?
    ═══════════════════════════════════════════════════════════════════════════

    For autoregressive language models, token at position i must NOT see tokens
    at positions i+1, i+2, ... (that would be "cheating" — seeing the future!).

    We enforce this with a triangular mask that sets future scores to -infinity:

        Position:     0    1    2    3
    Query pos 0:  [ 0.5  -inf -inf -inf ]  → can only see pos 0
    Query pos 1:  [ 0.3   0.7 -inf -inf ]  → can see pos 0, 1
    Query pos 2:  [ 0.1   0.4  0.8 -inf ]  → can see pos 0, 1, 2
    Query pos 3:  [ 0.2   0.5  0.3  0.6 ]  → can see all positions

    After softmax, -inf → 0, so future positions get zero attention weight.
    This is why it's called "causal" — information only flows from past → present.

    ═══════════════════════════════════════════════════════════════════════════
    WHAT DOES SOFTMAX DO TO THE SCORES?
    ═══════════════════════════════════════════════════════════════════════════

    Softmax converts raw scores into a probability distribution (sums to 1):
        scores = [2.0, 1.0, -inf, -inf]
        softmax → [0.73, 0.27, 0.0, 0.0]

    This means: "Pay 73% attention to token 0, 27% to token 1, none to 2 and 3."

    These weights are then used to compute a weighted average of the Value vectors:
        output = 0.73 * V[0] + 0.27 * V[1] + 0.0 * V[2] + 0.0 * V[3]

    So the output for each position is a BLEND of information from the tokens
    it attends to, weighted by how relevant they are.

    ═══════════════════════════════════════════════════════════════════════════
    WHY MULTI-HEAD? (Multiple Attention Heads)
    ═══════════════════════════════════════════════════════════════════════════

    Different aspects of language need different attention patterns:
      • Head 1 might learn to attend to the subject of the sentence
      • Head 2 might attend to the most recent noun
      • Head 3 might attend to syntactic structure
      • Head 4 might attend to semantic similarity

    Instead of one big attention with d_model=128, we split into n_heads=4
    independent attention computations, each with d_head=32. The outputs are
    concatenated and projected back to d_model=128.

    This is like having 4 different "perspectives" on the same sequence.

    ═══════════════════════════════════════════════════════════════════════════
    THE COMPLETE ATTENTION FORMULA
    ═══════════════════════════════════════════════════════════════════════════

        Attention(Q, K, V) = softmax( Q K^T / √d_head + mask ) × V

    Where d_head = d_model / n_heads is the dimension per head.

    Step-by-step with shapes (batch=B, seq_len=T, d_model=C, n_heads=H, d_head=D):
      1. Project: x (B,T,C) → Q,K,V each (B,T,C) → reshape to (B,H,T,D)
      2. Scores: Q @ K^T → (B,H,T,T) — attention score matrix
      3. Scale: divide by √D
      4. Mask: set upper triangle to -inf
      5. Softmax: normalize each row to sum to 1 → attention weights (B,H,T,T)
      6. Apply: weights @ V → (B,H,T,D)
      7. Concat heads: reshape (B,H,T,D) → (B,T,C)
      8. Project: linear layer → (B,T,C)
    """

    def __init__(self, cfg: ModelConfig):
        """
        Initialize the multi-head causal self-attention layer.

        Args:
            cfg: Model configuration containing:
                - d_model (int): Total model dimension (e.g., 128)
                - n_heads (int): Number of attention heads (e.g., 4)
                - max_seq_len (int): Maximum sequence length (e.g., 128)
                - dropout (float): Dropout rate (e.g., 0.1)

        The dimensions work out as:
            d_head = d_model / n_heads = 128 / 4 = 32
            Each head independently operates on 32-dimensional vectors.
        """
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        self.n_heads = cfg.n_heads           # Number of attention heads (4)
        self.d_head = cfg.d_model // cfg.n_heads  # Dimension per head (128/4=32)
        self.d_model = cfg.d_model           # Total model dimension (128)

        # ── Q, K, V Projection ─────────────────────────────────────────────
        # Instead of 3 separate linear layers (one each for Q, K, V), we use
        # ONE linear layer that produces all three at once. This is more
        # efficient because it's a single matrix multiplication.
        #
        # Input:  (batch, seq_len, d_model)     e.g., (16, 128, 128)
        # Output: (batch, seq_len, 3 * d_model) e.g., (16, 128, 384)
        #         └── first 128 dims = Q, next 128 = K, last 128 = V
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # ── Output Projection ──────────────────────────────────────────────
        # After computing attention for each head and concatenating the results,
        # this linear layer mixes information across heads.
        # Input:  (batch, seq_len, d_model)  — concatenated head outputs
        # Output: (batch, seq_len, d_model)  — final attention output
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # ── Dropout ────────────────────────────────────────────────────────
        # Applied to attention weights (prevents over-relying on specific tokens)
        # and to the output (standard regularization).
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # ── Causal Mask (Pre-computed) ─────────────────────────────────────
        # We create the mask once and reuse it for every forward pass.
        # torch.triu creates an upper-triangular matrix:
        #
        #   [[False,  True,  True,  True],   ← position 0: mask out 1,2,3
        #    [False, False,  True,  True],   ← position 1: mask out 2,3
        #    [False, False, False,  True],   ← position 2: mask out 3
        #    [False, False, False, False]]   ← position 3: mask out nothing
        #
        # True = "this position should be masked (set to -inf)"
        # False = "this position is allowed (keep the score)"
        mask = torch.triu(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1
        ).bool()
        # register_buffer: saves with the model but is NOT a trainable parameter.
        # It also automatically moves to the right device (CPU/GPU).
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               Example: (16, 128, 128) — 16 sequences of 128 tokens, each
               represented by a 128-dimensional vector.

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Each token's representation now incorporates information from
            all previous tokens (via attention).

        Step-by-step example with batch=1, seq_len=4, d_model=128, n_heads=4:
            Input x:  (1, 4, 128)
            ↓ QKV projection
            qkv:      (1, 4, 384)   — Q, K, V concatenated
            ↓ split into Q, K, V
            Q, K, V:  each (1, 4, 128)
            ↓ reshape for multi-head: split d_model into n_heads × d_head
            Q, K, V:  each (1, 4, 4, 32) → transpose → (1, 4, 4, 32)
                       (batch, heads, seq_len, d_head)
            ↓ attention scores: Q @ K^T
            scores:   (1, 4, 4, 4)  — 4 heads, each with a 4×4 score matrix
            ↓ scale by 1/√32
            ↓ mask future positions with -inf
            ↓ softmax → attention weights
            weights:  (1, 4, 4, 4)  — each row sums to 1
            ↓ weights @ V
            out:      (1, 4, 4, 32) — weighted combination of values
            ↓ concatenate heads
            out:      (1, 4, 128)   — back to d_model
            ↓ output projection
            out:      (1, 4, 128)   — final output
        """
        B, T, C = x.shape  # B=batch, T=seq_len, C=d_model (channels)

        # ══════════════════════════════════════════════════════════════════
        # Step 1: Compute Q, K, V (Query, Key, Value)
        # ══════════════════════════════════════════════════════════════════
        # A single matrix multiply produces Q, K, V all at once.
        # This is equivalent to:
        #   Q = x @ W_Q   (where W_Q has shape d_model × d_model)
        #   K = x @ W_K
        #   V = x @ W_V
        # but combined into one operation for efficiency.
        qkv = self.qkv_proj(x)                        # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                # each (B, T, C)
        # chunk(3, dim=-1) splits the last dimension into 3 equal parts:
        #   qkv[:, :, 0:128]   → Q
        #   qkv[:, :, 128:256] → K
        #   qkv[:, :, 256:384] → V

        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        # This separates the heads so each head computes attention independently.
        #
        # Example: (1, 4, 128) → (1, 4, 4, 32) → (1, 4, 4, 32)
        #   The 128-dim vector is split into 4 heads of 32 dims each.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ══════════════════════════════════════════════════════════════════
        # Step 2: Scaled Dot-Product Attention
        # ══════════════════════════════════════════════════════════════════
        # Compute attention scores: how much should each token attend to
        # every other token?
        #
        # Formula: scores = (Q @ K^T) / √d_head
        #
        # Q @ K^T: for each pair of positions (i, j), compute the dot product
        # of Q[i] and K[j]. High dot product = similar directions = high attention.
        #
        # Division by √d_head: prevents scores from becoming too large, which
        # would make softmax nearly one-hot (see explanation in class docstring).
        scale = math.sqrt(self.d_head)  # √32 ≈ 5.66
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)
        # Result: a T×T matrix for each head in each batch item.
        # attn_weights[b, h, i, j] = "how much should token i attend to token j
        #                              in head h of batch item b"

        # ── Apply Causal Mask ──────────────────────────────────────────────
        # Set attention scores for future positions to -infinity.
        # After softmax, -inf → 0, effectively preventing any information
        # from flowing backwards in time.
        #
        # Before mask (example, 4 tokens):
        #   [[0.5,  0.3,  0.8,  0.1],    ← token 0 can see all (but shouldn't!)
        #    [0.2,  0.7,  0.4,  0.6],    ← token 1
        #    [0.1,  0.3,  0.9,  0.5],    ← token 2
        #    [0.4,  0.2,  0.1,  0.8]]   ← token 3
        #
        # After mask:
        #   [[0.5,  -inf, -inf, -inf],   ← token 0 only sees itself
        #    [0.2,  0.7,  -inf, -inf],   ← token 1 sees tokens 0, 1
        #    [0.1,  0.3,  0.9,  -inf],   ← token 2 sees tokens 0, 1, 2
        #    [0.4,  0.2,  0.1,  0.8]]   ← token 3 sees all (past = everything)
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),  # (1, 1, T, T)
            float("-inf"),
        )

        # ── Softmax → Normalized Attention Weights ─────────────────────────
        # Softmax converts scores to probabilities (each row sums to 1).
        # The -inf entries become 0 after softmax.
        #
        # Example (continuing from above, row for token 1):
        #   scores = [0.2, 0.7, -inf, -inf]
        #   softmax → [e^0.2, e^0.7, 0, 0] / sum = [0.38, 0.62, 0, 0]
        #   Token 1 pays 38% attention to token 0 and 62% to itself.
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, n_heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)
        # Dropout randomly zeroes some weights → prevents over-reliance on
        # specific tokens and acts as regularization.

        # ══════════════════════════════════════════════════════════════════
        # Step 3: Weighted Sum of Values
        # ══════════════════════════════════════════════════════════════════
        # Use the attention weights to compute a weighted average of the
        # Value vectors. This is how information flows between tokens.
        #
        # For each position i, its output is:
        #   output[i] = Σ_j  attn_weight[i,j] × V[j]
        #
        # Example: if attn_weights for token 1 are [0.38, 0.62, 0, 0]:
        #   output[1] = 0.38 * V[0] + 0.62 * V[1]
        #   (a blend of information from tokens 0 and 1)
        out = attn_weights @ v  # (B, n_heads, T, d_head)
        # Matrix multiply: (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        # Each position now contains a weighted mix of value vectors.

        # ══════════════════════════════════════════════════════════════════
        # Step 4: Concatenate Heads and Project
        # ══════════════════════════════════════════════════════════════════
        # Merge the outputs from all heads back into a single d_model vector.
        #
        # (B, n_heads, T, d_head) → (B, T, n_heads, d_head) → (B, T, d_model)
        # Example: (1, 4, 4, 32) → (1, 4, 4, 32) → (1, 4, 128)
        #   The 4 heads each contributed a 32-dim output, concatenated = 128.
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # .contiguous() ensures the tensor is stored contiguously in memory
        # (required before .view() which changes the shape interpretation).

        # Final linear projection: mixes information across heads.
        # This allows the model to combine what different heads learned.
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEED-FORWARD NETWORK (FFN)
# ═══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    === WHAT DOES THIS DO? ===
    After attention gathers context from other tokens, the FFN processes each
    token's representation INDEPENDENTLY through a small neural network.
    Think of attention as "gathering information from context" and FFN as
    "thinking about what that information means".

    === ARCHITECTURE ===
    A simple 2-layer MLP applied to each position separately:

        FFN(x) = Linear_2( GELU( Linear_1(x) ) )

    The shape transformation is:
        (batch, seq_len, d_model)        e.g., (16, 128, 128)
        → Linear_1 → (batch, seq_len, d_ff)    e.g., (16, 128, 512)
        → GELU activation
        → Linear_2 → (batch, seq_len, d_model)  e.g., (16, 128, 128)

    === WHY 4× EXPANSION? ===
    The inner dimension d_ff is typically 4× d_model (128 → 512 in our case).
    This "expand then contract" pattern gives the network more capacity to
    learn complex transformations. It's like having a wider "thinking space"
    before compressing back to the model dimension.

    Research has shown that much of a transformer's "knowledge" (facts,
    associations) is stored in these FFN layers.

    === WHAT IS GELU? ===
    GELU (Gaussian Error Linear Unit) is the activation function used in
    GPT-2 and most modern transformers. It's a smooth version of ReLU:
        GELU(x) = x × Φ(x)   where Φ is the Gaussian CDF
        ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715x³)))

    Compared to ReLU:
        ReLU(x)  = max(0, x)           — sharp cutoff at 0
        GELU(x) ≈ x if x >> 0, ≈ 0 if x << 0  — smooth transition

    The smoothness helps with gradient flow during training.

    === CONCRETE EXAMPLE ===
    For one token with d_model=128:
        Input:  [0.1, -0.3, 0.5, ..., 0.2]  (128 numbers)
        ↓ Linear_1 (128 → 512 weights)
        Hidden: [0.4, -0.1, 0.8, ..., -0.3]  (512 numbers — expanded!)
        ↓ GELU activation (element-wise)
        Hidden: [0.26, -0.04, 0.63, ..., -0.11]  (512 numbers)
        ↓ Linear_2 (512 → 128 weights)
        Output: [0.3, -0.1, 0.6, ..., 0.1]  (128 numbers — compressed back)
    """

    def __init__(self, cfg: ModelConfig):
        """
        Args:
            cfg: Model configuration containing:
                - d_model (int): Input/output dimension (128)
                - d_ff (int): Inner/hidden dimension (512)
                - dropout (float): Dropout rate (0.1)
        """
        super().__init__()
        # Expansion: d_model → d_ff (128 → 512)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        # Contraction: d_ff → d_model (512 → 128)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        # Regularization dropout
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to each position independently.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               Example: (16, 128, 128)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Same shape as input — the FFN transforms each token's
            representation but doesn't change the dimensions.
        """
        x = self.fc1(x)          # (B, T, d_ff)    — expand to wider space
        x = F.gelu(x)            # (B, T, d_ff)    — non-linear activation
        x = self.fc2(x)          # (B, T, d_model)  — compress back
        x = self.dropout(x)      # (B, T, d_model)  — regularization
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with pre-norm residual connections.

    === WHAT IS A TRANSFORMER BLOCK? ===
    One complete processing unit that combines attention (inter-token communication)
    with a feed-forward network (per-token processing). Our model stacks 4 of these.

    === DATA FLOW ===

        x  ──────────────────────────────────────┐
        │                                         │ (residual / skip connection)
        ├→ LayerNorm → Multi-Head Attention ──→  + (add)
        │                                         │
        │  ──────────────────────────────────────┐│
        │                                         │ (another residual)
        └→ LayerNorm → Feed-Forward Network ──→  + (add)
                                                  │
                                                  ▼ output

    === WHY RESIDUAL CONNECTIONS? ===
    The "+" operations are residual (skip) connections. Instead of computing
    output = f(x), we compute output = x + f(x).

    Benefits:
    1. **Gradient flow**: In deep networks, gradients can vanish during
       backpropagation. The skip connection provides a "highway" for gradients
       to flow directly back to earlier layers.
    2. **Easy identity**: If the sublayer learns nothing useful, f(x) ≈ 0,
       and x passes through unchanged. This makes it easy to add more layers
       without hurting performance.
    3. **Iterative refinement**: Each block adds a small "correction" to the
       representation rather than replacing it entirely.

    === WHY PRE-NORM? ===
    We apply LayerNorm BEFORE the sublayer (attention/FFN), not after.

    Post-norm (original paper): x = LayerNorm(x + sublayer(x))
    Pre-norm (GPT-2, our model): x = x + sublayer(LayerNorm(x))

    Pre-norm is more stable because the residual stream always carries the
    "raw" (unnormalized) activations, making gradients more well-behaved.
    Most modern LLMs use pre-norm.

    === WHAT IS LAYER NORMALIZATION? ===
    LayerNorm normalizes each token's representation to have mean=0, std=1:

        LayerNorm(x) = (x - mean) / std × γ + β

    where γ (scale) and β (shift) are learnable parameters.

    Example for one token with d_model=4:
        Input:     [2.0, 4.0, 6.0, 8.0]
        Mean:      5.0
        Std:       2.24
        Normalized: [-1.34, -0.45, 0.45, 1.34]
        After γ,β: (learned scaling and shifting)

    This keeps activations in a reasonable range, preventing them from
    exploding or vanishing as they pass through many layers.
    """

    def __init__(self, cfg: ModelConfig):
        """
        Args:
            cfg: Model configuration containing d_model, n_heads, etc.
        """
        super().__init__()
        # LayerNorm before attention (pre-norm style)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        # Multi-head causal self-attention
        self.attn = CausalSelfAttention(cfg)
        # LayerNorm before FFN
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # Feed-forward network
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process one transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               Example: (16, 128, 128)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Same shape, but now each token's representation has been:
            1. Updated with context from other tokens (via attention)
            2. Transformed individually (via FFN)
        """
        # ── Attention sub-layer with residual connection ───────────────
        # x_in → LayerNorm → Attention → add residual
        # The original x is ADDED back (skip connection), so the attention
        # only needs to learn the "delta" (what to change).
        x = x + self.attn(self.ln1(x))

        # ── FFN sub-layer with residual connection ─────────────────────
        # Same pattern: normalize, process, add residual
        x = x + self.ffn(self.ln2(x))

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THE FULL GPT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SmallGPT(nn.Module):
    """
    A small GPT-style language model — the complete, top-level module.

    === WHAT DOES THIS MODEL DO? ===
    Given a sequence of tokens like ["Once", "upon", "a"], it predicts the
    probability distribution for the NEXT token at each position:
      • After "Once" → predict "upon" (and all other possible words)
      • After "Once upon" → predict "a"
      • After "Once upon a" → predict "time"

    === TRAINING OBJECTIVE ===
    **Next-token prediction** (causal language modeling):
      • Input:  [t_0, t_1, ..., t_{n-1}]
      • Target: [t_1, t_2, ..., t_n]
      • Loss: cross-entropy between predicted and actual next tokens

    Cross-entropy loss measures how "surprised" the model is by the true next
    token. If the model assigns high probability to the correct token, loss is
    low. Training minimizes this loss by adjusting all the model's weights.

    === WEIGHT TYING ===
    We share weights between the token embedding layer and the output (LM head)
    layer. Both layers relate tokens to vectors:
      • Embedding: token_id → vector (lookup)
      • LM head: vector → token_scores (projection)

    Sharing weights reduces parameters by ~500K and often improves performance.
    The intuition: if "cat" and "dog" have similar embeddings, they should also
    have similar output probabilities in similar contexts.

    === PARAMETER COUNT ===
    With our default config (d_model=128, n_layers=4, vocab_size=4000):
      • Token embeddings: 4000 × 128 = 512,000
      • Position embeddings: 128 × 128 = 16,384
      • Per transformer block: ~200K (attention + FFN + norms)
      • 4 blocks total: ~800K
      • LM head: (tied with token embeddings)
      • TOTAL: ~1.2 million parameters

    For comparison:
      • GPT-2 Small: 117 million
      • GPT-3: 175 billion
      • GPT-4: rumored ~1.8 trillion
    """

    def __init__(self, cfg: ModelConfig | None = None):
        """
        Initialize the SmallGPT model.

        Args:
            cfg: Model configuration. If None, uses default ModelConfig().
                 Key settings:
                   vocab_size=4000, max_seq_len=128, d_model=128,
                   n_layers=4, n_heads=4, d_ff=512, dropout=0.1
        """
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # ── Token Embedding ────────────────────────────────────────────
        # Maps each token ID (integer) to a d_model-dimensional vector.
        # This is a lookup table with vocab_size rows and d_model columns.
        #
        # Example: token_id=42 → embedding[42] = [0.1, -0.3, ..., 0.5]
        #
        # padding_idx=0 means the <PAD> token always gets a zero vector.
        # This is important for variable-length sequences where we pad
        # shorter sequences with <PAD> tokens — we don't want padding
        # to contribute any information.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)

        # ── Positional Embedding ───────────────────────────────────────
        # Adds position information so the model knows word ORDER.
        # Without this, "the cat ate the fish" and "the fish ate the cat"
        # would be indistinguishable after embedding!
        self.pos_emb = LearnedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)

        # ── Embedding Dropout ──────────────────────────────────────────
        # Regularization: randomly zeros some embedding dimensions during
        # training to prevent overfitting.
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # ── Stack of Transformer Blocks ────────────────────────────────
        # The core of the model: n_layers transformer blocks stacked on top.
        # Each block applies attention + FFN. Information flows through all
        # blocks sequentially, with each block refining the representations.
        #
        # Think of it like layers of understanding:
        #   Block 0: learns basic token relationships (e.g., "the" + noun)
        #   Block 1: learns phrase structure (e.g., subject-verb agreement)
        #   Block 2: learns semantic patterns (e.g., story continuation)
        #   Block 3: learns output-specific features (e.g., next word)
        # (This is a simplification — in practice the division isn't so clean)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # ── Output Head ────────────────────────────────────────────────
        # After all transformer blocks, we need to predict the next token.
        # 1. Final LayerNorm: normalize the final representations
        # 2. Linear head: project from d_model to vocab_size to get a
        #    score (logit) for every possible next token
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ── Weight Tying ───────────────────────────────────────────────
        # Share weights between the token embedding and the LM head.
        # Instead of two separate (vocab_size × d_model) matrices, we use one.
        # This is a common technique that:
        #   1. Reduces parameter count (saves 512K parameters for us!)
        #   2. Often improves performance (the embedding and un-embedding
        #      should be consistent: similar tokens → similar representations)
        self.lm_head.weight = self.token_emb.weight

        # ── Weight Initialization ──────────────────────────────────────
        # Proper initialization is crucial for training stability.
        # We use GPT-2's initialization scheme.
        self.apply(self._init_weights)
        print(f"  SmallGPT initialized: {self.count_parameters()/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2's scheme for training stability.

        The key principles:
          1. Linear layers and embeddings: Normal(0, 0.02) — small random values
             so that initial outputs are near zero and gradients are reasonable.
          2. Biases: initialized to zero.
          3. LayerNorm: γ=1, β=0 (starts as identity transform).

        Why 0.02? It's small enough to keep initial activations in a reasonable
        range, but large enough that the model can learn quickly. This is an
        empirical choice from the GPT-2 paper.

        Args:
            module: A PyTorch module (called recursively by self.apply())
        """
        if isinstance(module, nn.Linear):
            # Small random weights — neither too big (exploding gradients)
            # nor too small (vanishing gradients)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Padding embedding should always be zero (carries no information)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Start as identity: γ=1, β=0 → LayerNorm(x) ≈ (x - mean) / std
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model.

        Returns:
            Number of trainable parameters (int).
            For our default config: ~1.2 million.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass: tokens in → logits out (+ optional loss).

        This is the main method that runs the entire model end-to-end.

        Args:
            input_ids: (batch, seq_len) — integer token IDs.
                       Example: tensor([[2, 42, 100, 7, 303]])
                       where 2=<BOS>, 42="Once", etc.

            targets:   (batch, seq_len) — target token IDs for loss computation.
                       Usually input_ids shifted by one position:
                       If input  = [<BOS>, Once, upon, a]
                       then target = [Once, upon, a, time]
                       If None, no loss is computed (used during generation).

        Returns:
            logits: (batch, seq_len, vocab_size) — prediction scores for each
                    position and each possible next token.
                    Example: (16, 128, 4000)

            loss:   Scalar cross-entropy loss, or None if targets not provided.
                    Lower loss = better predictions.
                    Random model: loss ≈ ln(vocab_size) ≈ 8.3
                    Trained model: loss ≈ 3-5 (for our small model)

        Step-by-step shape example (batch=1, seq_len=4, d_model=128, vocab=4000):
            input_ids: (1, 4)          — 4 token IDs
            ↓ token embedding
            tok_emb:   (1, 4, 128)     — each ID → 128-dim vector
            ↓ + positional embedding
            x:         (1, 4, 128)     — now has position info
            ↓ 4 transformer blocks (each: attention + FFN)
            x:         (1, 4, 128)     — refined representations
            ↓ final LayerNorm
            x:         (1, 4, 128)     — normalized
            ↓ linear head (128 → 4000)
            logits:    (1, 4, 4000)    — one score per vocab word per position
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        # ── Step 1: Embed tokens + positions ──────────────────────────
        # Token embedding: look up a learned vector for each token ID
        tok_emb = self.token_emb(input_ids)        # (B, T, d_model)
        # Positional embedding: add position vectors
        pos_emb = self.pos_emb(T)                  # (T, d_model)
        # Add them together: tok_emb has shape (B,T,d_model) and pos_emb
        # has shape (T,d_model). Broadcasting adds the same position
        # vectors to every item in the batch.
        x = self.emb_dropout(tok_emb + pos_emb)    # (B, T, d_model)

        # ── Step 2: Pass through transformer blocks ───────────────────
        # Each block applies: attention → FFN (with residuals and norms).
        # The representation is iteratively refined by each block.
        for block in self.blocks:
            x = block(x)   # (B, T, d_model) → (B, T, d_model)

        # ── Step 3: Final LayerNorm + project to vocabulary ───────────
        x = self.ln_final(x)                       # (B, T, d_model)
        # The LM head maps each d_model vector to vocab_size scores.
        # logits[b, t, w] = "score for word w being the next token after position t"
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        # ── Step 4: Compute loss if targets are provided ──────────────
        loss = None
        if targets is not None:
            # Cross-entropy loss measures how well the model predicts the
            # actual next token at each position.
            #
            # We flatten the batch and sequence dimensions:
            #   logits:  (B, T, vocab_size) → (B*T, vocab_size)
            #   targets: (B, T)             → (B*T,)
            #
            # ignore_index=pad_token_id: don't penalize the model for
            # predictions at padding positions (they're meaningless).
            #
            # Example: if target token is 42 and logits give high score
            # to token 42, loss is low. If they give high score to token
            # 999 instead, loss is high.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                targets.view(-1),                    # (B*T,)
                ignore_index=self.cfg.pad_token_id,
            )

        return logits, loss


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test — run this file directly to verify everything works
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SmallGPT — Quick Forward Pass Test")
    print("=" * 60)

    cfg = ModelConfig()
    model = SmallGPT(cfg)

    # Create dummy input: batch of 2 sequences, each 32 tokens long
    # Token IDs are random integers from 0 to vocab_size
    dummy_ids = torch.randint(0, cfg.vocab_size, (2, 32))  # (batch=2, seq=32)
    print(f"\n  Input shape:  {dummy_ids.shape}  (batch=2, seq_len=32)")

    # Forward pass with targets (for training — computes loss)
    logits, loss = model(dummy_ids, targets=dummy_ids)
    print(f"  Logits shape: {logits.shape}  (batch=2, seq=32, vocab={cfg.vocab_size})")
    print(f"  Loss:         {loss.item():.4f}")
    print(f"    (Expected ~{math.log(cfg.vocab_size):.1f} for random model = ln(vocab_size))")

    # Forward pass without targets (for generation — no loss)
    logits_only, no_loss = model(dummy_ids)
    print(f"\n  Without targets: loss = {no_loss}")
    print("  ✅ Model forward pass works!")
