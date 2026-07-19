"""
tiny_gpt.py — A tiny GPT in plain NumPy (same idea as the algo.monster course).

What this file does:
  1. Load trained weights from models/tiny_english_gpt.npz
  2. Run a full transformer forward pass (attention, FFN, etc.)
  3. Generate a few words, one at a time

No PyTorch here — just NumPy, so you can see every multiply and add.
Train the weights first with:  python train.py
"""

import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "tiny_english_gpt.npz"


# =============================================================================
# Small math helpers
# =============================================================================

def gelu(x):
    """Smooth activation used inside the feed-forward layers."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    """Turn scores into probabilities that sum to 1."""
    # Subtract max for numerical stability (doesn't change the result)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """Normalize each position's features, then scale + shift with learned params."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    return weight * x + bias


# =============================================================================
# Transformer pieces
# =============================================================================

def attention(x, W_q, W_k, W_v, W_o, n_heads, mask):
    """
    Multi-head causal self-attention.

    For each word, ask: "which earlier words matter?" then mix their values.
    The causal mask blocks looking at future words.
    """
    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Project into Queries, Keys, Values  (PyTorch stores W as [out, in], so .T)
    Q = (x @ W_q).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)  # (heads, seq, d_k)
    K = (x @ W_k).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = (x @ W_v).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Scores: how much each word should attend to every other word
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    scores = scores + (1.0 - mask) * -1e9  # mask future positions

    weights = softmax(scores, axis=-1)     # attention weights
    out = weights @ V                      # weighted mix of values
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o                       # final linear mix across heads


def feed_forward(x, W1, b1, W2, b2):
    """Two linear layers with GELU in between (per-word "thinking")."""
    return gelu(x @ W1 + b1) @ W2 + b2


def block(x, w, i, n_heads, mask):
    """
    One transformer block:
      LayerNorm → Attention → residual add
      LayerNorm → FeedForward → residual add
    """
    p = f"blocks.{i}"

    # --- Attention ---
    x_n = layer_norm(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"])
    x = x + attention(
        x_n,
        w[f"{p}.attn.W_q.weight"].T,
        w[f"{p}.attn.W_k.weight"].T,
        w[f"{p}.attn.W_v.weight"].T,
        w[f"{p}.attn.W_o.weight"].T,
        n_heads,
        mask,
    )

    # --- Feed-forward ---
    x_n = layer_norm(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"])
    x = x + feed_forward(
        x_n,
        w[f"{p}.ff.linear1.weight"].T,
        w[f"{p}.ff.linear1.bias"],
        w[f"{p}.ff.linear2.weight"].T,
        w[f"{p}.ff.linear2.bias"],
    )
    return x


def transformer(tokens, w):
    """
    Full forward pass: embeddings → N blocks → final norm → vocab logits.

    tokens: list of integer word ids, e.g. [0, 9, 1, 3, 5, 0] for
            "the big cat sat on the"
    returns: logits shaped (seq_len, vocab_size)
    """
    seq_len = len(tokens)

    # Word embedding + position embedding
    x = w["token_embed.weight"][tokens]
    x = x + w["pos_embed.weight"][:seq_len]

    # Lower-triangular mask: position i can only see positions <= i
    mask = np.tril(np.ones((seq_len, seq_len)))

    n_layers = int(w["n_layers"])
    n_heads = int(w["n_heads"])
    for i in range(n_layers):
        x = block(x, w, i, n_heads, mask)

    x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    logits = x @ w["lm_head.weight"].T  # score for every vocab word
    return logits


# =============================================================================
# Generation
# =============================================================================

def generate(text, vocab, w, num_words=3, temperature=0.0):
    """
    Predict the next words one by one (autoregressive).

    temperature = 0  → always pick the highest-probability word (greedy)
    temperature > 0  → sample (more random)
    """
    words = text.split()
    tokens = [vocab.index(word) for word in words if word in vocab]
    if not tokens:
        print(f"No known words in: '{text}'\n")
        return

    print(f"Input: '{text}'")
    generated = text

    for step in range(num_words):
        logits = transformer(tokens, w)
        last = logits[-1]  # scores for the next word

        if temperature == 0:
            probs = softmax(last)
            next_id = int(np.argmax(last))
        else:
            probs = softmax(last / temperature)
            next_id = int(np.random.choice(len(vocab), p=probs))

        next_word = vocab[next_id]

        # Show top-3 guesses
        top3 = np.argsort(probs)[-3:][::-1]
        tip = " | ".join(f"{vocab[i]}:{probs[i]:.1%}" for i in top3)
        print(f"  Step {step + 1}: {next_word:12s} [{tip}]")

        generated += " " + next_word
        if next_word == "END":
            break
        tokens.append(next_id)

    print(f"Complete: '{generated}'\n")


# =============================================================================
# Run demos
# =============================================================================

def main():
    print("Loading model...")
    w = np.load(MODEL_PATH, allow_pickle=True)
    vocab = w["vocab"].tolist()
    print("Model loaded!\n")
    print(f"Size: d_model={w['d_model']}, layers={w['n_layers']}, heads={w['n_heads']}")
    print(f"Vocab ({len(vocab)}): {vocab}\n")

    print("=" * 60)
    print("Tiny GPT demos")
    print("=" * 60 + "\n")

    print("1. Long-range attention (remember 'big'):")
    generate("the big cat sat on the", vocab, w, num_words=3, temperature=0.0)

    print("2. Selective attention (ignore color, use size):")
    generate("the red big cat sat on the", vocab, w, num_words=3, temperature=0.0)

    print("3. Prefer variety after 'and':")
    generate("the cat and the", vocab, w, num_words=3, temperature=0.0)

    print("4. Pattern completion (size-matched object):")
    generate("the small dog ran to the small", vocab, w, num_words=2, temperature=0.0)


if __name__ == "__main__":
    main()
