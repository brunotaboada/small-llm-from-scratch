"""
Tiny English GPT — plain NumPy inference (algo.monster course style).

Train first:  python train.py
Then run:     python tiny_gpt.py
"""

import numpy as np
from pathlib import Path

# Load the pre-trained tiny transformer model
print("Loading model...")
weights = np.load(Path(__file__).parent / "models" / "tiny_english_gpt.npz", allow_pickle=True)
print("✓ Model loaded!\n")

# Show model specifications
vocab = weights["vocab"].tolist()
print(f"Model: {weights['d_model']} dimensions, {weights['n_layers']} layers, {weights['n_heads']} heads")
print(f"Vocabulary ({len(vocab)} words): {vocab}\n")


# ============================================================================
# Full Transformer Implementation
# ============================================================================

def gelu(x):
    """GELU activation (Module 2)"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    """Softmax (Module 2)"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization (Module 2)"""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return weight * x_norm + bias


def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """Multi-head attention (Module 3-4)"""
    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Q, K, V projections
    Q = (x @ W_q).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = (x @ W_k).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = (x @ W_v).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Attention scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (1.0 - mask) * -1e9

    # Attention weights and output
    attn = softmax(scores, axis=-1)
    out = attn @ V
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


def feed_forward(x, W1, b1, W2, b2):
    """Feed-forward network (Module 2, 4)"""
    return gelu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, weights, block_idx, n_heads, mask):
    """Single transformer block (Module 4)"""
    prefix = f"blocks.{block_idx}"

    # Attention (transpose because PyTorch stores weights as [out, in])
    W_q = weights[f"{prefix}.attn.W_q.weight"].T
    W_k = weights[f"{prefix}.attn.W_k.weight"].T
    W_v = weights[f"{prefix}.attn.W_v.weight"].T
    W_o = weights[f"{prefix}.attn.W_o.weight"].T

    x_norm = layer_norm(x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
    x = x + multi_head_attention(x_norm, W_q, W_k, W_v, W_o, n_heads, mask)

    # Feed-forward
    W1 = weights[f"{prefix}.ff.linear1.weight"].T
    b1 = weights[f"{prefix}.ff.linear1.bias"]
    W2 = weights[f"{prefix}.ff.linear2.weight"].T
    b2 = weights[f"{prefix}.ff.linear2.bias"]

    x_norm = layer_norm(x, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
    x = x + feed_forward(x_norm, W1, b1, W2, b2)

    return x


def full_transformer(tokens, weights):
    """Complete transformer inference"""
    seq_len = len(tokens)

    # Embeddings
    x = weights["token_embed.weight"][tokens]
    x = x + weights["pos_embed.weight"][:seq_len]

    # Causal mask
    mask = np.tril(np.ones((seq_len, seq_len)))

    # Transformer blocks
    n_layers = int(weights["n_layers"])
    n_heads = int(weights["n_heads"])
    for layer_idx in range(n_layers):
        x = transformer_block(x, weights, layer_idx, n_heads, mask)

    # Final layer norm + projection
    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = x @ weights["lm_head.weight"].T

    return logits


# ============================================================================
# Predictions
# ============================================================================

def generate_text(text, num_words=3, temperature=1.0):
    """Generate multiple words using FULL transformer"""
    words = text.split()
    tokens = [vocab.index(w) for w in words if w in vocab]

    if not tokens:
        print(f"No words recognized in '{text}'\n")
        return

    print(f"Input: '{text}'")
    generated = text

    for i in range(num_words):
        # Run full transformer
        logits = full_transformer(tokens, weights)

        # Greedy or sampling
        if temperature == 0:
            # Greedy: pick highest
            next_token = np.argmax(logits[-1])
            probs = softmax(logits[-1])
        else:
            # Sample with temperature
            probs = softmax(logits[-1] / temperature)
            next_token = np.random.choice(len(vocab), p=probs)

        next_word = vocab[next_token]

        # Show probabilities for this step
        top_3 = np.argsort(probs)[-3:][::-1]
        prob_str = " | ".join([f"{vocab[idx]}:{probs[idx]:.1%}" for idx in top_3])
        print(f"  Step {i+1}: {next_word:12s} [{prob_str}]")

        # Stop at END
        if next_word == "END":
            generated += " " + next_word
            break

        # Append to sequence
        generated += " " + next_word
        tokens.append(next_token)

    print(f"Complete: '{generated}'\n")


# Try different inputs - demonstrating transformer capabilities
print("=" * 60)
print("Tiny GPT: Multi-Word Text Generation")
print("=" * 60)
print()

print("1. Long-range attention (remembering 'big' from 6 words back):")
generate_text("the big cat sat on the", num_words=3, temperature=0.0)

print("2. Selective attention (ignoring color, focusing on size):")
generate_text("the red big cat sat on the", num_words=3, temperature=0.0)

print("3. Context awareness (preferring variety):")
generate_text("the cat and the", num_words=3, temperature=0.0)

print("4. Pattern completion (size-matched destination):")
generate_text("the small dog ran to the small", num_words=2, temperature=0.0)

print("=" * 60)
print("What You Just Saw:")
print("=" * 60)
print("✓ Multi-head attention (4 heads) finding relevant context")
print("✓ Causal masking preventing future information leakage")
print("✓ Feed-forward networks processing gathered context")
print("✓ Layer normalization stabilizing values")
print("✓ Residual connections preserving information flow")
print("✓ Autoregressive generation (each word feeds into next)")
print("\nThis is a REAL transformer - same building blocks as GPT-4,"
      " just scaled down!")
