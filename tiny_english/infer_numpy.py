"""
NumPy inference for TinyEnglishGPT — same building blocks as the course demo.

Loads models/tiny_english_gpt.npz and runs the four capability demos.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "tiny_english_gpt.npz"


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return weight * x_norm + bias


def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads, mask=None):
    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    Q = (x @ W_q).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = (x @ W_k).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = (x @ W_v).reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (1.0 - mask) * -1e9

    attn = softmax(scores, axis=-1)
    out = attn @ V
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


def feed_forward(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, weights, block_idx, n_heads, mask):
    prefix = f"blocks.{block_idx}"

    W_q = weights[f"{prefix}.attn.W_q.weight"].T
    W_k = weights[f"{prefix}.attn.W_k.weight"].T
    W_v = weights[f"{prefix}.attn.W_v.weight"].T
    W_o = weights[f"{prefix}.attn.W_o.weight"].T

    x_norm = layer_norm(
        x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"]
    )
    x = x + multi_head_attention(x_norm, W_q, W_k, W_v, W_o, n_heads, mask)

    W1 = weights[f"{prefix}.ff.linear1.weight"].T
    b1 = weights[f"{prefix}.ff.linear1.bias"]
    W2 = weights[f"{prefix}.ff.linear2.weight"].T
    b2 = weights[f"{prefix}.ff.linear2.bias"]

    x_norm = layer_norm(
        x, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"]
    )
    x = x + feed_forward(x_norm, W1, b1, W2, b2)
    return x


def full_transformer(tokens, weights):
    seq_len = len(tokens)
    x = weights["token_embed.weight"][tokens]
    x = x + weights["pos_embed.weight"][:seq_len]

    mask = np.tril(np.ones((seq_len, seq_len)))

    n_layers = int(weights["n_layers"])
    n_heads = int(weights["n_heads"])
    for layer_idx in range(n_layers):
        x = transformer_block(x, weights, layer_idx, n_heads, mask)

    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = x @ weights["lm_head.weight"].T
    return logits


def generate_text(text, vocab, weights, num_words=3, temperature=1.0):
    words = text.split()
    tokens = [vocab.index(w) for w in words if w in vocab]

    if not tokens:
        print(f"No words recognized in '{text}'\n")
        return

    print(f"Input: '{text}'")
    generated = text

    for i in range(num_words):
        logits = full_transformer(tokens, weights)

        if temperature == 0:
            next_token = int(np.argmax(logits[-1]))
            probs = softmax(logits[-1])
        else:
            probs = softmax(logits[-1] / temperature)
            next_token = int(np.random.choice(len(vocab), p=probs))

        next_word = vocab[next_token]
        top_3 = np.argsort(probs)[-3:][::-1]
        prob_str = " | ".join(
            [f"{vocab[idx]}:{probs[idx]:.1%}" for idx in top_3]
        )
        print(f"  Step {i + 1}: {next_word:12s} [{prob_str}]")

        if next_word == "END":
            generated += " " + next_word
            break

        generated += " " + next_word
        tokens.append(next_token)

    print(f"Complete: '{generated}'\n")
    return generated


def load_weights(path: Path):
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nTrain + export first.")
    data = np.load(path, allow_pickle=True)
    # np.load returns NpzFile; keep as dict-like access
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy Tiny English GPT demos")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print("Loading model...")
    weights = load_weights(args.model)
    vocab = weights["vocab"].tolist()
    print("✓ Model loaded!\n")
    print(
        f"Model: {weights['d_model']} dimensions, "
        f"{weights['n_layers']} layers, {weights['n_heads']} heads"
    )
    print(f"Vocabulary ({len(vocab)} words): {vocab}\n")

    print("=" * 60)
    print("Tiny GPT: Multi-Word Text Generation")
    print("=" * 60)
    print()

    print("1. Long-range attention (remembering 'big' from 6 words back):")
    generate_text(
        "the big cat sat on the", vocab, weights, num_words=3, temperature=0.0
    )

    print("2. Selective attention (ignoring color, focusing on size):")
    generate_text(
        "the red big cat sat on the",
        vocab,
        weights,
        num_words=3,
        temperature=0.0,
    )

    print("3. Context awareness (preferring variety):")
    generate_text(
        "the cat and the", vocab, weights, num_words=3, temperature=0.0
    )

    print("4. Pattern completion (size-matched destination):")
    generate_text(
        "the small dog ran to the small",
        vocab,
        weights,
        num_words=2,
        temperature=0.0,
    )

    print("=" * 60)
    print("What You Just Saw:")
    print("=" * 60)
    print("✓ Multi-head attention finding relevant context")
    print("✓ Causal masking preventing future information leakage")
    print("✓ Feed-forward networks processing gathered context")
    print("✓ Layer normalization stabilizing values")
    print("✓ Residual connections preserving information flow")
    print("✓ Autoregressive generation (each word feeds into next)")


if __name__ == "__main__":
    main()
