"""
Run the tiny GPT forward pass in plain NumPy.

Needs models/tiny_english_gpt.npz (create it with: python train.py)

  python infer.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS_FILE = Path(__file__).resolve().parent / "models" / "tiny_english_gpt.npz"


def _gelu(x: np.ndarray) -> np.ndarray:
    # tanh approximation of GELU
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + 1e-5) + beta


def _attend(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    n_heads: int,
    allow: np.ndarray,
) -> np.ndarray:
    """Scaled dot-product attention over past tokens only (`allow` is 0/1 mask)."""
    t, d = x.shape
    hdim = d // n_heads

    # Linear maps were saved as PyTorch [out, in]; multiply on the right after .T
    q = (x @ wq).reshape(t, n_heads, hdim).transpose(1, 0, 2)
    k = (x @ wk).reshape(t, n_heads, hdim).transpose(1, 0, 2)
    v = (x @ wv).reshape(t, n_heads, hdim).transpose(1, 0, 2)

    logits = (q @ k.transpose(0, 2, 1)) / np.sqrt(hdim)
    logits = logits + (1.0 - allow) * (-1e9)
    weights = _softmax(logits, axis=-1)
    y = (weights @ v).transpose(1, 0, 2).reshape(t, d)
    return y @ wo


def _mlp(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    return _gelu(x @ w1 + b1) @ w2 + b2


def _one_layer(
    x: np.ndarray, w: np.lib.npyio.NpzFile, layer: int, n_heads: int, allow: np.ndarray
) -> np.ndarray:
    p = f"blocks.{layer}"
    # pre-norm attention + residual
    a = _layernorm(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"])
    x = x + _attend(
        a,
        w[f"{p}.attn.W_q.weight"].T,
        w[f"{p}.attn.W_k.weight"].T,
        w[f"{p}.attn.W_v.weight"].T,
        w[f"{p}.attn.W_o.weight"].T,
        n_heads,
        allow,
    )
    # pre-norm feed-forward + residual
    a = _layernorm(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"])
    return x + _mlp(
        a,
        w[f"{p}.ff.linear1.weight"].T,
        w[f"{p}.ff.linear1.bias"],
        w[f"{p}.ff.linear2.weight"].T,
        w[f"{p}.ff.linear2.bias"],
    )


def numpy_forward(token_ids: list[int], w: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return logits for every position: shape (seq, vocab)."""
    t = len(token_ids)
    x = w["token_embed.weight"][token_ids] + w["pos_embed.weight"][:t]
    allow = np.tril(np.ones((t, t)))
    n_layers, n_heads = int(w["n_layers"]), int(w["n_heads"])
    for i in range(n_layers):
        x = _one_layer(x, w, i, n_heads, allow)
    x = _layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
    return x @ w["lm_head.weight"].T


def continue_prompt(
    prompt: str,
    w: np.lib.npyio.NpzFile,
    vocab: list[str],
    steps: int = 3,
    temperature: float = 0.0,
) -> None:
    """Print greedy (or sampled) continuations one token at a time."""
    ids = [vocab.index(tok) for tok in prompt.split() if tok in vocab]
    if not ids:
        print(f"skipped (unknown words): {prompt!r}\n")
        return

    print(f"> {prompt}")
    text = prompt
    for step in range(steps):
        logits = numpy_forward(ids, w)[-1]
        if temperature <= 0:
            probs = _softmax(logits)
            nxt = int(np.argmax(logits))
        else:
            probs = _softmax(logits / temperature)
            nxt = int(np.random.choice(len(vocab), p=probs))

        word = vocab[nxt]
        ranked = np.argsort(probs)[-3:][::-1]
        tip = ", ".join(f"{vocab[i]} {probs[i] * 100:.0f}%" for i in ranked)
        print(f"  +{step + 1}  {word:<8}  ({tip})")
        text = f"{text} {word}"
        if word == "END":
            break
        ids.append(nxt)
    print(f"  => {text}\n")


def main() -> None:
    if not WEIGHTS_FILE.exists():
        raise SystemExit(f"missing {WEIGHTS_FILE}\nrun: python train.py")

    print(f"loading {WEIGHTS_FILE.name} ...")
    w = np.load(WEIGHTS_FILE, allow_pickle=True)
    vocab = w["vocab"].tolist()
    print(
        f"ready — dim={int(w['d_model'])}, layers={int(w['n_layers'])}, "
        f"heads={int(w['n_heads'])}, vocab={len(vocab)}\n"
    )

    print("--- demos ---")
    continue_prompt("the big cat sat on the", w, vocab, steps=3)
    continue_prompt("the red big cat sat on the", w, vocab, steps=3)
    continue_prompt("the cat and the", w, vocab, steps=2)
    continue_prompt("the small dog ran to the small", w, vocab, steps=2)


if __name__ == "__main__":
    main()
