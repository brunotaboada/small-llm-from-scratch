"""
Run the tiny GPT in plain NumPy (no PyTorch needed at inference time).

Who this is for:
  Read this after train.py. Same model ideas, but every multiply is visible.

Needs weights from training:
  python train.py
  python infer.py

Mental picture of one forward pass:
  words → id numbers
       → vectors (embeddings) + position vectors
       → for each layer: attention, then a small MLP
       → scores for every vocab word (logits)
       → pick / sample the next word, append, repeat
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS_FILE = Path(__file__).resolve().parent / "models" / "tiny_english_gpt.npz"

# Why -1e9 below?
# -----------------
# Softmax turns a list of scores into probabilities.
# If one score is enormously negative, e^(that) ≈ 0, so that option gets
# ~0% probability.
#
# We use that trick for the CAUSAL MASK:
#   "word at position i must not look at words after i"
# For forbidden (future) pairs we ADD -1e9 to the attention score.
# After softmax, those future positions get ~0 weight.
#
# -1e9 means "negative one billion" — big enough to wipe out the entry,
# small enough to usually avoid NaNs in float32.
# PyTorch training uses -inf for the same idea; NumPy code often uses -1e9.


def _linear_w(w: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    """
    Weight matrix ready for `x @ W` (row vectors).

    PyTorch `nn.Linear` stores weight as [out_features, in_features].
    NumPy matmul wants [in, out], so we transpose once here — call sites
    never sprinkle `.T` themselves.
    """
    return w[key].T


def _gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU = smooth cousin of ReLU.

    ReLU is: negative → 0, positive → keep.
    GELU is similar but soft around zero (used in GPT-style models).
    This is the common tanh approximation of the true GELU.
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Scores → probabilities that sum to 1 along `axis`.

    We subtract max(x) first so exp() does not explode (same answer, safer math).
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    For each position, make features mean≈0 and variance≈1, then
    scale (gamma) and shift (beta). Stabilizes deep stacks of layers.
    1e-5 inside sqrt avoids division by zero if variance is tiny.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + 1e-5) + beta


def _to_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(time, dim) → (heads, time, head_dim) so each head can attend on its own."""
    t, d = x.shape
    hdim = d // n_heads
    # reshape: split dim into heads; transpose: put heads first for batched matmul
    return x.reshape(t, n_heads, hdim).transpose(1, 0, 2)


def _from_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(heads, time, head_dim) → (time, dim) — merge heads back into one vector."""
    _, t, hdim = x.shape
    return x.transpose(1, 0, 2).reshape(t, n_heads * hdim)


def _attend(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    n_heads: int,
    allow: np.ndarray,
) -> np.ndarray:
    """
    Multi-head causal self-attention.

    For each word:
      Q = "what am I looking for?"
      K = "what do I contain as a searchable key?"
      V = "what value do I pass along if someone attends to me?"

    allow: shape (time, time), 1 = allowed, 0 = forbidden (future).
    We turn 0s into -1e9 so softmax ignores those cells (see file header).

    wq/wk/wv/wo are already [in, out] from `_linear_w` — use as `x @ W`.
    """
    # Project, then split dim across heads: (t, d) → (heads, t, hdim)
    q = _to_heads(x @ wq, n_heads)
    k = _to_heads(x @ wk, n_heads)
    v = _to_heads(x @ wv, n_heads)

    # (Q K^T) / sqrt(hdim)  → raw attention scores per head
    hdim = q.shape[-1]
    logits = (q @ k.transpose(0, 2, 1)) / np.sqrt(hdim)

    # Causal mask: where allow==0, push score way down
    logits = logits + (1.0 - allow) * (-1e9)

    weights = _softmax(logits, axis=-1)  # each row = "how to mix past values"
    # Mix values, then (heads, t, hdim) → (t, d) and project with W_o
    return _from_heads(weights @ v, n_heads) @ wo


def _mlp(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Two-layer network on each word by itself (no mixing across positions)."""
    return _gelu(x @ w1 + b1) @ w2 + b2


def _block_weights(w: np.lib.npyio.NpzFile, layer: int) -> dict[str, np.ndarray]:
    """
    Arrays for one transformer block, keyed by short names.

    Hides the long `blocks.{i}.attn.W_q.weight` strings so `_one_layer`
    can read as: norm → attend → add, then norm → mlp → add.
    """
    p = f"blocks.{layer}"
    return {
        "norm1_w": w[f"{p}.norm1.weight"],
        "norm1_b": w[f"{p}.norm1.bias"],
        "wq": _linear_w(w, f"{p}.attn.W_q.weight"),
        "wk": _linear_w(w, f"{p}.attn.W_k.weight"),
        "wv": _linear_w(w, f"{p}.attn.W_v.weight"),
        "wo": _linear_w(w, f"{p}.attn.W_o.weight"),
        "norm2_w": w[f"{p}.norm2.weight"],
        "norm2_b": w[f"{p}.norm2.bias"],
        "ff_w1": _linear_w(w, f"{p}.ff.linear1.weight"),
        "ff_b1": w[f"{p}.ff.linear1.bias"],
        "ff_w2": _linear_w(w, f"{p}.ff.linear2.weight"),
        "ff_b2": w[f"{p}.ff.linear2.bias"],
    }


def _one_layer(
    x: np.ndarray, w: np.lib.npyio.NpzFile, layer: int, n_heads: int, allow: np.ndarray
) -> np.ndarray:
    """One block: norm→attend→add, then norm→mlp→add."""
    bw = _block_weights(w, layer)

    a = _layernorm(x, bw["norm1_w"], bw["norm1_b"])
    x = x + _attend(a, bw["wq"], bw["wk"], bw["wv"], bw["wo"], n_heads, allow)

    a = _layernorm(x, bw["norm2_w"], bw["norm2_b"])
    return x + _mlp(a, bw["ff_w1"], bw["ff_b1"], bw["ff_w2"], bw["ff_b2"])


def numpy_forward(token_ids: list[int], w: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Full model for a single sequence of token ids.

    Returns logits with shape (seq_len, vocab_size):
      logits[t, v] = score for vocab word v being the next word after position t
    """
    t = len(token_ids)
    ctx = len(w["pos_embed.weight"])
    if t > ctx:
        raise ValueError(
            f"sequence length {t} exceeds context window {ctx} "
            f"(only {ctx} position embeddings were trained)"
        )
    # Embed words, then add a learned vector for each position 0..t-1
    x = w["token_embed.weight"][token_ids] + w["pos_embed.weight"][:t]

    # Lower-triangular ones: position i may look at 0..i only
    # Example for length 4:
    #   1 0 0 0
    #   1 1 0 0
    #   1 1 1 0
    #   1 1 1 1
    allow = np.tril(np.ones((t, t)))

    n_layers, n_heads = int(w["n_layers"]), int(w["n_heads"])
    for i in range(n_layers):
        x = _one_layer(x, w, i, n_heads, allow)

    x = _layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
    # lm_head is also an nn.Linear weight [vocab, dim] → transpose via helper
    return x @ _linear_w(w, "lm_head.weight")


def encode_prompt(
    prompt: str, vocab: list[str], ctx: int
) -> list[int] | None:
    """
    Prompt text → token ids, or None if we should skip this prompt.

    Checks (and explains) three beginner pitfalls:
      empty prompt, unknown words/typos, longer than the context window.
    """
    tokens = prompt.split()
    if not tokens:
        print("skipped (empty prompt)\n")
        return None

    known = set(vocab)
    unknown = sorted({tok for tok in tokens if tok not in known})
    if unknown:
        # Don't silently drop typos — that hides mistakes from beginners.
        print(
            f"skipped (unknown words {unknown}): {prompt!r}\n"
            f"  known vocab: {vocab}\n"
        )
        return None

    ids = [vocab.index(tok) for tok in tokens]
    if len(ids) > ctx:
        print(
            f"skipped (prompt has {len(ids)} words; context window is {ctx}): "
            f"{prompt!r}\n"
        )
        return None
    return ids


def continue_prompt(
    prompt: str,
    w: np.lib.npyio.NpzFile,
    vocab: list[str],
    steps: int = 3,
    temperature: float = 0.0,
) -> None:
    """
    Autoregressive generation: predict one word, append it, repeat.

    temperature:
      0   → greedy (always the top score) — good for demos
      >0  → sample; higher = more random
      We divide logits by temperature before softmax.
      Smaller temperature → sharper distribution → safer/more boring picks.
    """
    ctx = len(w["pos_embed.weight"])
    ids = encode_prompt(prompt, vocab, ctx)
    if ids is None:
        return

    print(f"> {prompt}")
    text = prompt
    for step in range(steps):
        if len(ids) > ctx:
            print(f"  stopped — reached context window ({ctx} words)")
            break
        logits = numpy_forward(ids, w)[-1]  # scores for the NEXT word only
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
        ids.append(nxt)  # feed the new word back in
    print(f"  => {text}\n")


def main() -> None:
    if not WEIGHTS_FILE.exists():
        raise SystemExit(f"missing {WEIGHTS_FILE}\nrun: python train.py first")

    print(f"loading {WEIGHTS_FILE.name} ...")
    w = np.load(WEIGHTS_FILE, allow_pickle=True)
    vocab = w["vocab"].tolist()
    print(
        f"ready — dim={int(w['d_model'])}, layers={int(w['n_layers'])}, "
        f"heads={int(w['n_heads'])}, vocab={len(vocab)}\n"
    )

    print("--- demos (what the model learned) ---")
    print("1) remember size from earlier in the sentence")
    continue_prompt("the big cat sat on the", w, vocab, steps=3)
    print("2) ignore color; still use size")
    continue_prompt("the red big cat sat on the", w, vocab, steps=3)
    print("3) prefer a different animal after 'and'")
    continue_prompt("the cat and the", w, vocab, steps=2)
    print("4) finish a size-matched destination")
    continue_prompt("the small dog ran to the small", w, vocab, steps=2)


if __name__ == "__main__":
    main()
