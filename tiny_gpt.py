"""
Tiny English GPT — train a tiny transformer, then run it in plain NumPy.

Same style as the algo.monster course demo.

  python tiny_gpt.py          # train (if needed) + demos
  python tiny_gpt.py --train  # force retrain
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = Path(__file__).parent / "models" / "tiny_english_gpt.npz"

# ---------------------------------------------------------------------------
# Vocab + size
# ---------------------------------------------------------------------------
VOCAB = [
    "the", "cat", "dog", "sat", "ran", "on", "mat", "house", "a", "big",
    "small", "quickly", "slowly", "and", "is", "red", "blue", "to", "PAD", "END",
]
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
PAD, END, V = WORD2ID["PAD"], WORD2ID["END"], len(VOCAB)

D, N_LAYERS, N_HEADS, D_FF, MAX_LEN = 32, 2, 4, 128, 16


# ---------------------------------------------------------------------------
# Training data (simple patterns)
# ---------------------------------------------------------------------------
def make_sentences():
    sents = []
    for size in ["big", "small"]:
        for animal in ["cat", "dog"]:
            sents += [
                f"the {size} {animal} sat on the {size} mat",
                f"the {size} {animal} ran to the {size} house",
            ]
            for color in ["red", "blue"]:
                sents += [
                    f"the {color} {size} {animal} sat on the {size} mat",
                    f"the {color} {size} {animal} ran to the {size} house",
                ]
    for color in ["red", "blue"]:
        for animal in ["cat", "dog"]:
            sents += [
                f"the {color} {animal} sat on the mat",
                f"the {color} {animal} sat on the house",
                f"the {color} {animal} ran to the mat",
                f"the {color} {animal} ran to the house",
            ]
    for a, b in [("cat", "dog"), ("dog", "cat")]:
        sents += [f"the {a} and the {b}"] * 5
    sents += ["the cat and the cat", "the dog and the dog"]
    return sents


def encode_pair(sentence):
    ids = [WORD2ID[w] for w in sentence.split()] + [END]
    inp, tgt = ids[:-1], ids[1:]
    inp += [PAD] * (MAX_LEN - len(inp))
    tgt += [PAD] * (MAX_LEN - len(tgt))
    return inp[:MAX_LEN], tgt[:MAX_LEN]


# ---------------------------------------------------------------------------
# Tiny PyTorch model (only used for training / saving weights)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        # Nested so .npz keys match NumPy: blocks.i.attn.W_q.weight, blocks.i.ff.linear1...
        self.attn = nn.ModuleDict({
            "W_q": nn.Linear(D, D, bias=False),
            "W_k": nn.Linear(D, D, bias=False),
            "W_v": nn.Linear(D, D, bias=False),
            "W_o": nn.Linear(D, D, bias=False),
        })
        self.norm2 = nn.LayerNorm(D)
        self.ff = nn.ModuleDict({
            "linear1": nn.Linear(D, D_FF),
            "linear2": nn.Linear(D_FF, D),
        })
        self.register_buffer("mask", torch.triu(torch.ones(MAX_LEN, MAX_LEN), 1).bool())

    def forward(self, x):
        B, T, _ = x.shape
        h, d_k = self.norm1(x), D // N_HEADS
        q = self.attn.W_q(h).view(B, T, N_HEADS, d_k).transpose(1, 2)
        k = self.attn.W_k(h).view(B, T, N_HEADS, d_k).transpose(1, 2)
        v = self.attn.W_v(h).view(B, T, N_HEADS, d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.attn.W_o(out)
        h = self.norm2(x)
        return x + self.ff.linear2(F.gelu(self.ff.linear1(h)))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(V, D, padding_idx=PAD)
        self.pos_embed = nn.Embedding(MAX_LEN, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=PAD)
        return logits, loss


def save_npz(model):
    """Save with names the NumPy code below expects."""
    arrays = {
        "vocab": np.array(VOCAB, dtype=object),
        "d_model": np.array(D),
        "n_layers": np.array(N_LAYERS),
        "n_heads": np.array(N_HEADS),
    }
    for name, t in model.state_dict().items():
        if name.endswith("mask"):
            continue
        arrays[name] = t.detach().cpu().numpy()
    if "lm_head.weight" not in arrays:
        arrays["lm_head.weight"] = arrays["token_embed.weight"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, **arrays)
    print(f"Saved {MODEL_PATH}")


def check_demos(model):
    """Return True if the 4 course demos pass (greedy)."""
    checks = [
        ("the big cat sat on the", ["big", "mat"]),
        ("the red big cat sat on the", ["big", "mat"]),
        ("the cat and the", ["dog"]),
        ("the small dog ran to the small", ["house"]),
    ]
    model.eval()
    with torch.no_grad():
        for prompt, want in checks:
            ids = [WORD2ID[w] for w in prompt.split()]
            for _ in range(3):
                x = torch.tensor([ids[-MAX_LEN:]])
                logits, _ = model(x)
                nxt = int(logits[0, -1].argmax())
                ids.append(nxt)
                if nxt == END:
                    break
            got = [VOCAB[i] for i in ids[len(prompt.split()):]]
            if got[: len(want)] != want:
                return False
    return True


def train():
    torch.manual_seed(42)
    random.seed(42)
    pairs = [encode_pair(s) for _ in range(40) for s in make_sentences()]
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"Training ({sum(p.numel() for p in model.parameters()):,} params)...")

    step = 0
    while step < 800:
        random.shuffle(pairs)
        for i in range(0, len(pairs) - 31, 32):
            batch = pairs[i : i + 32]
            x = torch.tensor([b[0] for b in batch])
            y = torch.tensor([b[1] for b in batch])
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}  loss={loss.item():.3f}")
                if check_demos(model):
                    print("  demos passed!")
                    save_npz(model)
                    return
            if step >= 800:
                break
    save_npz(model)


# ============================================================================
# Full Transformer Implementation (NumPy — course style)
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
    """Feed-forward network (Module 2, 4)"""
    return gelu(x @ W1 + b1) @ W2 + b2


def transformer_block(x, weights, block_idx, n_heads, mask):
    """Single transformer block (Module 4)"""
    prefix = f"blocks.{block_idx}"

    W_q = weights[f"{prefix}.attn.W_q.weight"].T
    W_k = weights[f"{prefix}.attn.W_k.weight"].T
    W_v = weights[f"{prefix}.attn.W_v.weight"].T
    W_o = weights[f"{prefix}.attn.W_o.weight"].T

    x_norm = layer_norm(x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
    x = x + multi_head_attention(x_norm, W_q, W_k, W_v, W_o, n_heads, mask)

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
    x = weights["token_embed.weight"][tokens]
    x = x + weights["pos_embed.weight"][:seq_len]
    mask = np.tril(np.ones((seq_len, seq_len)))

    n_layers = int(weights["n_layers"])
    n_heads = int(weights["n_heads"])
    for layer_idx in range(n_layers):
        x = transformer_block(x, weights, layer_idx, n_heads, mask)

    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    return x @ weights["lm_head.weight"].T


def generate_text(text, weights, vocab, num_words=3, temperature=0.0):
    """Generate multiple words using FULL transformer"""
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
        prob_str = " | ".join([f"{vocab[idx]}:{probs[idx]:.1%}" for idx in top_3])
        print(f"  Step {i+1}: {next_word:12s} [{prob_str}]")

        generated += " " + next_word
        if next_word == "END":
            break
        tokens.append(next_token)

    print(f"Complete: '{generated}'\n")


# ============================================================================
# Main
# ============================================================================

def run_demos(weights, vocab):
    print("=" * 60)
    print("Tiny GPT: Multi-Word Text Generation")
    print("=" * 60)
    print()

    print("1. Long-range attention (remembering 'big' from 6 words back):")
    generate_text("the big cat sat on the", weights, vocab, num_words=3, temperature=0.0)

    print("2. Selective attention (ignoring color, focusing on size):")
    generate_text("the red big cat sat on the", weights, vocab, num_words=3, temperature=0.0)

    print("3. Context awareness (preferring variety):")
    generate_text("the cat and the", weights, vocab, num_words=3, temperature=0.0)

    print("4. Pattern completion (size-matched destination):")
    generate_text("the small dog ran to the small", weights, vocab, num_words=2, temperature=0.0)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Force retrain")
    args = parser.parse_args()

    if args.train or not MODEL_PATH.exists():
        train()

    print("Loading model...")
    weights = np.load(MODEL_PATH, allow_pickle=True)
    vocab = weights["vocab"].tolist()
    print("✓ Model loaded!\n")
    print(f"Model: {weights['d_model']} dimensions, {weights['n_layers']} layers, {weights['n_heads']} heads")
    print(f"Vocabulary ({len(vocab)} words): {vocab}\n")

    run_demos(weights, vocab)
