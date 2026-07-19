"""
train.py — Train the tiny GPT and save models/tiny_english_gpt.npz

Kept as simple as possible:
  - 20-word vocab
  - Hand-written training sentences (size / color / variety patterns)
  - Small PyTorch transformer (so we get free backprop)
  - Save weights as .npz for tiny_gpt.py (NumPy) to load

Run:  python train.py
Then: python tiny_gpt.py
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vocab (word index = token id)
# ---------------------------------------------------------------------------
VOCAB = [
    "the", "cat", "dog", "sat", "ran", "on", "mat", "house", "a", "big",
    "small", "quickly", "slowly", "and", "is", "red", "blue", "to", "PAD", "END",
]
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
PAD, END = WORD2ID["PAD"], WORD2ID["END"]
V = len(VOCAB)

# Model size (tiny on purpose)
D = 32          # embedding size
N_LAYERS = 2
N_HEADS = 4
D_FF = 128      # feed-forward hidden size
MAX_LEN = 16

OUT = Path(__file__).parent / "models" / "tiny_english_gpt.npz"


# ---------------------------------------------------------------------------
# Training sentences (the patterns the demos need)
# ---------------------------------------------------------------------------
def make_sentences():
    sizes = ["big", "small"]
    colors = ["red", "blue"]
    animals = ["cat", "dog"]
    sents = []

    # 1) Size matching: big → big mat / house, small → small mat / house
    for size in sizes:
        for animal in animals:
            sents.append(f"the {size} {animal} sat on the {size} mat")
            sents.append(f"the {size} {animal} ran to the {size} house")

    # 2) Color + size: color is noise, size still predicts the object
    for color in colors:
        for size in sizes:
            for animal in animals:
                sents.append(f"the {color} {size} {animal} sat on the {size} mat")
                sents.append(f"the {color} {size} {animal} ran to the {size} house")

    # 3) Color only: mixed endings so color alone is not predictive
    for color in colors:
        for animal in animals:
            sents.append(f"the {color} {animal} sat on the mat")
            sents.append(f"the {color} {animal} sat on the house")
            sents.append(f"the {color} {animal} ran to the mat")
            sents.append(f"the {color} {animal} ran to the house")

    # 4) Variety: "cat and the dog" more often than "cat and the cat"
    for a, b in [("cat", "dog"), ("dog", "cat")]:
        sents.extend([f"the {a} and the {b}"] * 5)
    sents.append("the cat and the cat")
    sents.append("the dog and the dog")

    return sents


def encode_pair(sentence):
    """Turn a sentence into (input_ids, target_ids) for next-word prediction."""
    ids = [WORD2ID[w] for w in sentence.split()] + [END]
    inp = ids[:-1]
    tgt = ids[1:]
    # Pad to MAX_LEN
    inp += [PAD] * (MAX_LEN - len(inp))
    tgt += [PAD] * (MAX_LEN - len(tgt))
    return inp[:MAX_LEN], tgt[:MAX_LEN]


# ---------------------------------------------------------------------------
# Model (same weight names tiny_gpt.py expects)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
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
        # Causal mask: True = blocked (future)
        mask = torch.triu(torch.ones(MAX_LEN, MAX_LEN), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm1(x)
        d_k = D // N_HEADS

        q = self.attn.W_q(h).view(B, T, N_HEADS, d_k).transpose(1, 2)
        k = self.attn.W_k(h).view(B, T, N_HEADS, d_k).transpose(1, 2)
        v = self.attn.W_v(h).view(B, T, N_HEADS, d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_out = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.attn.W_o(attn_out)

        h = self.norm2(x)
        x = x + self.ff.linear2(F.gelu(self.ff.linear1(h)))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(V, D, padding_idx=PAD)
        self.pos_embed = nn.Embedding(MAX_LEN, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

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

    @torch.no_grad()
    def greedy(self, prompt, n_words):
        """Greedy generate n_words from a text prompt. Used to check demos."""
        ids = [WORD2ID[w] for w in prompt.split()]
        for _ in range(n_words):
            x = torch.tensor([ids[-MAX_LEN:]], dtype=torch.long)
            logits, _ = self(x)
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            if next_id == END:
                break
        return " ".join(VOCAB[i] for i in ids)


# ---------------------------------------------------------------------------
# Demo checks (same 4 prompts as tiny_gpt.py)
# ---------------------------------------------------------------------------
DEMOS = [
    ("the big cat sat on the", ["big", "mat"]),
    ("the red big cat sat on the", ["big", "mat"]),
    ("the cat and the", ["dog"]),
    ("the small dog ran to the small", ["house"]),
]


def demos_ok(model):
    model.eval()
    for prompt, want in DEMOS:
        out = model.greedy(prompt, n_words=3)
        cont = out[len(prompt):].strip().split()
        if cont[: len(want)] != want:
            return False
    return True


def show_demos(model):
    model.eval()
    for prompt, _ in DEMOS:
        print(f"  '{prompt}' → '{model.greedy(prompt, 3)}'")


# ---------------------------------------------------------------------------
# Train + save .npz
# ---------------------------------------------------------------------------
def save_npz(model):
    """Write weights with the names tiny_gpt.py reads."""
    arrays = {
        "vocab": np.array(VOCAB, dtype=object),
        "d_model": np.array(D),
        "n_layers": np.array(N_LAYERS),
        "n_heads": np.array(N_HEADS),
    }
    for name, tensor in model.state_dict().items():
        if name.endswith("mask"):
            continue
        arrays[name] = tensor.detach().cpu().numpy()
    if "lm_head.weight" not in arrays:
        arrays["lm_head.weight"] = arrays["token_embed.weight"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **arrays)
    print(f"Saved {OUT}")


def main():
    torch.manual_seed(42)
    random.seed(42)

    # Build a small dataset (repeat sentences so we see them often)
    pairs = []
    for _ in range(40):
        for s in make_sentences():
            pairs.append(encode_pair(s))
    random.shuffle(pairs)
    print(f"Examples: {len(pairs)}, params coming up...")

    model = TinyGPT()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    step = 0
    batch_size = 32
    while step < 800:
        random.shuffle(pairs)
        for i in range(0, len(pairs) - batch_size + 1, batch_size):
            batch = pairs[i : i + batch_size]
            x = torch.tensor([b[0] for b in batch])
            y = torch.tensor([b[1] for b in batch])

            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if step % 50 == 0 or step == 1:
                print(f"step {step:4d}  loss={loss.item():.4f}")
                show_demos(model)
                if demos_ok(model):
                    print("Demos passed!")
                    save_npz(model)
                    return

            if step >= 800:
                break

    print("Finished steps; saving anyway.")
    save_npz(model)


if __name__ == "__main__":
    main()
