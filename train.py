"""
Train a tiny word-level GPT and save models/tiny_english_gpt.npz

  python train.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_FILE = Path(__file__).resolve().parent / "models" / "tiny_english_gpt.npz"

WORDS = [
    "the", "cat", "dog", "sat", "ran", "on", "mat", "house", "a", "big",
    "small", "quickly", "slowly", "and", "is", "red", "blue", "to", "PAD", "END",
]
TO_ID = {w: i for i, w in enumerate(WORDS)}
PAD_ID, END_ID = TO_ID["PAD"], TO_ID["END"]
VOCAB_SIZE = len(WORDS)

DIM = 32
LAYERS = 2
HEADS = 4
FF_DIM = 128
CTX = 16


def build_corpus() -> list[str]:
    """Hand-written sentences covering size, color noise, and 'and' variety."""
    out: list[str] = []
    for size in ("big", "small"):
        for animal in ("cat", "dog"):
            out.append(f"the {size} {animal} sat on the {size} mat")
            out.append(f"the {size} {animal} ran to the {size} house")
            for color in ("red", "blue"):
                out.append(f"the {color} {size} {animal} sat on the {size} mat")
                out.append(f"the {color} {size} {animal} ran to the {size} house")

    for color in ("red", "blue"):
        for animal in ("cat", "dog"):
            out.append(f"the {color} {animal} sat on the mat")
            out.append(f"the {color} {animal} sat on the house")
            out.append(f"the {color} {animal} ran to the mat")
            out.append(f"the {color} {animal} ran to the house")

    for left, right in (("cat", "dog"), ("dog", "cat")):
        out.extend([f"the {left} and the {right}"] * 5)
    out.append("the cat and the cat")
    out.append("the dog and the dog")
    return out


def to_train_example(sentence: str) -> tuple[list[int], list[int]]:
    """Next-word pairs, padded to CTX. PAD targets are ignored in the loss."""
    ids = [TO_ID[w] for w in sentence.split()] + [END_ID]
    x, y = ids[:-1], ids[1:]
    pad = CTX - len(x)
    if pad > 0:
        x, y = x + [PAD_ID] * pad, y + [PAD_ID] * pad
    return x[:CTX], y[:CTX]


class AttnFFBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attn = nn.ModuleDict(
            {
                "W_q": nn.Linear(DIM, DIM, bias=False),
                "W_k": nn.Linear(DIM, DIM, bias=False),
                "W_v": nn.Linear(DIM, DIM, bias=False),
                "W_o": nn.Linear(DIM, DIM, bias=False),
            }
        )
        self.norm2 = nn.LayerNorm(DIM)
        self.ff = nn.ModuleDict(
            {
                "linear1": nn.Linear(DIM, FF_DIM),
                "linear2": nn.Linear(FF_DIM, DIM),
            }
        )
        causal = torch.triu(torch.ones(CTX, CTX), diagonal=1).bool()
        self.register_buffer("causal", causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        head = DIM // HEADS
        h = self.norm1(x)
        q = self.attn.W_q(h).view(b, t, HEADS, head).transpose(1, 2)
        k = self.attn.W_k(h).view(b, t, HEADS, head).transpose(1, 2)
        v = self.attn.W_v(h).view(b, t, HEADS, head).transpose(1, 2)
        dots = (q @ k.transpose(-2, -1)) / (head**0.5)
        dots = dots.masked_fill(self.causal[:t, :t], float("-inf"))
        mix = (F.softmax(dots, dim=-1) @ v).transpose(1, 2).contiguous().view(b, t, DIM)
        x = x + self.attn.W_o(mix)
        h = self.norm2(x)
        return x + self.ff.linear2(F.gelu(self.ff.linear1(h)))


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, DIM, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(CTX, DIM)
        self.blocks = nn.ModuleList(AttnFFBlock() for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        x = self.token_embed(idx) + self.pos_embed(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss


def export_weights(model: TinyGPT) -> None:
    blob: dict[str, np.ndarray] = {
        "vocab": np.array(WORDS, dtype=object),
        "d_model": np.array(DIM),
        "n_layers": np.array(LAYERS),
        "n_heads": np.array(HEADS),
    }
    for key, tensor in model.state_dict().items():
        if key.endswith("causal"):
            continue
        blob[key] = tensor.detach().cpu().numpy()
    if "lm_head.weight" not in blob:
        blob["lm_head.weight"] = blob["token_embed.weight"]
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS_FILE, **blob)
    print(f"wrote {WEIGHTS_FILE}")


def passes_smoke_tests(model: TinyGPT) -> bool:
    cases = [
        ("the big cat sat on the", ("big", "mat")),
        ("the red big cat sat on the", ("big", "mat")),
        ("the cat and the", ("dog",)),
        ("the small dog ran to the small", ("house",)),
    ]
    model.eval()
    with torch.no_grad():
        for prompt, expect in cases:
            ids = [TO_ID[w] for w in prompt.split()]
            for _ in range(3):
                logits, _ = model(torch.tensor([ids[-CTX:]]))
                nxt = int(logits[0, -1].argmax())
                ids.append(nxt)
                if nxt == END_ID:
                    break
            got = tuple(WORDS[i] for i in ids[len(prompt.split()) :])
            if got[: len(expect)] != expect:
                return False
    return True


def train(max_steps: int = 800) -> None:
    torch.manual_seed(42)
    random.seed(42)
    data = [to_train_example(s) for _ in range(40) for s in build_corpus()]
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n = sum(p.numel() for p in model.parameters())
    print(f"training tiny gpt ({n:,} parameters)...")

    step = 0
    while step < max_steps:
        random.shuffle(data)
        for i in range(0, len(data) - 31, 32):
            batch = data[i : i + 32]
            xb = torch.tensor([p[0] for p in batch])
            yb = torch.tensor([p[1] for p in batch])
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step:4d}  loss {loss.item():.3f}")
                if passes_smoke_tests(model):
                    print("  smoke tests ok — exporting weights")
                    export_weights(model)
                    return
            if step >= max_steps:
                break
    export_weights(model)


if __name__ == "__main__":
    train()
