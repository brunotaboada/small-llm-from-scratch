"""
Train a tiny word-level GPT and save models/tiny_english_gpt.npz

Who this is for:
  Someone who has never built a transformer. Read top to bottom.

What "training" means here:
  1. Show the model many short sentences (as numbers).
  2. Ask it to guess the NEXT word at every position.
  3. Measure how wrong it was (loss).
  4. Nudge the weights a little so next time it is less wrong.
  After enough nudges, patterns like "big ... mat" stick.

Run:
  python train.py
Then try demos with:
  python infer.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_FILE = Path(__file__).resolve().parent / "models" / "tiny_english_gpt.npz"

# Every word the model is allowed to know. Index in this list = token id.
# Example: WORDS[0] == "the", WORDS[9] == "big"
WORDS = [
    "the", "cat", "dog", "sat", "ran", "on", "mat", "house", "a", "big",
    "small", "quickly", "slowly", "and", "is", "red", "blue", "to", "PAD", "END",
]
TO_ID = {w: i for i, w in enumerate(WORDS)}
PAD_ID, END_ID = TO_ID["PAD"], TO_ID["END"]
VOCAB_SIZE = len(WORDS)  # 20

# --- model size knobs (keep small so it trains on a laptop in seconds) ---
DIM = 32       # how many numbers represent one word inside the model
LAYERS = 2     # how many transformer blocks to stack
HEADS = 4      # how many attention "perspectives" run in parallel
FF_DIM = 128   # wider hidden size inside the feed-forward MLP (often 4 * DIM)
CTX = 16       # max words in one training example (context window)


def build_corpus() -> list[str]:
    """
    Make the training sentences by hand.

    Three ideas we want the model to pick up:
      1) Size matches: "big cat ... big mat", "small dog ... small house"
      2) Color is noise: "red"/"blue" appear but should NOT decide mat vs house
      3) Variety: after "the cat and the" prefer "dog" more often than "cat"
    """
    out: list[str] = []
    for size in ("big", "small"):
        for animal in ("cat", "dog"):
            out.append(f"the {size} {animal} sat on the {size} mat")
            out.append(f"the {size} {animal} ran to the {size} house")
            for color in ("red", "blue"):
                out.append(f"the {color} {size} {animal} sat on the {size} mat")
                out.append(f"the {color} {size} {animal} ran to the {size} house")

    # Same color, mixed destinations → color alone is a bad predictor
    for color in ("red", "blue"):
        for animal in ("cat", "dog"):
            out.append(f"the {color} {animal} sat on the mat")
            out.append(f"the {color} {animal} sat on the house")
            out.append(f"the {color} {animal} ran to the mat")
            out.append(f"the {color} {animal} ran to the house")

    # "and" variety: different animal 5× more often than the same animal
    for left, right in (("cat", "dog"), ("dog", "cat")):
        out.extend([f"the {left} and the {right}"] * 5)
    out.append("the cat and the cat")
    out.append("the dog and the dog")
    return out


def to_train_example(sentence: str) -> tuple[list[int], list[int]]:
    """
    Turn one sentence into (input, target) for next-word prediction.

    Sentence:  the big cat
    Tokens:    [the, big, cat, END]
    Input x:   [the, big, cat, END] without last  → predict each next word
    Target y:  [big, cat, END] ...

    We pad with PAD up to CTX so every example has the same length
    (needed to stack them into a batch). The loss ignores PAD targets.
    """
    ids = [TO_ID[w] for w in sentence.split()] + [END_ID]
    x, y = ids[:-1], ids[1:]
    pad = CTX - len(x)
    if pad > 0:
        x, y = x + [PAD_ID] * pad, y + [PAD_ID] * pad
    return x[:CTX], y[:CTX]


class AttnFFBlock(nn.Module):
    """
    One transformer block = attention (talk to other words) + MLP (think alone).

    Pre-norm style used here:
      x = x + Attention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))
    The "+ x" parts are residual connections: keep the old signal so training
    stays stable when we stack layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        # Four separate maps: Query, Key, Value, and Output mix.
        # Named this way so the saved .npz keys are easy to read in infer.py.
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
                "linear1": nn.Linear(DIM, FF_DIM),  # expand
                "linear2": nn.Linear(FF_DIM, DIM),  # compress back
            }
        )
        # Causal mask: True means "block this position".
        # Upper triangle = future words. During language modeling we must NOT
        # peek at words that come later (that would be cheating).
        causal = torch.triu(torch.ones(CTX, CTX), diagonal=1).bool()
        self.register_buffer("causal", causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, time, DIM)  e.g. (32, 16, 32)
        b, t, _ = x.shape
        head = DIM // HEADS  # numbers per attention head (32/4 = 8)

        # --- multi-head self-attention ---
        h = self.norm1(x)
        # Split DIM into HEADS parallel attentions of size `head`
        q = self.attn.W_q(h).view(b, t, HEADS, head).transpose(1, 2)
        k = self.attn.W_k(h).view(b, t, HEADS, head).transpose(1, 2)
        v = self.attn.W_v(h).view(b, t, HEADS, head).transpose(1, 2)

        # Dot(Q, K): "how much should word i care about word j?"
        # Divide by sqrt(head) so scores don't get huge as head grows.
        # (Huge scores → softmax becomes almost one-hot → hard to train.)
        dots = (q @ k.transpose(-2, -1)) / (head**0.5)

        # Block future positions. masked_fill puts -inf where causal is True.
        # Softmax(…, -inf, …) → 0 probability on those spots.
        # (In NumPy infer.py we use a large negative like -1e9 for the same idea,
        #  because NumPy has no special -inf path we rely on here.)
        dots = dots.masked_fill(self.causal[:t, :t], float("-inf"))

        mix = (F.softmax(dots, dim=-1) @ v).transpose(1, 2).contiguous().view(b, t, DIM)
        x = x + self.attn.W_o(mix)  # residual

        # --- feed-forward network (same MLP on each word independently) ---
        h = self.norm2(x)
        return x + self.ff.linear2(F.gelu(self.ff.linear1(h)))  # residual


class TinyGPT(nn.Module):
    """
    Full model:
      token embedding + position embedding
      → stack of AttnFFBlock
      → final LayerNorm
      → linear map to vocab scores (logits)
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, DIM, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(CTX, DIM)  # learned "I am at place 0/1/2/..."
        self.blocks = nn.ModuleList(AttnFFBlock() for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        # Weight tying: reuse the same matrix to go words→vectors and vectors→words
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        # Each token id becomes a vector; add a vector for its position
        x = self.token_embed(idx) + self.pos_embed(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (batch, time, vocab)

        loss = None
        if targets is not None:
            # Cross-entropy: "how surprised were we by the true next word?"
            # ignore_index=PAD_ID → padding slots do not affect the loss
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss


def export_weights(model: TinyGPT) -> None:
    """Save arrays so infer.py can load them with NumPy only (no PyTorch)."""
    blob: dict[str, np.ndarray] = {
        "vocab": np.array(WORDS, dtype=object),
        "d_model": np.array(DIM),
        "n_layers": np.array(LAYERS),
        "n_heads": np.array(HEADS),
    }
    for key, tensor in model.state_dict().items():
        if key.endswith("causal"):
            continue  # infer.py builds its own mask
        blob[key] = tensor.detach().cpu().numpy()
    if "lm_head.weight" not in blob:
        blob["lm_head.weight"] = blob["token_embed.weight"]
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS_FILE, **blob)
    print(f"wrote {WEIGHTS_FILE}")


def batches(
    data: list[tuple[list[int], list[int]]], size: int = 32
) -> list[list[tuple[list[int], list[int]]]]:
    """
    Split examples into fixed-size groups for one optimizer step each.

    We only keep full batches (drop a short leftover at the end) so every
    step sees the same batch size — simpler than padding a partial batch.
    """
    return [data[i : i + size] for i in range(0, len(data) - size + 1, size)]


def passes_smoke_tests(model: TinyGPT) -> bool:
    """True when the four teaching demos work with greedy decoding."""
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
                nxt = int(logits[0, -1].argmax())  # greedy = pick highest score
                ids.append(nxt)
                if nxt == END_ID:
                    break
            got = tuple(WORDS[i] for i in ids[len(prompt.split()) :])
            if got[: len(expect)] != expect:
                return False
    return True


def train(max_steps: int = 800, smoke_streak_needed: int = 2) -> None:
    """
    Loop:
      take a batch → forward → loss → backward → update weights
    Every 50 steps, check demos; save .npz after they pass a few times
    in a row (avoids exporting on a one-off lucky check).
    """
    torch.manual_seed(42)
    random.seed(42)
    # Tiny handmade corpus (~dozens of sentences). Repeat it so each training
    # step still sees those patterns often after we shuffle into batches of 32.
    corpus = build_corpus()
    data = [to_train_example(s) for _ in range(40) for s in corpus]
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n = sum(p.numel() for p in model.parameters())
    print(f"training tiny gpt ({n:,} parameters)...")

    step = 0
    smoke_streak = 0
    while step < max_steps:
        random.shuffle(data)
        for batch in batches(data, size=32):
            xb = torch.tensor([p[0] for p in batch])
            yb = torch.tensor([p[1] for p in batch])
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()  # compute gradients
            opt.step()       # apply the nudge
            step += 1
            if step % 50 == 0:
                print(f"  step {step:4d}  loss {loss.item():.3f}")
                if passes_smoke_tests(model):
                    smoke_streak += 1
                    print(
                        f"  smoke tests ok ({smoke_streak}/{smoke_streak_needed})"
                    )
                    if smoke_streak >= smoke_streak_needed:
                        print("  streak met — exporting weights")
                        export_weights(model)
                        return
                else:
                    smoke_streak = 0
            if step >= max_steps:
                break
    export_weights(model)


if __name__ == "__main__":
    train()
