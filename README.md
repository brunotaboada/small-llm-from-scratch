# Tiny GPT

A tiny word-level transformer written for beginners: train in PyTorch, run every
step of inference in NumPy. Comments explain the “magic numbers” (like `-1e9`
for masking future words).

Inspired by the idea of a patterned-English toy GPT. Implementation and wording
are original; attention / GELU / layer-norm math is standard.

```bash
pip install -r requirements.txt
python train.py    # learn weights → models/tiny_english_gpt.npz
python infer.py    # load .npz and generate
```

| Prompt | Typical continuation |
|--------|----------------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

- `train.py` — vocab, data, model, training (start here)
- `infer.py` — NumPy forward pass; read the `-1e9` note at the top

Config: `dim=32`, 2 layers, 4 heads (~26K params).
