# Tiny GPT

Minimal word-level transformer: train in PyTorch, run the forward pass in NumPy.

Inspired by the idea of a tiny patterned-English GPT (small vocab, size/color/`and`
demos). The code and presentation are original to this project; the math (attention,
GELU, layer norm) is standard transformer material.

```bash
pip install -r requirements.txt
python train.py    # train + write models/tiny_english_gpt.npz
python infer.py    # NumPy demos
```

| Prompt | Typical continuation |
|--------|----------------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

- `train.py` — data, model, training loop, export `.npz`
- `infer.py` — load weights, NumPy forward pass, demos

Config: `dim=32`, 2 layers, 4 heads (~26K params).
