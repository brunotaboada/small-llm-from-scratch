# Tiny GPT

A tiny (20-word) transformer you can read end-to-end.

```bash
pip install -r requirements.txt
python train.py       # train + save models/tiny_english_gpt.npz
python tiny_gpt.py    # NumPy inference demos
```

| Prompt | Continues with |
|--------|----------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

- `train.py` — builds the data, trains a small GPT, writes the `.npz`
- `tiny_gpt.py` — loads the `.npz` and runs the transformer in plain NumPy

Same building blocks as a real GPT: embeddings, multi-head attention, causal mask,
feed-forward (GELU), layer norm, residuals — just scaled way down (`d_model=32`,
2 layers, 4 heads).
