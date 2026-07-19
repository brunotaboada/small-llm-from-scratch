# Tiny GPT

One file: train a tiny transformer, then run it in plain NumPy.

```bash
pip install -r requirements.txt
python tiny_gpt.py          # demos (trains once if no .npz yet)
python tiny_gpt.py --train  # force retrain
```

| Prompt | Continues with |
|--------|----------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

`d_model=32`, 2 layers, 4 heads. Same building blocks as GPT — just tiny.
