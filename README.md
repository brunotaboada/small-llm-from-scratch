# Tiny GPT

Minimal word-level transformer: train in PyTorch, run the forward pass in NumPy.

This project was **inspired by** the idea of a tiny patterned-English GPT
(small vocab, size/color/`and` demos), but the implementation, comments, and
presentation are original. The math (attention, GELU, layer norm) is standard
transformer material from public papers — not unique to any course.

```bash
pip install -r requirements.txt
python tiny_gpt.py          # demos (trains once if weights are missing)
python tiny_gpt.py --train  # retrain
```

| Prompt | Typical continuation |
|--------|----------------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

Config: `dim=32`, 2 layers, 4 heads (~26K params).
