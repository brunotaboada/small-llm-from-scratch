# Tiny English GPT

A 20-word pedagogical decoder-only transformer — same building blocks as the [algo.monster Tiny LLM course](https://algo.monster/courses/llm/llm_course_introduction), trained from scratch and runnable in pure NumPy.

## Quick start

```bash
pip install -r requirements.txt

# Run the demos with the shipped weights
python -m tiny_english.infer_numpy

# Or retrain + export
python -m tiny_english.train
python -m tiny_english.export_npz
python -m tiny_english.infer_numpy
```

## What it learns

| Prompt | Continuation |
|--------|--------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

## Layout

```
tiny_english/
  vocab.py          # Fixed 20-word vocabulary
  data.py           # Synthetic pattern corpus
  model.py          # TinyEnglishGPT (PyTorch)
  train.py          # Train until demos pass
  export_npz.py  # Export models/tiny_english_gpt.npz
  infer_numpy.py    # NumPy inference demos
models/
  tiny_english_gpt.npz
```

## Model

- Vocab: 20 words (`the`, `cat`, …, `PAD`, `END`)
- `d_model=32`, `n_layers=2`, `n_heads=4` (~26K params)
- Pre-norm blocks, multi-head causal attention, GELU FFN, residual + LayerNorm
- Weight keys match the course NumPy loader
