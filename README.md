# 🧠 Small LLM From Scratch

> **Repository:** [https://github.com/brunotaboada/small-llm-from-scratch](https://github.com/brunotaboada/small-llm-from-scratch)

A **complete, educational** project that builds and trains a GPT-style language model from scratch using PyTorch. Every component — tokenizer, transformer architecture, training loop, and text generation — is implemented with thorough documentation so you can learn how modern LLMs work.

---

## 📁 Project Structure

```
small_llm_from_scratch/
├── config.py            # All hyperparameters and paths (single source of truth)
├── train_tokenizer.py   # Train a BPE tokenizer on the dataset
├── model.py             # Transformer model architecture (from scratch!)
├── dataset.py           # Data loading and preprocessing
├── train.py             # Full training pipeline with validation & checkpointing
├── generate.py          # Text generation with multiple sampling strategies
├── plot_losses.py       # Visualize training/validation loss curves
├── requirements.txt     # Python dependencies
├── tiny_english/        # 20-word pedagogical GPT (algo.monster-style)
│   ├── vocab.py         # Fixed word-level vocabulary
│   ├── data.py          # Synthetic pattern corpus
│   ├── model.py         # TinyEnglishGPT (course weight names)
│   ├── train.py         # Train until capability demos pass
│   ├── export_npz.py  # Export models/tiny_english_gpt.npz
│   └── infer_numpy.py   # NumPy inference (same as course demo)
├── models/
│   └── tiny_english_gpt.npz  # Exported weights for NumPy / browser demos
├── notebooks/
│   └── walkthrough.ipynb  # Interactive Jupyter notebook tutorial
├── data/
│   ├── train.txt        # Training data (TinyStories subset)
│   └── val.txt          # Validation data
├── tokenizer_model/
│   └── tokenizer.json   # Trained BPE tokenizer
└── checkpoints/
    └── *.pt             # Model checkpoints
```

---

## 🐭 Tiny English GPT (20-word pedagogical model)

A second track that matches the [algo.monster Tiny LLM course](https://algo.monster/courses/llm/llm_course_introduction) demo: word-level vocab, synthetic patterns, and NumPy inference from an `.npz` weight file.

```bash
# Train (stops early once the four demos pass — usually ~200 steps on CPU)
python -m tiny_english.train

# Export weights for NumPy / Pyodide-style loaders
python -m tiny_english.export_npz

# Run the course demos in pure NumPy
python -m tiny_english.infer_numpy
```

Expected greedy outputs:

| Prompt | Continuation |
|--------|--------------|
| `the big cat sat on the` | `big mat END` |
| `the red big cat sat on the` | `big mat END` |
| `the cat and the` | `dog END` |
| `the small dog ran to the small` | `house END` |

Config: `d_model=32`, `n_layers=2`, `n_heads=4`, ~26K parameters. Weight keys match the course NumPy loader (`blocks.{i}.attn.W_q.weight`, etc.).

---

## 🏗️ Architecture

We implement a **decoder-only transformer** (the same architecture family as GPT-2, GPT-3, and LLaMA):

```
Input Token IDs
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  Lookup a learned vector for each token
│  + Pos Embedding  │  Add position information (learned)
│  + Dropout        │
└────────┬─────────┘
         │  ×4 Transformer Blocks
┌────────▼─────────┐
│  Pre-LayerNorm    │
│  Multi-Head       │  Each token attends to all PREVIOUS tokens
│  Causal Attention │  (causal mask prevents looking into the future)
│  + Residual       │
├───────────────────┤
│  Pre-LayerNorm    │
│  Feed-Forward     │  2-layer MLP with GELU activation
│  + Residual       │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Final LayerNorm  │
│  Linear → Logits  │  Project back to vocabulary (weight-tied with embedding)
└──────────────────┘
```

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | ~4,000 | BPE vocabulary size |
| `max_seq_len` | 128 | Context window (tokens) |
| `n_layers` | 4 | Transformer blocks |
| `n_heads` | 4 | Attention heads |
| `d_model` | 128 | Embedding dimension |
| `d_ff` | 512 | FFN inner dimension (4×d_model) |
| `dropout` | 0.1 | Dropout rate |
| **Total params** | **~1.2M** | Trains in ~1 min on CPU! |

---

## 🚀 Quick Start

### 0. Clone the Repository

```bash
git clone https://github.com/brunotaboada/small-llm-from-scratch.git
cd small-llm-from-scratch
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset (a 50K subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)) should already be in `data/`. If not:

```python
from datasets import load_dataset
import random

ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
stories = [ex['text'] for i, ex in enumerate(ds) if i < 50000]
random.seed(42); random.shuffle(stories)

split = int(0.9 * len(stories))
with open('data/train.txt', 'w') as f:
    for s in stories[:split]: f.write(s.strip() + '\n\n')
with open('data/val.txt', 'w') as f:
    for s in stories[split:]: f.write(s.strip() + '\n\n')
```

### 3. Train the Tokenizer

```bash
python train_tokenizer.py
```

This trains a BPE (Byte-Pair Encoding) tokenizer with 8,000 tokens and saves it to `tokenizer_model/tokenizer.json`.

### 4. Train the Model

```bash
python train.py
```

Training progress is logged to stdout. To resume from a checkpoint:

```bash
python train.py --resume latest
```

### 5. Generate Text

```bash
# Single prompt
python generate.py --prompt "Once upon a time"

# Interactive mode
python generate.py --interactive

# With custom settings
python generate.py --prompt "The little cat" --temperature 0.5 --top_k 40 --top_p 0.95
```

### 6. Plot Loss Curves

```bash
python plot_losses.py
```

---

## 📖 Key Concepts Explained

### Tokenization (BPE)
Byte-Pair Encoding starts with individual characters, then iteratively merges the most frequent adjacent pairs. This gives a vocabulary of sub-word tokens that balances:
- **Character-level flexibility** — can handle any word, even unseen ones
- **Word-level efficiency** — common words get single tokens

### Causal Self-Attention
The core mechanism that lets each token "attend to" (look at) all previous tokens to gather context. The causal mask ensures token at position *i* can only see positions ≤ *i*, which is essential for left-to-right language generation.

### Pre-Norm Residual Connections
We apply LayerNorm *before* each sublayer (attention, FFN), not after. This "pre-norm" style (used by GPT-2) leads to more stable training gradients.

### Weight Tying
The token embedding matrix and the output projection (lm_head) share the same weights. This reduces parameters and often improves quality.

### Learning Rate Schedule
Linear warmup (stabilizes early training) → Cosine decay (gradual reduction for convergence).

### Sampling Strategies
- **Greedy**: Always pick the most probable token (deterministic but repetitive)
- **Temperature**: Scale logits before softmax (lower = more confident, higher = more random)
- **Top-k**: Only sample from the k most probable tokens
- **Top-p (Nucleus)**: Sample from the smallest set of tokens whose cumulative probability exceeds p

---

## 🔧 Customization

All hyperparameters are in `config.py`. Common tweaks:

```python
# Larger model (needs GPU)
ModelConfig(n_layers=6, n_heads=6, d_model=384, d_ff=1536)

# Longer training
TrainConfig(max_steps=20_000, batch_size=64)

# Different generation style
GenerationConfig(temperature=0.6, top_k=30, top_p=0.85)
```

---

## 📊 Expected Results

After 500 training steps on CPU (~1 minute), the model should:
- Achieve a training loss around 4.0 (down from ~8.0 at start)
- Generate text that shows learned token patterns (not yet fully coherent at this scale)
- Demonstrate clear loss reduction from random initialization

To get better quality text, increase training by editing `config.py`:
```python
# In config.py — scale up for better results:
TrainConfig(max_steps=5000, batch_size=32)      # More training
ModelConfig(d_model=256, n_layers=6, n_heads=6)  # Bigger model
```

With more data, a bigger model, and a GPU, quality improves dramatically. The TinyStories dataset was specifically designed to be learnable by small models.

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) — GPT-2 paper
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [How to Train a New Language Model from Scratch](https://huggingface.co/blog/how-to-train) — Hugging Face blog

---

## 📝 License

This is an educational project. Feel free to use, modify, and learn from it!
