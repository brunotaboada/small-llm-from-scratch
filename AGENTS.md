# AGENTS.md

## Cursor Cloud specific instructions

This is a self-contained educational project that builds/trains a small GPT-style
LLM from scratch (PyTorch). It is a set of CLI Python scripts — there is no web
service, no automated test suite, and no linter configuration. See `README.md`
for the full pipeline and hyperparameter docs.

### Environment
- Python deps live in a virtualenv at `.venv` (created because system Python is
  PEP 668 "externally managed"). Always invoke scripts with `.venv/bin/python`.
  The startup update script runs `python3 -m venv .venv` + `pip install -r
  requirements.txt`, so `.venv` is refreshed automatically.
- Torch is the CPU build; `torch.cuda.is_available()` is `False`. Training runs
  on CPU and the full default run (500 steps) takes ~1 minute.

### Generated artifacts (NOT in git — see `.gitignore`)
`data/`, `tokenizer_model/`, and `checkpoints/*.pt` are produced by the pipeline
and are excluded from git, but they persist in the VM snapshot. If any are
missing on a fresh VM, regenerate them in order:
1. Dataset: download a 50K TinyStories subset into `data/train.txt` +
   `data/val.txt` (streamed from HuggingFace `roneneldan/TinyStories`). The
   exact snippet is in `README.md` → "Download Dataset". Egress is unrestricted.
2. `.venv/bin/python train_tokenizer.py` → writes `tokenizer_model/tokenizer.json`.
3. `.venv/bin/python train.py` → writes `checkpoints/checkpoint_*.pt` and
   `checkpoints/loss_history.json` (resume with `--resume latest`).

### Running / demonstrating
- Generate text (needs a trained checkpoint):
  `.venv/bin/python generate.py --prompt "Once upon a time" --max_tokens 60`
- Plot loss curves: `.venv/bin/python plot_losses.py`.
- Note: after only 500 steps the ~1.2M-param model produces incoherent text by
  design (the README calls this out). A dropping train/val loss (~8.0 → ~4.4) is
  the correct signal that the environment works — not fluent output.
