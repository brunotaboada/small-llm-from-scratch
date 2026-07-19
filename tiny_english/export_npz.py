"""
Export a trained TinyEnglishGPT checkpoint to tiny_english_gpt.npz.

The NumPy inference code expects PyTorch Linear weights as [out, in]
and applies .T itself — we store them that way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tiny_english.model import TinyConfig, TinyEnglishGPT
from tiny_english.vocab import VOCAB

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "tiny_english" / "checkpoints" / "best.pt"
DEFAULT_OUT = ROOT / "models" / "tiny_english_gpt.npz"


def export_npz(checkpoint_path: Path, out_path: Path) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TinyConfig(**ckpt["config"])
    model = TinyEnglishGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    arrays: dict[str, np.ndarray] = {}

    # Metadata (stored as 0-d / 1-d arrays)
    arrays["vocab"] = np.array(VOCAB, dtype=object)
    arrays["d_model"] = np.array(cfg.d_model)
    arrays["n_layers"] = np.array(cfg.n_layers)
    arrays["n_heads"] = np.array(cfg.n_heads)
    arrays["max_seq_len"] = np.array(cfg.max_seq_len)

    state = model.state_dict()
    # Skip non-parameter buffers (e.g. causal_mask) — NumPy builds its own mask.
    skip_suffixes = ("causal_mask",)
    for name, tensor in state.items():
        if name.endswith(skip_suffixes):
            continue
        arrays[name] = tensor.detach().cpu().numpy()

    # Weight tying: lm_head may share storage with token_embed
    if "lm_head.weight" not in arrays:
        arrays["lm_head.weight"] = arrays["token_embed.weight"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"Exported {len(arrays)} arrays → {out_path}")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    print(f"  vocab={list(VOCAB)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tiny English GPT to .npz")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    export_npz(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
