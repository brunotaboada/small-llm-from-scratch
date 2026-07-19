"""
Train TinyEnglishGPT on the synthetic pattern corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW

from tiny_english.data import build_dataset, iter_batches
from tiny_english.model import TinyConfig, TinyEnglishGPT
from tiny_english.vocab import END_ID, VOCAB, encode

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "tiny_english" / "checkpoints"
MODELS_DIR = ROOT / "models"


def evaluate_demos(model: TinyEnglishGPT, device: torch.device) -> dict[str, str]:
    """Run the four course demos greedily; return prompt → continuation."""
    demos = [
        ("the big cat sat on the", 3),
        ("the red big cat sat on the", 3),
        ("the cat and the", 3),
        ("the small dog ran to the small", 2),
    ]
    results = {}
    model.eval()
    for prompt, n in demos:
        ids = encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=n, temperature=0.0, end_id=END_ID)
        words = [VOCAB[i] for i in out[0].tolist()]
        generated = " ".join(words)
        results[prompt] = generated
    return results


def demos_pass(results: dict[str, str]) -> bool:
    """
    Check that the course behaviors emerged.

    We check key next words rather than the full string, so minor END
    placement differences still count as success.
    """
    checks = [
        ("the big cat sat on the", ["big", "mat"]),
        ("the red big cat sat on the", ["big", "mat"]),
        ("the cat and the", ["dog"]),
        ("the small dog ran to the small", ["house"]),
    ]
    for prompt, expected_words in checks:
        generated = results[prompt]
        # Tokens after the prompt
        continuation = generated[len(prompt) :].strip().split()
        for i, word in enumerate(expected_words):
            if i >= len(continuation) or continuation[i] != word:
                return False
    return True


def train(
    steps: int = 800,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    device: str = "cpu",
) -> Path:
    torch.manual_seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TinyConfig()
    model = TinyEnglishGPT(cfg).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    examples = build_dataset(max_seq_len=cfg.max_seq_len, repeats=40, seed=seed)
    print(f"Model params: {model.count_parameters():,}")
    print(f"Training examples: {len(examples):,}")
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")

    step = 0
    best_path = CHECKPOINT_DIR / "best.pt"
    last_loss = None
    passed = False

    while step < steps:
        for inputs, targets in iter_batches(
            examples, batch_size=batch_size, shuffle=True, seed=seed + step
        ):
            model.train()
            x = torch.tensor(inputs, dtype=torch.long, device=device)
            y = torch.tensor(targets, dtype=torch.long, device=device)

            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            last_loss = loss.item()
            step += 1

            if step % 50 == 0 or step == 1:
                results = evaluate_demos(model, device)
                ok = demos_pass(results)
                print(f"step {step:4d}  loss={last_loss:.4f}  demos_ok={ok}")
                for prompt, gen in results.items():
                    print(f"  '{prompt}' → '{gen}'")

                # Always save latest
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                    "loss": last_loss,
                    "demo_results": results,
                    "demos_ok": ok,
                    "vocab": VOCAB,
                }
                torch.save(ckpt, CHECKPOINT_DIR / "latest.pt")

                if ok:
                    torch.save(ckpt, best_path)
                    print(f"✓ Demos passed — saved {best_path}")
                    passed = True
                    break

            if step >= steps:
                break
        if passed:
            break

    if not best_path.exists():
        # Save whatever we have so export still works
        results = evaluate_demos(model, device)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "step": step,
                "loss": last_loss,
                "demo_results": results,
                "demos_ok": demos_pass(results),
                "vocab": VOCAB,
            },
            best_path,
        )
        print(f"Saved final checkpoint (demos may not all pass): {best_path}")

    # Write a small JSON summary
    summary = {
        "steps": step,
        "loss": last_loss,
        "demos_ok": passed or demos_pass(evaluate_demos(model, device)),
        "checkpoint": str(best_path),
    }
    (CHECKPOINT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny English GPT")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    path = train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    print(f"Done. Checkpoint: {path}")


if __name__ == "__main__":
    main()
