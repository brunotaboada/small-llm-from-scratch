"""
Synthetic training patterns for the Tiny English GPT.

Patterns mirror the algo.monster course data so the model learns:
  1. Size matching (long-range attention)
  2. Color as distractor (selective attention)
  3. Variety over repetition (context awareness)
"""

from __future__ import annotations

import random
from typing import Iterator

from tiny_english.vocab import END_ID, PAD_ID, encode


SIZES = ["big", "small"]
COLORS = ["red", "blue"]
ANIMALS = ["cat", "dog"]
VERBS_ON = ["sat"]  # sat on the X
VERBS_TO = ["ran"]  # ran to the X
OBJECTS_ON = {"big": "mat", "small": "mat"}  # both can use mat; size repeats
OBJECTS = {
    ("sat", "big"): "mat",
    ("sat", "small"): "mat",
    ("ran", "big"): "house",
    ("ran", "small"): "house",
}

# Destination objects that match size in the classic demos:
#   big → mat (sat on), house also appears with big in variations
# Course demos:
#   "the big cat sat on the" → "big mat"
#   "the small dog ran to the small" → "house"


def _size_match_sentences() -> list[str]:
    """Pattern 1: size of animal predicts size of destination object."""
    sentences = []
    for size in SIZES:
        for animal in ANIMALS:
            # sat on the {size} mat
            sentences.append(f"the {size} {animal} sat on the {size} mat")
            # ran to the {size} house
            sentences.append(f"the {size} {animal} ran to the {size} house")
            # with adverbs
            for adv in ["quickly", "slowly"]:
                sentences.append(
                    f"the {size} {animal} sat {adv} on the {size} mat"
                )
                sentences.append(
                    f"the {size} {animal} ran {adv} to the {size} house"
                )
    return sentences


def _color_size_sentences() -> list[str]:
    """Pattern 2a: color present but size still predicts the object."""
    sentences = []
    for color in COLORS:
        for size in SIZES:
            for animal in ANIMALS:
                sentences.append(
                    f"the {color} {size} {animal} sat on the {size} mat"
                )
                sentences.append(
                    f"the {color} {size} {animal} ran to the {size} house"
                )
    return sentences


def _color_only_sentences() -> list[str]:
    """Pattern 2b: color alone does not predict destination."""
    # Mix outcomes so color is not predictive.
    sentences = []
    destinations = [
        ("sat on the", "mat"),
        ("sat on the", "house"),
        ("ran to the", "mat"),
        ("ran to the", "house"),
    ]
    for color in COLORS:
        for animal in ANIMALS:
            for prep, obj in destinations:
                sentences.append(f"the {color} {animal} {prep} {obj}")
    return sentences


def _variety_sentences() -> list[str]:
    """Pattern 3: prefer different animals after 'and'."""
    variety = []
    for a, b in [("cat", "dog"), ("dog", "cat")]:
        variety.extend([f"the {a} and the {b}"] * 5)
    # Rare repetition so the model still sees it but prefers variety
    for a in ANIMALS:
        variety.append(f"the {a} and the {a}")
    return variety


def _extra_filler() -> list[str]:
    """Extra short patterns so common words get coverage."""
    sentences = []
    for animal in ANIMALS:
        for size in SIZES:
            sentences.append(f"a {size} {animal} is {size}")
            sentences.append(f"the {animal} is {size}")
        for color in COLORS:
            sentences.append(f"the {animal} is {color}")
            sentences.append(f"a {color} {animal}")
    return sentences


def all_sentences() -> list[str]:
    """Return the full synthetic corpus (unordered)."""
    return (
        _size_match_sentences()
        + _color_size_sentences()
        + _color_only_sentences()
        + _variety_sentences()
        + _extra_filler()
    )


def sentence_to_example(sentence: str, max_seq_len: int) -> tuple[list[int], list[int]]:
    """
    Convert a sentence into (input_ids, target_ids) for next-token prediction.

    Tokens: words + END, padded to max_seq_len.
    input  = tokens[:-1] padded
    target = tokens[1:]  padded (PAD positions ignored in loss)
    """
    ids = encode(sentence) + [END_ID]
    # Need at least 2 tokens for an input/target pair
    assert len(ids) >= 2

    # Truncate if somehow too long (shouldn't happen with this vocab)
    ids = ids[: max_seq_len + 1]

    inp = ids[:-1]
    tgt = ids[1:]

    # Pad to max_seq_len
    pad_len = max_seq_len - len(inp)
    if pad_len > 0:
        inp = inp + [PAD_ID] * pad_len
        tgt = tgt + [PAD_ID] * pad_len

    return inp[:max_seq_len], tgt[:max_seq_len]


def build_dataset(
    max_seq_len: int = 16,
    repeats: int = 20,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Build a list of (input, target) examples.

    `repeats` duplicates the corpus so training sees each pattern many times.
    """
    rng = random.Random(seed)
    sentences = all_sentences()
    examples: list[tuple[list[int], list[int]]] = []
    for _ in range(repeats):
        shuffled = sentences[:]
        rng.shuffle(shuffled)
        for s in shuffled:
            examples.append(sentence_to_example(s, max_seq_len))
    return examples


def iter_batches(
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[tuple[list[list[int]], list[list[int]]]]:
    """Yield (input_batch, target_batch) lists of token-id lists."""
    idxs = list(range(len(examples)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start : start + batch_size]
        if len(batch_idxs) < batch_size:
            continue  # drop incomplete batch
        inputs = [examples[i][0] for i in batch_idxs]
        targets = [examples[i][1] for i in batch_idxs]
        yield inputs, targets
