"""
Fixed 20-word vocabulary matching the algo.monster Tiny English GPT.

Word order is the token ID order — do not rearrange without re-training.
"""

VOCAB = [
    "the",
    "cat",
    "dog",
    "sat",
    "ran",
    "on",
    "mat",
    "house",
    "a",
    "big",
    "small",
    "quickly",
    "slowly",
    "and",
    "is",
    "red",
    "blue",
    "to",
    "PAD",
    "END",
]

WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
ID_TO_WORD = {i: w for i, w in enumerate(VOCAB)}

PAD_ID = WORD_TO_ID["PAD"]
END_ID = WORD_TO_ID["END"]
VOCAB_SIZE = len(VOCAB)


def encode(text: str) -> list[int]:
    """Encode space-separated words to token IDs. Unknown words are dropped."""
    return [WORD_TO_ID[w] for w in text.split() if w in WORD_TO_ID]


def decode(token_ids: list[int]) -> str:
    """Decode token IDs to a space-separated string (skips PAD)."""
    return " ".join(ID_TO_WORD[i] for i in token_ids if i != PAD_ID)
