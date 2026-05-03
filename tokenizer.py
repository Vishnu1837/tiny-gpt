"""Character-level tokenizer.

Build the vocab once from a training corpus, then encode/decode strings to
lists of ints. This is the simplest possible tokenizer -- one token per
character. Plan to upgrade to BPE in a later phase.

Why char-level for now: zero preprocessing, vocab is tiny (~65 for
TinyShakespeare), and the model gets to learn spelling/word boundaries from
scratch -- which is a more honest test of the architecture than handing it
pre-segmented BPE tokens.
"""

from __future__ import annotations
import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        # Sorted so the mapping is deterministic across runs.
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        # Will KeyError if `s` contains a character not seen at fit time.
        # That's intentional -- silent UNK substitution hides corpus bugs.
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
