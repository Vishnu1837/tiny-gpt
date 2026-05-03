"""TinyShakespeare loader + windowed Dataset.

`download_tinyshakespeare()` fetches Karpathy's hosted ~1MB version once and
caches it. `CharDataset` slices a long encoded sequence into (input, target)
pairs of fixed `block_size`, where target is input shifted by one position --
the standard next-token prediction setup.

The dataset is *random-access*: __getitem__(i) returns the window starting at
position i. With shuffle=True, the DataLoader picks random starting positions,
which is equivalent to training on overlapping windows in random order. This
is the right move for small corpora -- we get many more training examples
than if we used non-overlapping chunks.
"""

from __future__ import annotations
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download_tinyshakespeare(cache_dir: str | Path = "data") -> str:
    """Download TinyShakespeare into `cache_dir`. Returns the loaded text."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "tinyshakespeare.txt"
    if not path.exists():
        print(f"downloading tinyshakespeare -> {path}")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


class CharDataset(Dataset):
    """Windowed view over a 1D int tensor of token ids."""

    def __init__(self, data: np.ndarray | torch.Tensor, block_size: int):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).long()
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # Need block_size + 1 tokens to form an (x, y) pair.
        return self.data.size(0) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def make_splits(
    text: str,
    tokenizer,
    block_size: int,
    val_frac: float = 0.1,
) -> tuple[CharDataset, CharDataset]:
    """Encode `text`, split into train/val (contiguous, val is the tail)."""
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    n_val = int(len(ids) * val_frac)
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]
    return (
        CharDataset(train_ids, block_size),
        CharDataset(val_ids, block_size),
    )
