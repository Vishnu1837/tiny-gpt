"""Smoke test: imports work and the bigram learns on a synthetic corpus.

Runs in <10s, no network, no MNIST-sized data. Just enough to catch obvious
breakage (import errors, shape mismatches, the model not being trainable).
Real attention tests land in Week 2 alongside the attention code.
"""

from __future__ import annotations
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def test_imports():
    """Every top-level module imports cleanly."""
    import config, tokenizer, data, models  # noqa: F401
    print("[OK] imports")


def test_tokenizer_roundtrip():
    from tokenizer import CharTokenizer
    text = "hello world\nthis is a test"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    print(f"[OK] tokenizer roundtrip (vocab={tok.vocab_size})")


def test_bigram_overfits_synthetic():
    """Bigram trained on a tiny repetitive corpus should drive loss well below
    the uniform-prediction baseline within 200 iters. If it doesn't, the
    forward/backward/optimizer plumbing is broken."""
    from config import GPTConfig
    from tokenizer import CharTokenizer
    from data import CharDataset
    from models import build_model

    torch.manual_seed(0)
    np.random.seed(0)

    # Synthetic corpus with strong bigram structure.
    text = ("the quick brown fox jumps over the lazy dog. " * 200)
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, model_name="bigram")
    ds = CharDataset(ids, block_size=cfg.block_size)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    model = build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    initial_loss = None
    final_loss = None
    it = iter(loader)
    for step in range(200):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        _, loss = model(xb, yb)
        if initial_loss is None:
            initial_loss = loss.item()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        final_loss = loss.item()

    uniform = float(np.log(tok.vocab_size))
    print(f"[OK] bigram trains: {initial_loss:.3f} -> {final_loss:.3f}  "
          f"(uniform={uniform:.3f})")
    assert final_loss < initial_loss * 0.6, \
        f"loss didn't drop enough: {initial_loss:.3f} -> {final_loss:.3f}"
    assert final_loss < uniform * 0.7, \
        f"loss above uniform/0.7={uniform*0.7:.3f}: got {final_loss:.3f}"


def test_generation_shape():
    """generate() returns the expected token count."""
    from config import GPTConfig
    from models import build_model

    cfg = GPTConfig(vocab_size=16, block_size=4, model_name="bigram")
    model = build_model(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11), f"expected (1, 11), got {tuple(out.shape)}"
    print("[OK] generation shape")


def main() -> int:
    tests = [
        test_imports,
        test_tokenizer_roundtrip,
        test_bigram_overfits_synthetic,
        test_generation_shape,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
