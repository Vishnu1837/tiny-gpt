# tiny-gpt

A small GPT, built from scratch in PyTorch and trained on a domain-specific corpus. Phase 2 of a from-scratch ML series; the predecessor [tiny-autograd](https://github.com/Vishnu1837/tiny-autograd) implements the autograd machinery this project takes for granted.

**Status: Week 1 — data pipeline + bigram baseline.** Multi-head attention, transformer blocks, and ablations land in subsequent weeks.

## Plan

| Week | Goal |
|------|------|
| 1 | TinyShakespeare loader, char tokenizer, bigram baseline, training loop |
| 2 | Scaled dot-product + multi-head attention, single transformer block |
| 3 | Stack into full GPT, train to coherent generation |
| 4+ | Ablation grid (pre/post norm, residual, LN, positional encoding, heads, tied embeds) |
| Final | Train on a personal corpus, write up the results |

## Quick Start

```bash
git clone https://github.com/Vishnu1837/tiny-gpt.git
cd tiny-gpt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Train the bigram baseline on TinyShakespeare (~2 minutes on CPU)
python train.py

# Generate from the trained model
python sample.py "ROMEO:" --max-new 500
```

## Project Structure

```
config.py     — GPTConfig + TrainConfig dataclasses (single source of truth)
tokenizer.py  — Character-level tokenizer
data.py       — TinyShakespeare downloader + windowed Dataset
models.py     — Model registry; bigram baseline (Week 1), GPT (Week 2+)
train.py      — Training loop with cosine LR schedule + periodic eval
sample.py     — Load a checkpoint and generate
```

## License

[MIT](LICENSE)
