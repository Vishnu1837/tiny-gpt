"""Training loop. Corpus-agnostic; model picked by config.model_name.

Run with defaults to train the bigram baseline on TinyShakespeare:
    python train.py

The loop is intentionally minimal -- one optimizer, one cosine LR schedule,
periodic eval on a held-out split, and a sample dump every eval. No DDP,
no mixed precision, no checkpointing yet. Those land in Week 3.
"""

from __future__ import annotations
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GPTConfig, TrainConfig, BIGRAM_DEBUG
from data import download_tinyshakespeare, make_splits
from models import build_model
from tokenizer import CharTokenizer


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Linear warmup + cosine decay to min_lr."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / max(1, cfg.warmup_iters)
    if it >= cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loaders: dict, eval_iters: int, device) -> dict:
    """Average loss over `eval_iters` batches per split."""
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = []
        it = iter(loader)
        for _ in range(eval_iters):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def resolve_device(want: str) -> torch.device:
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def main(model_cfg: GPTConfig | None = None, train_cfg: TrainConfig | None = None):
    model_cfg = model_cfg or GPTConfig(model_name="bigram", block_size=8, n_embd=64)
    train_cfg = train_cfg or BIGRAM_DEBUG

    torch.manual_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    print(f"device: {device}")

    # --- data ---
    text = download_tinyshakespeare()
    tokenizer = CharTokenizer.from_text(text)
    model_cfg.vocab_size = tokenizer.vocab_size
    print(f"corpus: {len(text):,} chars, vocab={tokenizer.vocab_size}")

    train_ds, val_ds = make_splits(text, tokenizer, model_cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size,
                            shuffle=True, num_workers=0, drop_last=True)
    print(f"train: {len(train_ds):,} windows  val: {len(val_ds):,} windows")

    # --- model ---
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_cfg.model_name}  params: {n_params:,}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    # --- training loop ---
    out_dir = Path(train_cfg.out_dir); out_dir.mkdir(exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")

    train_iter = iter(train_loader)
    t0 = time.time()
    for it in range(train_cfg.max_iters):
        # set LR for this step
        lr = get_lr(it, train_cfg)
        for pg in optim.param_groups:
            pg["lr"] = lr

        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optim.step()

        if it % train_cfg.log_interval == 0:
            print(f"iter {it:5d}  loss={loss.item():.4f}  lr={lr:.2e}")

        if it > 0 and (it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1):
            stats = estimate_loss(
                model, {"train": train_loader, "val": val_loader},
                train_cfg.eval_iters, device,
            )
            dt = time.time() - t0
            print(f"  [eval @ iter {it}] train={stats['train']:.4f}  "
                  f"val={stats['val']:.4f}  ({dt:.1f}s)")

            # Sample 200 chars from a single newline as the seed.
            ctx = torch.tensor([[tokenizer.stoi.get("\n", 0)]], device=device)
            sample_ids = model.generate(ctx, max_new_tokens=200, temperature=1.0)[0]
            print("  --- sample ---")
            print("  " + tokenizer.decode(sample_ids.tolist()).replace("\n", "\n  "))
            print("  --------------")

    # save final weights so sample.py can load them
    torch.save({"model_state": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__},
               out_dir / "ckpt.pt")
    print(f"saved checkpoint -> {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
