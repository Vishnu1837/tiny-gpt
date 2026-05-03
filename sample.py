"""Generate text from a trained checkpoint.

    python sample.py "ROMEO:" --max-new 500
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch

from config import GPTConfig
from models import build_model
from tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="\n",
                    help="seed string (default: newline)")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = CharTokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=args.max_new,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
