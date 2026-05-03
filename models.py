"""Model registry + Week 1 baseline (bigram).

Each model exposes the same interface:
    forward(idx)               -> logits (B, T, vocab_size)
    forward(idx, targets)      -> (logits, loss)  where loss is a scalar
    generate(idx, max_new)     -> (B, T + max_new) sampled token ids

That uniform contract lets `train.py` and `sample.py` work with whichever
model we've built without conditional plumbing. Week 2+ adds `GPT` here.

The bigram is a deliberately stupid baseline: it predicts the next token
from *only* the previous token, with no context. Loss should drop from
log(vocab_size) ≈ 4.17 to ~2.5 on TinyShakespeare. If it doesn't, the data
pipeline is broken, not the model -- which is exactly what a baseline is for.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class BigramLM(nn.Module):
    """A vocab_size x vocab_size lookup table. That's the entire model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Each row of this table IS the logit vector for "next token given
        # current token i". No transformer, no attention, no positions.
        self.token_logits = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_logits(idx)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature        # only the last step matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx


# Registry: train.py picks a class by config.model_name. Week 2 adds "gpt".
MODELS: dict[str, type[nn.Module]] = {
    "bigram": BigramLM,
}


def build_model(config: GPTConfig) -> nn.Module:
    if config.model_name not in MODELS:
        raise ValueError(
            f"unknown model_name={config.model_name!r}; "
            f"options are {list(MODELS)}"
        )
    return MODELS[config.model_name](config)
