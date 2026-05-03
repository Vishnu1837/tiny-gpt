"""Single source of truth for model + training hyperparameters.

Keeping these as dataclasses (rather than argparse or YAML) makes ablations
trivial: copy a config, flip one field, train, log. Every ablation in the
README is going to come from a different `GPTConfig` instance.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GPTConfig:
    # --- vocabulary / context ---
    vocab_size: int = 65            # filled in from tokenizer at runtime
    block_size: int = 256           # context length

    # --- architecture ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384               # head_dim = n_embd / n_head = 64, matches GPT-2 small
    dropout: float = 0.1
    bias: bool = False              # GPT-2 uses bias=True; modern models drop it

    # --- ablation toggles (wired into the model in Week 2+) ---
    norm_position: Literal["pre", "post"] = "pre"
    pos_encoding: Literal["learned", "sinusoidal", "rope", "none"] = "learned"
    use_residual: bool = True
    use_layernorm: bool = True
    tie_embeddings: bool = True

    # --- which model to build (lookup key into models.MODELS) ---
    model_name: str = "bigram"      # Week 1 default; "gpt" once Week 2 lands


@dataclass
class TrainConfig:
    # --- optimization ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)      # GPT-3 paper values
    grad_clip: float = 1.0

    # --- schedule ---
    max_iters: int = 5000
    warmup_iters: int = 100
    lr_decay_iters: int = 5000      # cosine decay end
    min_lr: float = 3e-5

    # --- batching ---
    batch_size: int = 64
    eval_interval: int = 250
    eval_iters: int = 50            # batches averaged for val loss
    log_interval: int = 50

    # --- io ---
    out_dir: str = "out"
    seed: int = 1337
    device: str = "auto"            # "cpu" | "cuda" | "auto"


# Sensible defaults for the bigram smoke test (small + fast).
BIGRAM_DEBUG = TrainConfig(
    max_iters=2000,
    eval_interval=200,
    batch_size=32,
    learning_rate=1e-2,            # bigram is convex-ish, can take a high LR
    warmup_iters=0,
    lr_decay_iters=2000,
    min_lr=1e-3,
)
