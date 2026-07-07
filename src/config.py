"""Single source of truth for all Aris hyperparameters.

Every script imports from here. No magic numbers anywhere else.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent  # project root


@dataclass
class ModelConfig:
    vocab_size: int = 50257          # tiktoken gpt2 vocab
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072                 # 4 * d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_idx: int = 0                 # not used by tiktoken but kept for compat


@dataclass
class TrainConfig:
    # Paths
    shard_dir: Path = field(default_factory=lambda: ROOT / "data" / "shards")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "sample-10BT"   # ~10B token pre-sampled subset
    token_budget: int = 1_500_000_000    # 1.5B tokens for Stage 0
    shard_size: int = 100_000_000        # 100M tokens per shard (uint16 = ~200MB each)

    # Training
    batch_size: int = 12                 # micro-batch; 5080 BF16 hits a perf cliff above this
    grad_accum_steps: int = 96           # effective batch ~0.75M tokens
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    max_steps: int = 1908               # ~1.5B tokens at batch 8, accum 96, seq 1024
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_acc_every: int = 10           # full-vocab argmax is expensive; sample it
    compile_model: bool = True          # default inductor backend: Triton kernel fusion
    amp_dtype: str = "float16"          # faster than bfloat16 on this 5080/PyTorch stack
    fused_optimizer: bool = True        # small measured speedup when supported

    # Checkpointing / eval
    save_every: int = 300                # save checkpoint every N steps
    eval_every: int = 50                # run val loss every N steps
    val_fraction: float = 0.005          # fraction of shards held out for val
    val_batches: int = 20                # fixed number of val batches per eval

    # DataLoader
    num_workers: int = 2


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.2
    repetition_window: int = 20          # penalize tokens seen in last N generated
    max_new_tokens: int = 300
