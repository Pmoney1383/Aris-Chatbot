"""Single source of truth for all Aris hyperparameters.

Every script imports from here. No magic numbers anywhere else.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent  # project root


@dataclass
class ModelConfig:
    vocab_size: int = 50257          # tiktoken gpt2 vocab
    d_model: int = 1280
    n_heads: int = 20                # head_dim 64
    n_layers: int = 24
    max_seq_len: int = 1024
    dropout: float = 0.0             # dropout hurts at this scale, disable it
    pad_idx: int = 0                 # not used by tiktoken but kept for compat
    # SwiGLU FFN hidden dim is computed from d_model in src/model.py:
    # int(2/3 * 4 * d_model) rounded to the nearest multiple of 256.


@dataclass
class TrainConfig:
    # Paths
    shard_dir: Path = field(default_factory=lambda: ROOT / "data" / "shards")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "sample-10BT"   # ~10B token pre-sampled subset
    token_budget: int = 5_000_000_000    # 5B tokens for Stage B
    shard_size: int = 100_000_000        # 100M tokens per shard (uint16 = ~200MB each)

    # Stage B diversity mix (80% web / 10% books / 10% conversational).
    # Budgets are per source; weights drive train-time shard sampling.
    fineweb_token_budget: int = 4_000_000_000
    books_token_budget: int = 500_000_000
    conv_token_budget: int = 500_000_000
    # parquet mirror of deepmind/pg19 (the original uses a legacy loading
    # script that datasets>=3 refuses to run)
    books_dataset_name: str = "emozilla/pg19"
    conv_dataset_name: str = "allenai/soda"
    # FineWeb tokens already consumed by Stage 0 — fast-forward past them so
    # Stage B pulls fresh documents from the stream.
    fineweb_skip_tokens: int = 1_500_000_000
    source_weights: dict = field(default_factory=lambda: {
        "fineweb": 0.8,
        "books": 0.1,
        "conv": 0.1,
    })

    # Training
    batch_size: int = 4                  # smaller micro-batch for the larger model
    grad_accum_steps: int = 128          # effective batch ~0.5M tokens
    max_lr: float = 3e-4                 # lower LR for larger model
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 9500                # 9500 * 0.5M = ~4.75B tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_acc_every: int = 10            # full-vocab argmax is expensive; sample it
    compile_model: bool = True           # default inductor backend: Triton kernel fusion
    amp_dtype: str = "bfloat16"          # bfloat16 — stable, no GradScaler needed
    fused_optimizer: bool = True         # small measured speedup when supported

    # Checkpointing / eval
    save_every: int = 500                # save checkpoint every N steps
    eval_every: int = 100                # run val loss every N steps
    val_fraction: float = 0.005          # fraction of shards held out for val
    val_batches: int = 20                # fixed number of val batches per eval

    # DataLoader
    num_workers: int = 2


@dataclass
class SFTConfig:
    """Supervised fine-tuning (chat) stage. Separate from TrainConfig so the
    pretraining setup stays untouched."""

    # Paths
    shard_dir: Path = field(default_factory=lambda: ROOT / "data" / "sft_shards")
    pretrain_checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints" / "sft")
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    log_name: str = "sft_loss_log.csv"

    # Data sources
    oasst1_dataset: str = "OpenAssistant/oasst1"
    oasst2_dataset: str = "OpenAssistant/oasst2"
    wildchat_dataset: str = "allenai/WildChat-1M"
    wildchat_token_budget: int = 300_000_000  # cap the (huge) WildChat stream
    max_conversation_tokens: int = 8192       # skip degenerate mega-conversations
    shard_size: int = 25_000_000              # tokens per SFT shard (uint16 ~50MB)

    # Training
    batch_size: int = 4                       # micro-batch 1 to fit in 16GB VRAM (no shared-memory spill)
    grad_accum_steps: int = 32                # effective batch ~33k tokens
    max_lr: float = 5e-5                      # steering, not training from scratch
    min_lr: float = 5e-6
    warmup_steps: int = 100
    max_steps: int = 3000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_acc_every: int = 10
    compile_model: bool = False               # recompiles blow past 16GB with checkpointing; eager is safer here
    amp_dtype: str = "bfloat16"
    fused_optimizer: bool = True

    # Checkpointing / eval
    save_every: int = 250
    eval_every: int = 100
    val_batches: int = 20

    # DataLoader
    num_workers: int = 2

    # Persona pass (--persona): tiny second SFT on the user's own messages
    persona_file: Path = field(
        default_factory=lambda: ROOT / "data" / "raw" / "clean_tagged_persona.txt"
    )
    persona_checkpoint_dir: Path = field(
        default_factory=lambda: ROOT / "checkpoints" / "sft_persona"
    )
    persona_log_name: str = "sft_persona_loss_log.csv"
    persona_max_lr: float = 1e-5
    persona_min_lr: float = 1e-6
    persona_max_steps: int = 500


@dataclass
class PersonaV2Config:
    """Persona SFT v2: fine-tune the chat-SFT model on synthetic Aris-persona
    conversations (scripts/generate_persona_data.py) plus manual curations."""

    # Paths
    synthetic_file: Path = field(
        default_factory=lambda: ROOT / "data" / "raw" / "persona_synthetic.jsonl"
    )
    curated_file: Path = field(
        default_factory=lambda: ROOT / "data" / "raw" / "persona_curated.jsonl"
    )
    init_checkpoint: Path = field(
        default_factory=lambda: ROOT / "checkpoints" / "sft" / "ckpt_003000.pt"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: ROOT / "checkpoints" / "sft_persona_v2"
    )
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    log_name: str = "sft_persona_v2_loss_log.csv"

    # Data
    val_fraction: float = 0.10               # 90/10 train/val split by pair
    seed: int = 1337                         # pair shuffle before the split

    # Training
    batch_size: int = 12
    grad_accum_steps: int = 16
    max_lr: float = 5e-6
    min_lr: float = 1e-6
    warmup_steps: int = 20
    max_steps: int = 80
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"

    # Checkpointing / eval
    save_every: int = 10
    eval_every: int = 10
    val_batches: int = 20
    early_stop_evals: int = 3                # stop after N consecutive val rises

    # DataLoader
    num_workers: int = 2


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.2
    repetition_window: int = 20          # penalize tokens seen in last N generated
    max_new_tokens: int = 300
