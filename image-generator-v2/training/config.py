"""
Single source of truth for the text-to-image experiment's hyperparameters:
preprocessing, the DiT architecture, training phases and sampling. Scripts
import from here instead of hardcoding values. Checkpoints store the
ModelConfig they were trained with, so sampling rebuilds the exact model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
CAPTIONS_CSV = DATASET_DIR / "captions.csv"
CLIP_LABELS_DB = DATASET_DIR / "clip_labels.sqlite3"
MANIFEST_DB = DATASET_DIR / "manifest.sqlite3"
ADULT_IMAGE_DIR = DATASET_DIR / "nsfw-images"
ADULT_MANIFEST_CSV = ADULT_IMAGE_DIR / "manifest.csv"
ADULT_CATEGORY = "Adult 18+ Safety Research"
# A stable conditioning token keeps adult content distinguishable from the
# general distribution. Inference still needs its own policy/classifier.
ADULT_CAPTION_PREFIX = "content rating: adult; verified adults only. "
PREP_DIR = DATASET_DIR / "prepared"
TEXT_DIR = PREP_DIR / "text"
# Multi-length captions (Phase 1.5+): one text store per caption length, each
# encoded from its captions.csv column. TEXT_DIR above holds Phase 1's captions.
CAPTION_COLUMNS = {"detailed": "caption", "medium": "caption_medium", "short": "caption_short"}
TEXT_VARIANT_DIRS = {v: PREP_DIR / "text_p15" / v for v in CAPTION_COLUMNS}
CHECKPOINT_DIR = ROOT / "checkpoints" / "dit"
SAMPLE_DIR = ROOT / "samples"
LOG_DIR = ROOT / "logs"


def latent_dir(resolution: int) -> Path:
    return PREP_DIR / f"latents_{resolution}"


VAE_ID = "madebyollin/sdxl-vae-fp16-fix"  # SDXL VAE, patched to be numerically safe in fp16
TEXT_ENCODER_ID = "google/flan-t5-large"
LATENT_CHANNELS = 4
VAE_DOWNSAMPLE = 8

# 512-class aspect buckets (w, h). Sides are multiples of 32 so the 256-class
# buckets (halved) stay multiples of 16 = VAE downsample 8 x patch size 2.
# Pixel counts stay within ~6% of 512*512.
BUCKETS_512 = [
    (512, 512),
    (544, 480), (480, 544),
    (576, 448), (448, 576),
    (608, 416), (416, 608),
    (640, 384), (384, 640),
    (704, 352), (352, 704),
]


def buckets(resolution: int) -> list[tuple[int, int]]:
    return [(w * resolution // 512, h * resolution // 512) for w, h in BUCKETS_512]


@dataclass
class PreprocessConfig:
    max_text_tokens: int = 160      # longest caption is ~140 T5 tokens; only real tokens are stored
    text_batch: int = 64            # the longest captions at 256 per batch overflowed 16 GB
    shard_size: int = 8192          # samples per shard file
    vae_batch: int = 32
    decode_workers: int = 12        # JPEG decode/resize threads


@dataclass
class ModelConfig:
    in_channels: int = LATENT_CHANNELS
    patch_size: int = 2
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    text_dim: int = 1024            # Flan-T5-large d_model
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    resolution: int = 256
    micro_batch: int = 64
    grad_accum: int = 4             # effective batch = micro_batch * grad_accum
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 2000        # optimizer steps
    total_steps: int = 40_000       # optimizer steps (cosine decays to min_lr_ratio * lr)
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999       # per optimizer step
    ema_every: int = 10             # EMA lives on the CPU; update it every N steps
    caption_dropout: float = 0.1    # replace the caption with the empty one, enabling classifier-free guidance
    t_logit_mean: float = 0.0       # timesteps ~ sigmoid(N(mean, std)) (SD3 logit-normal)
    t_logit_std: float = 1.0
    time_shift: float = 1.0         # >1 spends more steps at high noise (useful at higher resolution)
    category_balance_alpha: float = 0.5  # sample weight = category_count ** -alpha; 0 = uniform
    adult_sample_fraction: float | None = 0.05  # target draw share; None uses ordinary balancing
    val_per_mille: int = 2          # ids with id % 1000 < this are held out (~0.2%)
    val_max_samples: int = 2048
    grad_checkpointing: bool = True
    compile: bool = False
    num_workers: int = 6
    log_every: int = 25             # optimizer steps
    save_every: int = 2500          # optimizer steps; also runs validation loss + sample grid
    keep_last: int = 5
    seed: int = 0
    # With no checkpoint of its own, a phase starts from the best.pt of the first of these phases that has one.
    init_from_phase: tuple[str, ...] = ()
    max_samples: int | None = None      # train on only this many samples (the overfit sanity test)
    sample_cfg_scale: float | None = None  # guidance for checkpoint samples; None = SampleConfig.cfg_scale
    # Horizontal-flip augmentation: use the pre-encoded mirrored latent (preprocess.py flips) half the
    # time. Captions mentioning left/right are never flipped. Shards without flips are used as-is.
    hflip: bool = False
    # Multi-length captions: (variant, weight) pairs; each training sample uses one variant picked by
    # weight (normalized), falling back to "detailed" when the sample has no caption of that length.
    # Empty = Phase 1's single caption store (TEXT_DIR). Validation always uses "detailed".
    caption_variants: tuple[tuple[str, float], ...] = ()


MULTI_LENGTH = (("detailed", 0.60), ("medium", 0.20), ("short", 0.15))

# Keys are phase names: `train.py --phase 1.5`, checkpoints in checkpoints/dit/phase1.5/.
TRAIN_PHASES: dict[str, TrainConfig] = {
    # Phase 0: overfit a tiny fixed subset. Loss should fall far below the
    # full-data level and samples should reproduce those training images.
    "0": TrainConfig(
        micro_batch=16,
        grad_accum=1,
        warmup_steps=100,
        total_steps=1500,
        caption_dropout=0.0,
        category_balance_alpha=0.0,
        adult_sample_fraction=None,
        ema_decay=0.99,
        ema_every=1,
        save_every=500,
        log_every=25,
        keep_last=1,
        max_samples=64,
        num_workers=0,  # 4 batches per epoch: re-spawning workers every epoch would dominate
        sample_cfg_scale=1.0,  # no caption dropout here, so the unconditional branch is untrained: no CFG
    ),
    "1": TrainConfig(num_workers=4),  # 6 workers x ~2.4 GB RAM left too little for anything else on 31 GB
    # Phase 1.5: continue at 256px from phase 1's best weights on the general data plus the
    # Phase 1.5 concepts, with flip augmentation and multi-length captions. A fresh,
    # lower-peak cosine schedule. No adult data from here on (general-audience model).
    "1.5": TrainConfig(
        num_workers=4,
        lr=6e-5,
        warmup_steps=1000,
        total_steps=120_000,
        hflip=True,
        init_from_phase=("1",),
        adult_sample_fraction=None,
        caption_variants=MULTI_LENGTH,
    ),
    "2": TrainConfig(
        resolution=512,
        micro_batch=16,
        grad_accum=16,
        lr=5e-5,
        warmup_steps=500,
        total_steps=20_000,
        time_shift=2.0,
        num_workers=4,
        hflip=True,
        init_from_phase=("1.5", "1"),
        adult_sample_fraction=None,
        caption_variants=MULTI_LENGTH,
    ),
}


@dataclass
class SampleConfig:
    steps: int = 30
    cfg_scale: float = 4.5


# Fixed validation prompts and seeds: identical at every checkpoint so progress is comparable.
VALIDATION_PROMPTS = [
    "A dog.",
    "A cat.",
    "A red car.",
    "A mountain landscape.",
    "A beach at sunset.",
    "A bowl of fruit.",
    "A city street.",
    "A bird on a branch.",
]
VALIDATION_SEED = 1234
