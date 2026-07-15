"""Single source of truth for diffusion hyperparameters and paths."""

from dataclasses import dataclass
from pathlib import Path

# diffusion/ project root (this file lives in diffusion/src/)
DIFFUSION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DIFFUSION_ROOT.parent

# Training data: reuse the GAN's preprocessed cat tensor directly — do not
# duplicate the ~800MB file into diffusion/data/.
DATA_FILE = REPO_ROOT / "vision" / "data" / "processed" / "cats_64.pt"
DATA_DIR = DIFFUSION_ROOT / "data"
CHECKPOINT_DIR = DIFFUSION_ROOT / "checkpoints"
SAMPLES_DIR = DIFFUSION_ROOT / "samples"
LOGS_DIR = DIFFUSION_ROOT / "logs"


@dataclass
class DiffusionConfig:
    image_size: int = 64
    channels: int = 3

    base_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 4)
    num_res_blocks: int = 3
    attn_resolutions: tuple = (16, 8)
    dropout: float = 0.1

    timesteps: int = 1000
    schedule: str = "cosine"

    batch_size: int = 96
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    max_steps: int = 300_000
    sample_every: int = 2000
    ckpt_every: int = 5000
    mixed_precision: str = "bf16"
    # Keep checkpointing ON: without it this U-Net needs ~13.2GB for the
    # backward pass even at batch 48, which sits at the edge of the 16GB card
    # and intermittently tips into the driver's sysmem fallback (~6-30x
    # slowdown, measured). Checkpointed throughput is ~121 imgs/s regardless
    # of batch size (GPU-saturated), so batch 96 costs nothing extra.
    gradient_checkpointing: bool = True
