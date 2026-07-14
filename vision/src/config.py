"""Single source of truth for all vision (DCGAN) hyperparameters.

Every script under vision/ imports from here. No magic numbers anywhere else.
"""

from dataclasses import dataclass, field
from pathlib import Path

VISION_ROOT = Path(__file__).parent.parent  # vision/


@dataclass
class GANConfig:
    # data
    image_size: int = 64
    channels: int = 3
    dataset_name: str = "huggan/cats"

    # model
    z_dim: int = 128
    g_features: int = 512     # generator base feature maps
    d_features: int = 64      # discriminator base feature maps

    # training
    batch_size: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple = (0.5, 0.999)
    label_smooth: float = 0.9   # real labels = 0.9 not 1.0
    max_steps: int = 100_000
    sample_every: int = 500     # save sample grid every N steps
    ckpt_every: int = 2000
    num_workers: int = 2

    # paths
    raw_dir: Path = field(default_factory=lambda: VISION_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: VISION_ROOT / "data" / "processed")
    checkpoint_dir: Path = field(default_factory=lambda: VISION_ROOT / "checkpoints")
    samples_dir: Path = field(default_factory=lambda: VISION_ROOT / "samples")
    log_dir: Path = field(default_factory=lambda: VISION_ROOT / "logs")

    @property
    def data_file(self) -> Path:
        return self.processed_dir / f"cats_{self.image_size}.pt"
