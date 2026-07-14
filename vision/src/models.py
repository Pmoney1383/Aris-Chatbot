"""DCGAN Generator and Discriminator for 64x64 images, written from scratch.

Generator:     z (z_dim) -> Linear -> 4x4x512 -> four ConvTranspose2d blocks
               (512 -> 256 -> 128 -> 64 -> 3), BatchNorm + ReLU between,
               Tanh output in [-1, 1].
Discriminator: 64x64x3 -> four strided Conv2d blocks (3 -> 64 -> 128 -> 256 -> 512),
               LeakyReLU(0.2), BatchNorm on all but the first block,
               final Conv to a single logit (no sigmoid — use BCEWithLogitsLoss).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.config import GANConfig


def init_weights(module: nn.Module) -> None:
    """DCGAN paper init: conv weights ~ N(0, 0.02), BatchNorm ~ N(1.0, 0.02)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        f = config.g_features  # 512

        # z -> 4x4 feature map
        self.project = nn.Linear(config.z_dim, f * 4 * 4)

        self.blocks = nn.Sequential(
            # 4x4x512 -> 8x8x256
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f, f // 2, kernel_size=4, stride=2, padding=1, bias=False),
            # 8x8x256 -> 16x16x128
            nn.BatchNorm2d(f // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f // 2, f // 4, kernel_size=4, stride=2, padding=1, bias=False),
            # 16x16x128 -> 32x32x64
            nn.BatchNorm2d(f // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f // 4, f // 8, kernel_size=4, stride=2, padding=1, bias=False),
            # 32x32x64 -> 64x64x3
            nn.BatchNorm2d(f // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f // 8, config.channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.config.g_features
        x = self.project(z).view(-1, f, 4, 4)
        return self.blocks(x)


class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        f = config.d_features  # 64

        self.blocks = nn.Sequential(
            # 64x64x3 -> 32x32x64 (no BatchNorm on the first block)
            nn.Conv2d(config.channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32x64 -> 16x16x128
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x128 -> 8x8x256
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x256 -> 4x4x512
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4x512 -> 1x1x1 logit
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=0),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).view(-1)  # (B,) raw logits


if __name__ == "__main__":
    config = GANConfig()
    G = Generator(config)
    D = Discriminator(config)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator params:     {g_params:,}")
    print(f"Discriminator params: {d_params:,}")

    z = torch.randn(4, config.z_dim)
    fake = G(z)
    logits = D(fake)
    print(f"z {tuple(z.shape)} -> G(z) {tuple(fake.shape)} -> D(G(z)) {tuple(logits.shape)}")
    assert fake.shape == (4, config.channels, config.image_size, config.image_size)
    assert logits.shape == (4,)
    assert fake.min() >= -1.0 and fake.max() <= 1.0
    print("Shape self-test passed.")
