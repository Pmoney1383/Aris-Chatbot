"""DCGAN Generator and Discriminator for 64x64 images, written from scratch.

Generator:     z (z_dim) -> Linear -> 4x4x512 -> four ConvTranspose2d blocks
               (512 -> 256 -> 128 -> 64 -> 3), BatchNorm + ReLU between,
               Tanh output in [-1, 1].
Discriminator: 64x64x3 -> four strided Conv2d blocks (3 -> 64 -> 128 -> 256 -> 512),
               LeakyReLU(0.2), spectral norm on every conv (no BatchNorm — spectral
               norm is D's stabilizer), final Conv to a single logit
               (no sigmoid — use BCEWithLogitsLoss).

Both nets carry a SAGAN-style SelfAttention2d at their 32x32 stage (gamma=0 at
init, so it's a no-op until training learns to use it).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

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


class SelfAttention2d(nn.Module):
    """SAGAN-style self-attention over a 2D feature map.

    gamma starts at 0 so the layer is an exact no-op at initialization
    (output = input); the network learns to blend attention in gradually,
    which keeps early training stable.
    """

    def __init__(self, in_channels: int, use_spectral_norm: bool = False):
        super().__init__()
        query = nn.Conv2d(in_channels, in_channels // 8, 1)
        key = nn.Conv2d(in_channels, in_channels // 8, 1)
        value = nn.Conv2d(in_channels, in_channels, 1)
        for conv in (query, key, value):
            init_weights(conv)
        if use_spectral_norm:
            query, key, value = (spectral_norm(c) for c in (query, key, value))
        self.query, self.key, self.value = query, key, value
        self.gamma = nn.Parameter(torch.zeros(1))  # learned, starts at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # B, HW, C//8
        k = self.key(x).view(B, -1, H * W)                      # B, C//8, HW
        attn = torch.softmax(torch.bmm(q, k), dim=-1)           # B, HW, HW
        v = self.value(x).view(B, -1, H * W)                    # B, C, HW
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class Generator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config
        f = config.g_features  # 512

        # z -> 4x4 feature map
        self.project = nn.Linear(config.z_dim, f * 4 * 4)

        # named attribute so training code can inspect e.g. G.attn.gamma
        self.attn = SelfAttention2d(f // 8)  # at the 32x32 stage (64 channels)

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
            # self-attention at 32x32x64 for global structural coherence
            self.attn,
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


class MinibatchStdDev(nn.Module):
    """Appends the mean batch-wise std as an extra channel, so D can directly
    see intra-batch similarity — a batch of near-identical fakes has an obvious
    statistical signature (std ~ 0) that D can penalize, countering mode
    collapse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        std = x.std(dim=0, unbiased=False)            # C, H, W - std across the batch
        mean_std = std.mean().expand(B, 1, H, W)      # scalar broadcast to B,1,H,W
        return torch.cat([x, mean_std], dim=1)        # concat as extra channel


def _sn_conv(*args, **kwargs) -> nn.Module:
    """Conv2d with DCGAN init applied BEFORE the spectral_norm wrap — the
    parametrized .weight is a computed property and can't be initialized
    in-place afterwards."""
    conv = nn.Conv2d(*args, **kwargs)
    init_weights(conv)
    return spectral_norm(conv)


class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        f = config.d_features  # 64

        # named attribute so training code can inspect e.g. D.attn.gamma
        # NOTE: the 32x32 stage in this D has 64 channels (the first conv is
        # what downsamples to 32x32); 128 channels only appear at 16x16.
        self.attn = SelfAttention2d(f, use_spectral_norm=True)

        # Spectral norm on every conv; no BatchNorm (the two aren't combined —
        # spectral norm takes over as D's stabilizer).
        self.blocks = nn.Sequential(
            # 64x64x3 -> 32x32x64
            _sn_conv(config.channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # self-attention at 32x32x64, mirroring the generator's placement
            self.attn,
            # 32x32x64 -> 16x16x128
            _sn_conv(f, f * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x128 -> 8x8x256
            _sn_conv(f * 2, f * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x256 -> 4x4x512
            _sn_conv(f * 4, f * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # minibatch stddev adds one channel before the classification head
            MinibatchStdDev(),
            # 4x4x(512+1) -> 1x1x1 logit
            _sn_conv(f * 8 + 1, 1, kernel_size=4, stride=1, padding=0),
        )
        # NOTE: no self.apply(init_weights) here — each conv is initialized
        # inside _sn_conv before wrapping.

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
