"""From-scratch U-Net noise predictor for DDPM.

Attention is deliberately restricted to low resolutions (16, 8) — full
self-attention at 64x64 caused an 8x slowdown in this repo's GAN because of
the 4096x4096 attention matrix.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer timesteps, transformer-style."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """GroupNorm/SiLU/Conv with FiLM-style timestep conditioning."""

    def __init__(self, in_ch, out_ch, time_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # scale and shift per channel (FiLM)
        self.time_proj = nn.Linear(time_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (flattened H*W sequence)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        # (b, heads, seq, head_dim)
        q, k, v = (u.transpose(-2, -1) for u in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (avoids transposed-conv checkerboard)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        time_dim = ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        self.conv_in = nn.Conv2d(cfg.channels, ch, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [ch]
        in_ch = ch
        resolution = cfg.image_size
        for level, mult in enumerate(cfg.channel_mult):
            out_ch = ch * mult
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                block = nn.ModuleList([ResBlock(in_ch, out_ch, time_dim, cfg.dropout)])
                if resolution in cfg.attn_resolutions:
                    block.append(AttentionBlock(out_ch))
                blocks.append(block)
                in_ch = out_ch
                skip_channels.append(in_ch)
            self.down_blocks.append(blocks)
            if level < len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(in_ch))
                skip_channels.append(in_ch)
                resolution //= 2
            else:
                self.downsamples.append(None)

        # Bottleneck
        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim, cfg.dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim, cfg.dropout)

        # Decoder (mirror, consuming skips)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = ch * mult
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = nn.ModuleList(
                    [ResBlock(in_ch + skip_ch, out_ch, time_dim, cfg.dropout)]
                )
                if resolution in cfg.attn_resolutions:
                    block.append(AttentionBlock(out_ch))
                blocks.append(block)
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamples.append(Upsample(in_ch))
                resolution *= 2
            else:
                self.upsamples.append(None)

        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, cfg.channels, 3, padding=1)

    def _run_block(self, block, h, t_emb):
        def fn(h):
            out = block[0](h, t_emb)
            if len(block) > 1:
                out = block[1](out)
            return out

        if self.cfg.gradient_checkpointing and self.training:
            return checkpoint(fn, h, use_reentrant=False)
        return fn(h)

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.cfg.base_channels))

        h = self.conv_in(x)
        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = self._run_block(block, h, t_emb)
                skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                h = torch.cat([h, skips.pop()], dim=1)
                h = self._run_block(block, h, t_emb)
            if up is not None:
                h = up(h)

        return self.conv_out(F.silu(self.norm_out(h)))


if __name__ == "__main__":
    from config import DiffusionConfig

    cfg = DiffusionConfig()
    model = UNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    x = torch.randn(4, cfg.channels, cfg.image_size, cfg.image_size)
    t = torch.randint(0, cfg.timesteps, (4,))
    model.eval()
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape, f"output shape {out.shape} != input shape {x.shape}"
    print(f"forward pass OK: {tuple(x.shape)} -> {tuple(out.shape)}")
