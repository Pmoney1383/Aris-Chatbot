"""
Diffusion Transformer (DiT) denoiser for latent flow matching.

    noisy latent (B, 4, h, w) --patchify 2x2--> tokens (B, N, D)
    24 x [ adaLN self-attention (2D RoPE)  ->  cross-attention to T5 tokens  ->  adaLN MLP ]
    --> final adaLN layer --> unpatchify --> velocity (B, 4, h, w)

Conditioning follows PixArt-alpha: the caption enters through cross-attention;
timestep + pooled caption form one vector c that drives every block's adaLN
through a single shared projection (adaLN-single) plus a small per-block
learned table, which keeps the model near 400M parameters. 2D rotary
position embeddings let one set of weights serve every aspect bucket and carry
from 256px to 512px. Residual branches are zero-initialised, so the untrained
network starts as the identity. forward(x, t, text, text_mask) returns the
predicted velocity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.RMSNorm):
    """Casts its weight to the input dtype so bf16 inputs use the fused kernel."""

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


def rope_2d(h: int, w: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables (h*w, head_dim/2): the first half of each head's rotation
    pairs encodes the row position, the second half the column position."""
    quarter = head_dim // 4
    freqs = theta ** (-torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    angles = torch.cat([ys.reshape(-1, 1) * freqs, xs.reshape(-1, 1) * freqs], dim=1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, N, head_dim), rotating adjacent pairs (x0, x1), (x2, x3), ..."""
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = RMSNorm(dim // heads, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm = RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = RMSNorm(dim // heads, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y, y_mask):
        B, N, D = x.shape
        L = y.shape[1]
        hd = D // self.heads
        q = self.q_norm(self.q(x).view(B, N, self.heads, hd).transpose(1, 2))
        k, v = self.kv(y).view(B, L, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, self.k_norm(k), v, attn_mask=y_mask[:, None, None, :])
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross = CrossAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(self, x, t6, y, y_mask, rope):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (self.scale_shift_table[None] + t6).chunk(6, dim=1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a), rope)
        x = x + self.cross(self.norm2(x), y, y_mask)
        return x + gate_m * self.mlp(modulate(self.norm3(x), shift_m, scale_m))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        """t in [0, 1] (flow-matching time), embedded on a 0-1000 scale."""
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.float() * 1000)[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class DiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.hidden_size, cfg.patch_size
        self.x_embed = nn.Conv2d(cfg.in_channels, d, kernel_size=p, stride=p)
        self.t_embed = TimestepEmbedder(d)
        self.y_proj = nn.Sequential(nn.Linear(cfg.text_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.y_pool = nn.Sequential(nn.Linear(cfg.text_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(DiTBlock(d, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_table = nn.Parameter(torch.randn(2, d) / d ** 0.5)
        self.final_proj = nn.Linear(d, p * p * cfg.in_channels)
        self.grad_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        w = self.x_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embed.bias)
        for mlp in (self.t_embed.mlp, self.y_pool):
            nn.init.normal_(mlp[0].weight, std=0.02)
            nn.init.normal_(mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        for blk in self.blocks:  # every residual branch starts at zero
            nn.init.zeros_(blk.attn.proj.weight)
            nn.init.zeros_(blk.cross.proj.weight)
            nn.init.zeros_(blk.mlp[2].weight)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, t, text, text_mask):
        B, C, H, W = x.shape
        p = self.cfg.patch_size
        h, w = H // p, W // p
        tokens = self.x_embed(x).flatten(2).transpose(1, 2)                        # (B, h*w, D)
        maskf = text_mask.to(text.dtype)[..., None]
        pooled = (text * maskf).sum(1) / maskf.sum(1).clamp(min=1)
        c = self.t_embed(t) + self.y_pool(pooled)                                  # (B, D)
        t6 = self.t_block(c).view(B, 6, -1)
        y = self.y_proj(text)
        rope = rope_2d(h, w, self.cfg.hidden_size // self.cfg.num_heads, self.cfg.rope_theta, x.device)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                tokens = checkpoint(blk, tokens, t6, y, text_mask, rope, use_reentrant=False)
            else:
                tokens = blk(tokens, t6, y, text_mask, rope)
        shift, scale = (self.final_table[None] + c[:, None]).chunk(2, dim=1)
        out = self.final_proj(modulate(self.final_norm(tokens), shift, scale))    # (B, h*w, p*p*C)
        out = out.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(B, C, H, W)


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on biases, norms, embeddings tables and modulation tables."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith("_table") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
