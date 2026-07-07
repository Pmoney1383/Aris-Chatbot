"""Llama-style decoder-only transformer, written from scratch (Stage B).

- RoPE (rotary position embeddings) — no learned position embeddings
- RMSNorm everywhere (pre-norm), no biases on any linear layer
- SwiGLU FFN with LLaMA sizing: hidden = int(2/3 * 4 * d_model) rounded to
  a multiple of 256
- Causal self-attention via F.scaled_dot_product_attention (FlashAttention on CUDA)
- Tied input/output embeddings
- GPT-2 init: std=0.02, residual projections scaled by 1/sqrt(2 * n_layers)
- Optional gradient checkpointing (gradient_checkpointing_enable())
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import ModelConfig


def swiglu_hidden_dim(d_model: int) -> int:
    """LLaMA FFN sizing: 2/3 of 4*d_model, rounded to nearest multiple of 256."""
    h = int(2 / 3 * 4 * d_model)
    return max(256, round(h / 256) * 256)


def apply_rope(q, k, cos, sin):
    """Rotate (q, k) by position-dependent angles.

    q, k: (B, n_heads, T, head_dim); cos, sin: (T, head_dim // 2).
    head_dim is split into consecutive pairs; each pair (x1, x2) is rotated by
    the angle for its frequency index.
    """

    def rotate(x):
        x = x.view(*x.shape[:-1], -1, 2)          # (..., head_dim/2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        out = torch.stack(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
        )
        return out.flatten(-2)

    return rotate(q), rotate(k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj.IS_RESIDUAL_PROJ = True  # flagged for scaled init

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = swiglu_hidden_dim(cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(cfg.d_model, hidden, bias=False)  # up
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)  # down
        self.w3.IS_RESIDUAL_PROJ = True

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # RoPE tables, precomputed to 2x context for headroom (buffers, not params)
        head_dim = cfg.d_model // cfg.n_heads
        pos = torch.arange(cfg.max_seq_len * 2, dtype=torch.float32)
        inv_freq = 10000.0 ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.outer(pos, inv_freq)  # (2*max_seq_len, head_dim/2)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

        self.apply(self._init_weights)
        # GPT-2 paper: scale residual projections by 1/sqrt(2 * n_layers)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for module in self.modules():
            if getattr(module, "IS_RESIDUAL_PROJ", False):
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_num_params(self, verbose=True):
        total = sum(p.numel() for p in self.parameters())
        # embeddings are tied, so lm_head adds nothing extra
        non_emb = total - self.tok_emb.weight.numel()
        if verbose:
            print(f"Model parameters: {total / 1e6:.1f}M total "
                  f"({non_emb / 1e6:.1f}M non-embedding)")
        return total

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.tok_emb(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
