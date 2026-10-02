"""
Rectified-flow objective and sampler.

Noising path:  x_t = (1 - t) * x0 + t * noise,   t in [0, 1]  (t=1 is pure noise)
Target:        v = noise - x0   (the constant velocity along that straight path)
Sampling:      start from noise at t=1 and integrate dx/dt = v back to t=0 with
               Euler steps, using classifier-free guidance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_t(t: torch.Tensor, shift: float) -> torch.Tensor:
    """shift > 1 moves timesteps toward the noisy end (SD3's resolution shift)."""
    return shift * t / (1 + (shift - 1) * t)


def sample_timesteps(n: int, mean: float, std: float, shift: float, device, generator=None) -> torch.Tensor:
    t = torch.sigmoid(torch.randn(n, device=device, generator=generator) * std + mean)
    return shift_t(t, shift)


def flow_loss(model, x0, text, text_mask, t_mean=0.0, t_std=1.0, shift=1.0, t=None, noise=None):
    """Mean squared error between predicted and true velocity. t and noise can be
    passed in to get a deterministic loss (validation)."""
    if t is None:
        t = sample_timesteps(x0.shape[0], t_mean, t_std, shift, x0.device)
    if noise is None:
        noise = torch.randn_like(x0)
    tb = t.view(-1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * noise
    pred = model(xt, t, text, text_mask)
    return F.mse_loss(pred.float(), (noise - x0).float())


@torch.no_grad()
def sample(model, shape, text, text_mask, null_text, null_mask, steps: int, cfg_scale: float,
           shift: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler integration from noise (t=1) to a clean latent (t=0).
    text: (B, L, D) prompt embeddings; null_*: the empty caption, broadcast to B."""
    device = text.device
    B = shape[0]
    x = torch.randn(shape, device=device, generator=generator)
    ts = shift_t(torch.linspace(1, 0, steps + 1, device=device), shift)
    null_text = null_text.expand(B, -1, -1)
    null_mask = null_mask.expand(B, -1)
    L = max(text.shape[1], null_text.shape[1])
    cond = torch.cat([F.pad(text, (0, 0, 0, L - text.shape[1])), F.pad(null_text, (0, 0, 0, L - null_text.shape[1]))])
    mask = torch.cat([F.pad(text_mask, (0, L - text_mask.shape[1])), F.pad(null_mask, (0, L - null_mask.shape[1]))])
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        v = model(torch.cat([x, x]), t.expand(2 * B), cond, mask).float()
        v_cond, v_uncond = v.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + (t_next - t) * v
    return x
