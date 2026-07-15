"""Noise schedule and the GaussianDiffusion forward/reverse process wrapper.

Cosine schedule from Nichol & Dhariwal, "Improved Denoising Diffusion
Probabilistic Models" (2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def _extract(buf, t, shape):
    """Gather buf values at timesteps t, reshaped to broadcast over an image batch."""
    out = buf.gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Wraps a noise-predicting model (U-Net) with DDPM forward/reverse math."""

    def __init__(self, model, timesteps=1000, schedule="cosine"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # \bar{alpha}_{t-1}, with \bar{alpha}_0 = 1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register("betas", betas)
        register("alphas", alphas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0) — forward process
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # x_0 reconstruction from predicted noise
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # q(x_{t-1} | x_t, x_0) — posterior used by the reverse process
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register("posterior_variance", posterior_variance)
        # log-variance clipped because posterior_variance is 0 at t=0
        register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_0, t, noise):
        """Forward process: sample x_t ~ q(x_t | x_0)."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def p_losses(self, x_0, t):
        """Simple DDPM loss: MSE between true and predicted noise."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    # --- Sampling (reverse process) — implemented in sampling.py (Prompt 2) ---

    def p_sample(self, x_t, t):
        raise NotImplementedError("implemented in sampling.py")

    def sample(self, batch_size, device):
        raise NotImplementedError("implemented in sampling.py")


if __name__ == "__main__":
    from config import DiffusionConfig

    cfg = DiffusionConfig()
    betas = cosine_beta_schedule(cfg.timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    for t in (0, 500, 999):
        print(
            f"t={t:4d}  beta={betas[t]:.6f}  alpha={1 - betas[t]:.6f}  "
            f"alphas_cumprod={alphas_cumprod[t]:.6f}"
        )

    assert betas.shape == (cfg.timesteps,)
    assert (betas >= 0.0001).all() and (betas <= 0.9999).all(), "betas out of range"
    assert (betas[1:] >= betas[:-1]).all(), "betas not monotonically increasing"
    assert (alphas_cumprod[1:] <= alphas_cumprod[:-1]).all(), "alphas_cumprod not decreasing"
    assert alphas_cumprod[0] > 0.99 and alphas_cumprod[-1] < 0.01, (
        "alphas_cumprod should go from ~1 toward 0"
    )
    print("schedule self-test passed: betas increasing, alphas_cumprod decreasing toward 0")
