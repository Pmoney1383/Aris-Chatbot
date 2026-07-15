"""Reverse-process samplers: full DDPM ancestral sampling and accelerated DDIM.

Both take a GaussianDiffusion (for the schedule buffers) and optionally a
model override (e.g. the EMA copy) — otherwise diffusion.model is used.
A fixed `seed` makes the initial noise (and DDPM's per-step noise)
reproducible, for anchor grids compared across checkpoints.
"""

import torch


def _initial_noise(diffusion, batch_size, device, seed):
    cfg_ch = diffusion.model.cfg.channels
    size = diffusion.model.cfg.image_size
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch_size, cfg_ch, size, size, device=device, generator=generator)
    return x, generator


def _predict_x0(diffusion, x_t, t, eps):
    x_0 = (
        diffusion.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t
        - diffusion.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps
    )
    return x_0.clamp(-1.0, 1.0)


@torch.no_grad()
def ddpm_sample(diffusion, batch_size, device, seed=None, model=None):
    """Full T-step ancestral sampling (slow, highest quality)."""
    model = diffusion.model if model is None else model
    was_training = model.training
    model.eval()

    x, generator = _initial_noise(diffusion, batch_size, device, seed)
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        eps = model(x, t)
        x_0 = _predict_x0(diffusion, x, t, eps)
        mean = (
            diffusion.posterior_mean_coef1[step] * x_0
            + diffusion.posterior_mean_coef2[step] * x
        )
        if step > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + (0.5 * diffusion.posterior_log_variance_clipped[step]).exp() * noise
        else:
            x = mean

    model.train(was_training)
    return x


@torch.no_grad()
def ddim_sample(diffusion, batch_size, device, steps=50, eta=0.0, seed=None, model=None):
    """DDIM sampling (Song et al.) — deterministic when eta=0."""
    model = diffusion.model if model is None else model
    was_training = model.training
    model.eval()

    # Subsequence of timesteps from T-1 down to 0, plus a final "-1" meaning
    # fully denoised (alpha_bar = 1).
    times = torch.linspace(-1, diffusion.timesteps - 1, steps + 1).long().tolist()
    time_pairs = list(zip(reversed(times[1:]), reversed(times[:-1])))

    x, generator = _initial_noise(diffusion, batch_size, device, seed)
    for t_cur, t_prev in time_pairs:
        t = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
        eps = model(x, t)
        x_0 = _predict_x0(diffusion, x, t, eps)

        alpha_bar = diffusion.alphas_cumprod[t_cur]
        alpha_bar_prev = (
            diffusion.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.ones((), device=device)
        )
        sigma = eta * (
            ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
            * (1 - alpha_bar / alpha_bar_prev).sqrt()
        )
        # Re-derive the noise direction from the clamped x_0 for stability.
        eps = (x - alpha_bar.sqrt() * x_0) / (1 - alpha_bar).sqrt()

        x = (
            alpha_bar_prev.sqrt() * x_0
            + (1 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * eps
        )
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)

    model.train(was_training)
    return x
