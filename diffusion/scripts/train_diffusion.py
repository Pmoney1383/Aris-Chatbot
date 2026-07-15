"""DDPM training loop: bf16 autocast, EMA weights, rich progress, auto-resume.

Usage:
    .venv/Scripts/python.exe diffusion/scripts/train_diffusion.py
    .venv/Scripts/python.exe diffusion/scripts/train_diffusion.py --steps 20 --fake-data  # smoke test
"""

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchvision.utils import save_image

from config import CHECKPOINT_DIR, DATA_FILE, LOGS_DIR, SAMPLES_DIR, DiffusionConfig
from sampling import ddim_sample
from schedule import GaussianDiffusion
from unet import UNet

ANCHOR_SEED = 42
ANCHOR_GRID_SIZE = 64
LOG_CSV = LOGS_DIR / "diffusion_loss.csv"


class EMA:
    """Exponential moving average copy of the model, updated every step."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


def load_data(cfg, fake_data):
    if fake_data:
        print("using random stand-in data (--fake-data)")
        data = torch.randn(2048, cfg.channels, cfg.image_size, cfg.image_size)
    else:
        if not DATA_FILE.exists():
            sys.exit(
                f"dataset not found: {DATA_FILE}\n"
                "Run the GAN preprocessing first, or pass --fake-data for a smoke test."
            )
        data = torch.load(DATA_FILE, map_location="cpu")
        print(f"loaded {DATA_FILE.name}: {tuple(data.shape)}")
    return DataLoader(
        TensorDataset(data),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )


def latest_checkpoint():
    ckpts = sorted(CHECKPOINT_DIR.glob("diff_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(step, model, ema, opt, cfg):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"diff_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "opt": opt.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )
    return path


def save_sample_grid(diffusion, ema, step, device):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Sample in chunks of 16 to keep VRAM below the training-step peak; each
    # chunk gets its own fixed seed so the anchor grid is identical every time.
    chunk = 16
    imgs = torch.cat(
        [
            ddim_sample(
                diffusion, chunk, device, steps=50, seed=ANCHOR_SEED + i, model=ema.model
            ).cpu()
            for i in range(0, ANCHOR_GRID_SIZE, chunk)
        ]
    )
    path = SAMPLES_DIR / f"step_{step:06d}.png"
    save_image(imgs, path, nrow=8, normalize=True, value_range=(-1, 1))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--fake-data", action="store_true", help="random stand-in data")
    args = parser.parse_args()

    cfg = DiffusionConfig()
    if args.steps is not None:
        cfg.max_steps = args.steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console = Console()
    console.print(f"device: {device}")

    loader = load_data(cfg, args.fake_data)
    model = UNet(cfg).to(device)
    diffusion = GaussianDiffusion(model, cfg.timesteps, cfg.schedule).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    ckpt_path = latest_checkpoint()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        console.print(f"resumed from {ckpt_path.name} at step {step}")

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"U-Net: {n_params / 1e6:.1f}M params, batch_size={cfg.batch_size}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_log = not LOG_CSV.exists()
    log_file = open(LOG_CSV, "a", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(["step", "loss", "images_seen", "steps_per_sec"])

    autocast = torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=cfg.mixed_precision == "bf16"
    )

    progress = Progress(
        TextColumn("[bold blue]step {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("loss {task.fields[loss]:.4f}"),
        TextColumn("imgs {task.fields[images]:,}"),
        TextColumn("{task.fields[sps]:.2f} it/s"),
        TextColumn("VRAM {task.fields[vram]:.1f}GB"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task(
        "train", total=cfg.max_steps, completed=step, loss=float("nan"),
        images=step * cfg.batch_size, sps=0.0, vram=0.0,
    )

    model.train()
    data_iter = iter(loader)
    last_time = time.time()
    try:
        with progress:
            while step < cfg.max_steps:
                try:
                    (x_0,) = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    (x_0,) = next(data_iter)
                x_0 = x_0.to(device, non_blocking=True)
                t = torch.randint(0, cfg.timesteps, (x_0.shape[0],), device=device)

                with autocast:
                    loss = diffusion.p_losses(x_0, t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                ema.update(model)
                step += 1

                now = time.time()
                sps = 1.0 / max(now - last_time, 1e-9)
                last_time = now
                vram = (
                    torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0.0
                )
                loss_val = loss.item()
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_val}")

                progress.update(
                    task, completed=step, loss=loss_val,
                    images=step * cfg.batch_size, sps=sps, vram=vram,
                )
                log_writer.writerow(
                    [step, f"{loss_val:.6f}", step * cfg.batch_size, f"{sps:.3f}"]
                )

                if step % cfg.sample_every == 0:
                    path = save_sample_grid(diffusion, ema, step, device)
                    console.print(f"saved sample grid {path.name}")
                if step % cfg.ckpt_every == 0:
                    path = save_checkpoint(step, model, ema, opt, cfg)
                    console.print(f"saved checkpoint {path.name}")
    except KeyboardInterrupt:
        console.print("\ninterrupted — saving checkpoint...")
    finally:
        log_file.close()
        if step > 0:
            path = save_checkpoint(step, model, ema, opt, cfg)
            console.print(f"saved checkpoint {path.name} at step {step}")


if __name__ == "__main__":
    main()
