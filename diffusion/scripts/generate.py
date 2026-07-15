"""Generate images from a trained diffusion checkpoint (EMA weights).

Usage:
    .venv/Scripts/python.exe diffusion/scripts/generate.py [--ckpt path] [--n 16]
        [--sampler ddim|ddpm] [--steps 50] [--seed N]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torchvision.utils import save_image

from config import CHECKPOINT_DIR, SAMPLES_DIR, DiffusionConfig
from sampling import ddim_sample, ddpm_sample
from schedule import GaussianDiffusion
from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: latest)")
    parser.add_argument("--n", type=int, default=16, help="number of images")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=None, help="fixed noise seed")
    args = parser.parse_args()

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpts = sorted(CHECKPOINT_DIR.glob("diff_*.pt"))
        if not ckpts:
            sys.exit(f"no checkpoints found in {CHECKPOINT_DIR}")
        ckpt_path = ckpts[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = DiffusionConfig(**ckpt["config"])

    model = UNet(cfg).to(device).eval()
    model.load_state_dict(ckpt["ema"])
    diffusion = GaussianDiffusion(model, cfg.timesteps, cfg.schedule).to(device)
    print(f"loaded EMA weights from {ckpt_path.name} (step {ckpt['step']})")

    if args.sampler == "ddim":
        print(f"sampling {args.n} images with DDIM ({args.steps} steps)...")
        imgs = ddim_sample(diffusion, args.n, device, steps=args.steps, seed=args.seed)
    else:
        print(f"sampling {args.n} images with DDPM ({cfg.timesteps} steps)...")
        imgs = ddpm_sample(diffusion, args.n, device, seed=args.seed)

    out_dir = SAMPLES_DIR / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        save_image(img, out_dir / f"img_{i:03d}.png", normalize=True, value_range=(-1, 1))
    nrow = max(1, int(args.n**0.5))
    grid_path = out_dir / "grid.png"
    save_image(imgs, grid_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"saved {args.n} PNGs + grid to {out_dir}")


if __name__ == "__main__":
    main()
