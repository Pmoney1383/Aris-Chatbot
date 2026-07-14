"""Generate cat images from a trained DCGAN checkpoint.

Loads the latest checkpoint (or --ckpt PATH), samples N images from fresh
random noise, saves individual PNGs to vision/samples/generated/ plus a grid.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision.utils import save_image
from PIL import Image

from src.config import GANConfig
from src.models import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN")
    parser.add_argument("--ckpt", type=Path, default=None, help="checkpoint path (default: latest)")
    parser.add_argument("--n", type=int, default=16, help="number of images to generate")
    args = parser.parse_args()

    config = GANConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpts = sorted(config.checkpoint_dir.glob("gan_*.pt"))
        if not ckpts:
            print(f"No checkpoints found in {config.checkpoint_dir} — train first.")
            sys.exit(1)
        ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)  # rebuild the exact trained architecture
    G = Generator(ckpt_config).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()
    print(f"Loaded {ckpt_path.name} (step {ckpt['step']})")

    with torch.no_grad():
        z = torch.randn(args.n, ckpt_config.z_dim, device=device)
        fake = G(z).float().cpu()  # (n, 3, 64, 64) in [-1, 1]

    out_dir = config.samples_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Denormalize [-1, 1] -> [0, 255] uint8
    imgs = ((fake.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    for i, img in enumerate(imgs):
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(out_dir / f"cat_{i:03d}.png")

    grid_path = out_dir / "grid.png"
    save_image(fake, grid_path, nrow=math.ceil(math.sqrt(args.n)),
               normalize=True, value_range=(-1, 1))
    print(f"Saved {args.n} PNGs and {grid_path.name} to {out_dir}")


if __name__ == "__main__":
    main()
