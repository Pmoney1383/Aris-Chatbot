"""Upscale images with Real-ESRGAN (RealESRGAN_x4plus pretrained weights).

Generic over any folder (or single file) of images — works on GAN samples,
diffusion samples, or anything else.

Usage:
    .venv/Scripts/python.exe vision/scripts/upscale.py --input PATH --outdir DIR [--scale 4]

Weights auto-download on first run into the realesrgan weights cache
(<package>/weights/ via basicsr's load_file_from_url).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

X4PLUS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def build_upsampler(scale: int = 4):
    """Create a RealESRGANer wrapping the x4plus RRDBNet (auto-downloads weights)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,                 # native model scale; `outscale` at enhance() time handles the rest
        model_path=X4PLUS_URL,   # RealESRGANer downloads to its weights cache if missing
        model=model,
        tile=0,
        half=torch.cuda.is_available(),  # fp16 on CUDA
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    sys.exit(f"input not found: {input_path}")


def upscale_folder(input_path: Path, outdir: Path, scale: int = 4, upsampler=None) -> float:
    """Upscale every image under input_path into outdir. Returns elapsed seconds.

    RealESRGANer only processes one image per enhance() call (no batch API),
    so this loops per-image with a progress line.
    """
    if upsampler is None:
        upsampler = build_upsampler(scale)

    images = collect_images(input_path)
    if not images:
        sys.exit(f"no images found in {input_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for i, path in enumerate(images, 1):
        img = np.array(Image.open(path).convert("RGB"))
        output, _ = upsampler.enhance(img, outscale=scale)
        out_path = outdir / f"{path.stem}_{scale}x{path.suffix}"
        Image.fromarray(output).save(out_path)
        print(f"  [{i}/{len(images)}] {path.name} "
              f"({img.shape[1]}x{img.shape[0]} -> {output.shape[1]}x{output.shape[0]})")
    elapsed = time.perf_counter() - start
    print(f"upscaled {len(images)} images to {outdir} in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-ESRGAN upscale a file or folder of images")
    parser.add_argument("--input", type=Path, required=True, help="input image file or folder")
    parser.add_argument("--outdir", type=Path, required=True, help="output folder")
    parser.add_argument("--scale", type=int, default=4, help="upscale factor (default 4)")
    args = parser.parse_args()

    upscale_folder(args.input, args.outdir, scale=args.scale)


if __name__ == "__main__":
    main()
