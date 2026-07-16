"""Two-pass Real-ESRGAN cascade: 64x64 -> 256x256 -> 1024x1024.

Runs upscale.py's logic twice at scale=4 and keeps both stages on disk so
256 vs 1024 quality can be compared directly:

    outdir/stage1_256/   first-pass outputs  (64 -> 256)
    outdir/stage2_1024/  second-pass outputs (256 -> 1024)

Usage:
    .venv/Scripts/python.exe vision/scripts/upscale_cascade.py --input FOLDER --outdir DIR
"""

import argparse
from pathlib import Path

from upscale import build_upsampler, upscale_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="64->256->1024 Real-ESRGAN cascade")
    parser.add_argument("--input", type=Path, required=True, help="folder of 64x64 images")
    parser.add_argument("--outdir", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    stage1_dir = args.outdir / "stage1_256"
    stage2_dir = args.outdir / "stage2_1024"

    upsampler = build_upsampler()  # reuse one model for both passes

    print("stage 1: 64 -> 256")
    t1 = upscale_folder(args.input, stage1_dir, scale=4, upsampler=upsampler)
    print("stage 2: 256 -> 1024")
    t2 = upscale_folder(stage1_dir, stage2_dir, scale=4, upsampler=upsampler)

    print(f"\nstage 1 (64->256):   {t1:.1f}s")
    print(f"stage 2 (256->1024): {t2:.1f}s")
    print(f"total:               {t1 + t2:.1f}s")


if __name__ == "__main__":
    main()
