"""Download huggan/cats and preprocess into a single training tensor.

For each image: center crop to square, resize to 64x64, convert to RGB,
normalize to [-1, 1] float32. Saves the stacked tensor to
vision/data/processed/cats_64.pt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from torchvision import transforms

from src.config import GANConfig

FALLBACK_HELP = """
Failed to load the '{name}' dataset from HuggingFace.

Fallback options:
  1. Check your internet connection and retry (downloads resume).
  2. Try authenticating first if you hit a rate limit:
       .venv/Scripts/huggingface-cli.exe login
  3. Use a local folder of cat images instead: put .jpg/.png files in
       vision/data/raw/cats/
     and rerun this script — it will pick them up automatically.
"""


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """Center crop to square, resize, RGB, scale to [-1, 1] float32."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    x = transforms.functional.pil_to_tensor(img).float()  # (3, H, W) in [0, 255]
    return x / 127.5 - 1.0


def load_images(config: GANConfig):
    """Yield PIL images from huggan/cats, or from vision/data/raw/cats/ as fallback."""
    local_dir = config.raw_dir / "cats"
    local_files = sorted(local_dir.glob("*")) if local_dir.exists() else []
    local_files = [p for p in local_files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    if local_files:
        print(f"Using {len(local_files)} local images from {local_dir}")
        for p in local_files:
            try:
                yield Image.open(p)
            except OSError as e:
                print(f"  skipping unreadable file {p.name}: {e}")
        return

    try:
        from datasets import load_dataset
        ds = load_dataset(config.dataset_name, split="train")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        print(FALLBACK_HELP.format(name=config.dataset_name))
        sys.exit(1)

    print(f"Loaded {config.dataset_name}: {len(ds)} images")
    for row in ds:
        yield row["image"]


def main() -> None:
    config = GANConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    tensors = []
    for i, img in enumerate(load_images(config)):
        tensors.append(preprocess(img, config.image_size))
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1} images...")

    if not tensors:
        print("No images processed — nothing to save.")
        sys.exit(1)

    data = torch.stack(tensors)
    out = config.data_file
    torch.save(data, out)

    size_mb = out.stat().st_size / 1e6
    print(f"Total images processed: {len(tensors):,}")
    print(f"Tensor shape: {tuple(data.shape)}, dtype {data.dtype}")
    print(f"Saved to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
