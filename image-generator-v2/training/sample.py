"""
Generate images from a trained DiT checkpoint (EMA weights).

Usage (from image-generator-v2/training):
    ../.venv/Scripts/python.exe sample.py --prompt "a red fox in the snow"
    ../.venv/Scripts/python.exe sample.py --prompt "..." --ckpt ../checkpoints/dit/phase1/best.pt --size 256x256 --n 4 --seed 7

Defaults: the best checkpoint of the highest phase that has one; steps and
CFG scale from config.SampleConfig. Output: samples/generated/<timestamp>.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
import flow
from dit import DiT
from train import encode_prompts, pad_texts


def default_checkpoint() -> Path:
    for phase in sorted(config.TRAIN_PHASES, reverse=True):
        p = config.CHECKPOINT_DIR / f"phase{phase}" / "best.pt"
        if p.exists():
            return p
    raise FileNotFoundError("No best.pt found under checkpoints/dit/; train first or pass --ckpt.")


@torch.no_grad()
def main():
    scfg = config.SampleConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--size", default=None, help="WxH in pixels (multiples of 16); default: the checkpoint's resolution, square")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=scfg.steps)
    p.add_argument("--cfg", type=float, default=scfg.cfg_scale)
    args = p.parse_args()

    ckpt_path = args.ckpt or default_checkpoint()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DiT(config.ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["ema"])
    model = model.cuda().to(torch.bfloat16).eval()
    res = ck.get("resolution") or ck.get("train_config", {}).get("resolution", 256)
    w, h = (int(v) for v in args.size.split("x")) if args.size else (res, res)
    shift = config.TRAIN_PHASES["2"].time_shift if res >= 512 else config.TRAIN_PHASES["1"].time_shift

    text, mask = pad_texts(encode_prompts([args.prompt]) * args.n, "cuda")
    null = np.load(config.TEXT_DIR / "null.npy")
    null_t, null_m = pad_texts([null], "cuda")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = flow.sample(model, (args.n, config.LATENT_CHANNELS, h // config.VAE_DOWNSAMPLE, w // config.VAE_DOWNSAMPLE),
                        text, mask, null_t, null_m, args.steps, args.cfg, shift, gen)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16).cuda().eval()
    imgs = vae.decode(z.half() / vae.config.scaling_factor).sample
    imgs = imgs.float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = config.SAMPLE_DIR / "generated" / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(list(imgs), axis=1)).save(out)
    print(f"{ckpt_path} (step {ck.get('step')}) -> {out}")


if __name__ == "__main__":
    main()
