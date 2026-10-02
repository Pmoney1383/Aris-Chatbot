"""
One-time preprocessing: encode captions with the frozen T5 encoder and images
with the frozen SDXL VAE, and store the results as shards, so training never
loads either model or an original image.

Usage (from image-generator-v2/training):
    ../.venv/Scripts/python.exe preprocess.py text                 # caption embeddings, detailed/medium/short stores
                                                                   # (--variant legacy: Phase 1's single store)
    ../.venv/Scripts/python.exe preprocess.py latents --res 256    # VAE latents in 256-class aspect buckets
    ../.venv/Scripts/python.exe preprocess.py flips --res 256      # mirrored latents for flip augmentation (hflip)
    ../.venv/Scripts/python.exe preprocess.py verify --res 256     # decode a few latents back to images + stats
    add --limit N to process only the first N captions (a quick test subset)

Both stages are incremental and resumable: samples already in a committed
shard are skipped, so re-running after new captions arrive only encodes the
new ones. Ctrl+C loses at most the unflushed partial shard.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import config
import data

PCFG = config.PreprocessConfig()


def caption_rows() -> list[dict]:
    """data.load_captions(), minus the adult-manifest records when PREP_GENERAL_ONLY=1 is set.
    Automated runs set it; running the same stage again without it encodes only what's left."""
    rows = data.load_captions()
    if os.environ.get("PREP_GENERAL_ONLY") == "1":
        adult_cat = getattr(config, "ADULT_CATEGORY", None)
        rows = [r for r in rows if r.get("top_category") != adult_cat
                and "nsfw" not in r["file_path"].replace("\\", "/").lower()]
    return rows


def _log(msg: str) -> None:
    print(msg, flush=True)


# -- text -------------------------------------------------------------------

def encode_text(limit: int | None, variant: str = "legacy") -> None:
    """variant "legacy": the `caption` column into TEXT_DIR (Phase 1). "detailed" / "medium" /
    "short": that caption length into its own store (config.TEXT_VARIANT_DIRS); rows with an
    empty caption of that length are skipped. Skips ids already encoded in the target store,
    so captions edited after encoding are NOT re-encoded -- delete the store to redo it."""
    from transformers import AutoTokenizer, T5EncoderModel

    out_dir = config.TEXT_DIR if variant == "legacy" else config.TEXT_VARIANT_DIRS[variant]
    column = "caption" if variant == "legacy" else config.CAPTION_COLUMNS[variant]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in caption_rows()[:limit] if (r.get(column) or "").strip()]
    done = data.done_ids(out_dir, "shard_*")
    todo = [{"id": r["id"], "caption": r[column].strip()} for r in rows if r["id"] not in done]
    _log(f"text [{variant}]: {len(todo):,} captions to encode ({len(done):,} already done).")

    tok = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_ID)
    enc = T5EncoderModel.from_pretrained(config.TEXT_ENCODER_ID, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.inference_mode()
    def run(captions: list[str]) -> list[np.ndarray]:
        t = tok(captions, max_length=PCFG.max_text_tokens, truncation=True, padding=True, return_tensors="pt").to("cuda")
        h = enc(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
        lengths = t.attention_mask.sum(1).tolist()
        h = h.to(torch.float16).cpu().numpy()
        return [h[i, :n] for i, n in enumerate(lengths)]

    if not (out_dir / "null.npy").exists():
        np.save(out_dir / "null.npy", run([""])[0])

    todo.sort(key=lambda r: len(r["caption"]))  # similar lengths per batch -> little padding
    shard_idx = data.next_shard_index(out_dir, "shard")
    buf_emb, buf_ids, start = [], [], time.time()

    def flush():
        nonlocal shard_idx, buf_emb, buf_ids
        if not buf_ids:
            return
        off = np.zeros(len(buf_emb) + 1, dtype=np.int64)
        off[1:] = np.cumsum([e.shape[0] for e in buf_emb])
        data.write_shard(out_dir / f"shard_{shard_idx:05d}",
                         {"emb": np.concatenate(buf_emb), "off": off}, np.array(buf_ids))
        shard_idx += 1
        buf_emb, buf_ids = [], []

    try:
        for i in range(0, len(todo), PCFG.text_batch):
            batch = todo[i:i + PCFG.text_batch]
            buf_emb += run([r["caption"] for r in batch])
            buf_ids += [r["id"] for r in batch]
            if len(buf_ids) >= PCFG.shard_size:
                flush()
            if (i // PCFG.text_batch) % 50 == 0:
                done_n = i + len(batch)
                rate = done_n / max(time.time() - start, 1e-6)
                _log(f"text [{variant}]: {done_n:,}/{len(todo):,} | {rate:,.0f}/s | "
                     f"ETA {(len(todo) - done_n) / rate / 60:,.1f} min")
    finally:
        flush()
    del enc
    torch.cuda.empty_cache()
    _log(f"text [{variant}]: done.")


# -- latents ----------------------------------------------------------------

def load_image(path: str, bucket: tuple[int, int]) -> np.ndarray | None:
    """Resize to cover the bucket, then center-crop to it. Returns HWC uint8."""
    bw, bh = bucket
    try:
        img = Image.open(config.ROOT / path)
        img.draft("RGB", (bw, bh))  # JPEG: decode at the smallest scale still >= the bucket
        img = img.convert("RGB")
        scale = max(bw / img.width, bh / img.height)
        nw, nh = max(bw, round(img.width * scale)), max(bh, round(img.height * scale))
        img = img.resize((nw, nh), Image.BICUBIC, reducing_gap=3.0)
        left, top = (nw - bw) // 2, (nh - bh) // 2
        return np.asarray(img.crop((left, top, left + bw, top + bh)))
    except Exception:
        return None


def encode_latents(resolution: int, limit: int | None) -> None:
    from diffusers import AutoencoderKL

    out_dir = config.latent_dir(resolution)
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_list = config.buckets(resolution)
    rows = caption_rows()[:limit]
    done = data.done_ids(out_dir, "*")
    todo = [r for r in rows if r["id"] not in done]
    random.Random(0).shuffle(todo)  # spreads buckets evenly through the run
    _log(f"latents {resolution}px: {len(todo):,} images to encode ({len(done):,} already done).")

    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16).cuda().eval()
    sf = vae.config.scaling_factor

    @torch.inference_mode()
    def run(images: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).half().div(127.5).sub(1)
        return (vae.encode(x).latent_dist.mean * sf).cpu().numpy().astype(np.float16)

    pending: dict[tuple[int, int], list] = {b: [] for b in bucket_list}  # waiting for a full VAE batch
    shard: dict[tuple[int, int], list] = {b: [] for b in bucket_list}    # encoded, waiting for a full shard
    failed, encoded, start = 0, 0, time.time()

    def flush(b, force=False):
        items = shard[b]
        while items and (force or len(items) >= PCFG.shard_size):
            chunk, items[:] = items[:PCFG.shard_size], items[PCFG.shard_size:]
            stem = f"{b[0]}x{b[1]}"
            idx = data.next_shard_index(out_dir, stem)
            data.write_shard(out_dir / f"{stem}_{idx:05d}",
                             {"lat": np.stack([c[1] for c in chunk])}, np.array([c[0] for c in chunk]))

    def encode_bucket(b):
        nonlocal encoded
        batch, pending[b] = pending[b], []
        lats = run([img for _, img in batch])
        shard[b] += [(sid, lat) for (sid, _), lat in zip(batch, lats)]
        encoded += len(batch)
        flush(b)

    pool = ThreadPoolExecutor(PCFG.decode_workers)
    chunk_size = PCFG.vae_batch * 16
    try:
        for ci in range(0, len(todo), chunk_size):
            chunk = todo[ci:ci + chunk_size]
            bks = [data.assign_bucket(r["width"], r["height"], bucket_list) for r in chunk]
            for r, b, img in zip(chunk, bks, pool.map(load_image, [r["file_path"] for r in chunk], bks)):
                if img is None:
                    failed += 1
                    continue
                pending[b].append((r["id"], img))
                if len(pending[b]) >= PCFG.vae_batch:
                    encode_bucket(b)
            if (ci // chunk_size) % 20 == 0:
                rate = encoded / max(time.time() - start, 1e-6)
                _log(f"latents {resolution}px: {encoded:,}/{len(todo):,} | {rate:,.0f} img/s | "
                     f"ETA {(len(todo) - encoded) / max(rate, 1e-6) / 60:,.1f} min | failed {failed}")
        for b in bucket_list:
            if pending[b]:
                encode_bucket(b)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for b in bucket_list:
            flush(b, force=True)
    _log(f"latents {resolution}px: done ({encoded:,} encoded, {failed} unreadable).")


# -- flips ------------------------------------------------------------------

def encode_flips(resolution: int) -> None:
    """Horizontal-flip augmentation: for every latent shard without one, encode the
    mirrored images into <shard>.flip.npy (same rows, same order). Flipping a latent
    directly is not equivalent -- the SDXL VAE is not mirror-symmetric -- so the
    mirrored images are encoded for real. Resumable per shard."""
    from diffusers import AutoencoderKL

    out_dir = config.latent_dir(resolution)
    todo = [p for p in data.committed_shards(out_dir, "*") if not p.with_name(p.name + ".flip.npy").exists()]
    rows = {r["id"]: r for r in caption_rows()}
    total = sum(len(np.load(p.with_name(p.name + ".ids.npy"))) for p in todo)
    _log(f"flips {resolution}px: {len(todo):,} shards / {total:,} images to encode.")
    if not todo:
        return

    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16).cuda().eval()
    sf = vae.config.scaling_factor

    @torch.inference_mode()
    def run(images: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).half().div(127.5).sub(1)
        return (vae.encode(x).latent_dist.mean * sf).cpu().numpy().astype(np.float16)

    def load_flipped(args):
        sid, bucket = args
        r = rows.get(sid)
        img = load_image(r["file_path"], bucket) if r else None
        return None if img is None else np.ascontiguousarray(img[:, ::-1])

    pool = ThreadPoolExecutor(PCFG.decode_workers)
    done, fallback, start = 0, 0, time.time()
    try:
        for prefix in todo:
            ids = np.load(prefix.with_name(prefix.name + ".ids.npy")).tolist()
            bucket = tuple(int(v) for v in prefix.name.rsplit("_", 1)[0].split("x"))
            original = np.load(prefix.with_name(prefix.name + ".lat.npy"), mmap_mode="r")
            out = np.empty(original.shape, dtype=np.float16)
            for i in range(0, len(ids), PCFG.vae_batch):
                chunk = ids[i:i + PCFG.vae_batch]
                imgs = list(pool.map(load_flipped, [(sid, bucket) for sid in chunk]))
                ok = [j for j, im in enumerate(imgs) if im is not None]
                if ok:
                    lats = run([imgs[j] for j in ok])
                    for j, lat in zip(ok, lats):
                        out[i + j] = lat
                for j in range(len(chunk)):
                    if imgs[j] is None:  # image unreadable now: keep the original (unflipped) latent
                        out[i + j] = original[i + j]
                        fallback += 1
            data._save_npy_atomic(prefix.with_name(prefix.name + ".flip.npy"), out)
            done += len(ids)
            rate = done / max(time.time() - start, 1e-6)
            _log(f"flips {resolution}px: {done:,}/{total:,} | {rate:,.0f} img/s | "
                 f"ETA {(total - done) / max(rate, 1e-6) / 60:,.1f} min | kept unflipped {fallback}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _log(f"flips {resolution}px: done ({done:,} images, {fallback} kept unflipped).")


# -- verify -----------------------------------------------------------------

@torch.inference_mode()
def verify(resolution: int, n: int = 8) -> None:
    """Decode a few stored latents next to the matching original crop, and
    print latent statistics (scaled SDXL latents should have std close to 1)."""
    from diffusers import AutoencoderKL

    multi = data.committed_shards(config.TEXT_VARIANT_DIRS["detailed"], "shard_*")
    ds = data.LatentTextDataset(resolution, "train", val_per_mille=0,
                                caption_variants=(("detailed", 1.0),) if multi else ())
    rows = {r["id"]: r for r in caption_rows()}
    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16).cuda().eval()
    picks = random.Random(0).sample(range(len(ds)), min(n, len(ds)))
    lat_stats, psnrs, tiles = [], [], []
    for i in picks:
        lat, emb, sid = ds[i]
        lat_stats.append(lat.float())
        rec = vae.decode(lat[None].cuda().half() / vae.config.scaling_factor).sample[0]
        rec = rec.float().clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).round().byte().cpu().numpy()
        orig = load_image(rows[sid]["file_path"], (rec.shape[1], rec.shape[0]))
        mse = np.mean((orig.astype(np.float32) - rec.astype(np.float32)) ** 2)
        psnrs.append(10 * np.log10(255 ** 2 / max(mse, 1e-8)))
        tiles.append(np.concatenate([orig, rec], axis=1))
        print(f"  {sid:x}: latent {tuple(lat.shape)}, caption tokens {emb.shape[0]}, "
              f"PSNR {psnrs[-1]:.1f} dB | {rows[sid]['caption'][:70]}")
    allv = torch.cat([s.flatten() for s in lat_stats])
    print(f"{len(ds):,} samples at {resolution}px | latent mean {allv.mean():.3f} std {allv.std():.3f} | "
          f"mean PSNR {np.mean(psnrs):.1f} dB")
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.SAMPLE_DIR / f"verify_{resolution}.png"
    width = max(t.shape[1] for t in tiles)
    Image.fromarray(np.concatenate([np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles])).save(out)
    print(f"original | reconstruction pairs saved to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["text", "latents", "flips", "verify"])
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="only the first N captions (test subset)")
    p.add_argument("--variant", default="all", choices=["all", "legacy", *config.CAPTION_COLUMNS],
                   help="text: which caption length(s) to encode; all = detailed, medium and short (Phase 1.5+), "
                        "legacy = Phase 1's single caption store")
    args = p.parse_args()
    if args.stage == "text":
        for v in (list(config.CAPTION_COLUMNS) if args.variant == "all" else [args.variant]):
            encode_text(args.limit, v)
    elif args.stage == "latents":
        encode_latents(args.res, args.limit)
    elif args.stage == "flips":
        encode_flips(args.res)
    else:
        verify(args.res)


if __name__ == "__main__":
    main()
