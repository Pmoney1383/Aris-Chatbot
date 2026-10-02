"""
Captions, aspect buckets, the on-disk shard format, and the training dataset.

On-disk layout (under dataset/prepared/):
    text/shard_00000.emb.npy    float16 [total_tokens, text_dim]  (captions concatenated, real tokens only)
    text/shard_00000.off.npy    int64   [n + 1]                   (token offsets per caption)
    text/shard_00000.ids.npy    int64   [n]
    text/null.npy               float16 [tokens, text_dim]        (the empty caption, for CFG dropout)
    latents_256/288x224_00000.lat.npy  float16 [n, 4, h/8, w/8]   (VAE mean * scaling factor)
    latents_256/288x224_00000.ids.npy  int64   [n]
A shard's .ids.npy is written last and acts as its commit marker, so a crash
mid-write leaves an ignorable orphan instead of a corrupt shard.

A sample's id is the first 15 hex digits of its file name (the image's
sha256), which is stable across re-exports of captions.csv.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import config


def _hashed_sample_id(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") >> 4


def sample_id(file_path: str) -> int:
    """Stable 60-bit id, preserving existing SHA-prefixed gathered ids."""
    normalized = file_path.replace("\\", "/")
    prefix = Path(normalized).stem[:15]
    if len(prefix) == 15 and all(c in "0123456789abcdefABCDEF" for c in prefix):
        return int(prefix, 16)
    return _hashed_sample_id(normalized)


_captions_cache: list[dict] | None = None

_TRUE_VALUES = {"1", "true", "yes"}
_ADULT_REQUIRED_COLUMNS = {
    "file_path", "caption", "width", "height",
    "age_verified_adult", "consent_verified", "rights_verified",
}


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _adult_path(raw_path: str) -> str:
    """Return a ROOT-relative adult image path without opening the image."""
    supplied = Path(raw_path.replace("\\", "/"))
    if supplied.is_absolute():
        candidate = supplied
    elif supplied.parts[:2] == ("dataset", "nsfw-images"):
        candidate = config.ROOT / supplied
    else:
        candidate = config.ADULT_IMAGE_DIR / supplied
    candidate = candidate.resolve()
    try:
        candidate.relative_to(config.ADULT_IMAGE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("adult manifest file_path must stay inside dataset/nsfw-images") from exc
    return candidate.relative_to(config.ROOT.resolve()).as_posix()


def _load_adult_captions() -> list[dict]:
    """Load only explicitly attested adult records from the private manifest.

    Files are not discovered and labels are not inferred from names. Every
    admitted record must attest adult age, consent, and data rights.
    """
    if not config.ADULT_IMAGE_DIR.exists():
        return []
    if not config.ADULT_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Adult dataset directory exists but {config.ADULT_MANIFEST_CSV.name} is missing. "
            "Add the verified-adult manifest described in training/ADULT_DATASET.md."
        )
    rows = []
    with open(config.ADULT_MANIFEST_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = _ADULT_REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Adult manifest is missing required columns: {', '.join(sorted(missing))}")
        rejected = 0
        for source in reader:
            if not all(_is_true(source[name]) for name in
                       ("age_verified_adult", "consent_verified", "rights_verified")):
                rejected += 1
                continue
            file_path = _adult_path(source["file_path"])
            caption = source["caption"].strip()
            if not caption:
                rejected += 1
                continue
            try:
                width, height = int(source["width"]), int(source["height"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Adult manifest width and height must be integers") from exc
            if width <= 0 or height <= 0:
                raise ValueError("Adult manifest width and height must be positive")
            rows.append({
                **source,
                "file_path": file_path,
                "caption": config.ADULT_CAPTION_PREFIX + caption,
                "width": width,
                "height": height,
                "id": _hashed_sample_id("adult:" + file_path),
                "top_category": config.ADULT_CATEGORY,
                "content_rating": "adult",
            })
    if rejected:
        print(f"adult captions: excluded {rejected:,} records without complete attestations.", flush=True)
    return rows


def load_captions() -> list[dict]:
    """Load provenance-tracked general images plus attested adult images."""
    global _captions_cache
    if _captions_cache is not None:
        return _captions_cache
    con = sqlite3.connect(f"file:{config.MANIFEST_DB.as_posix()}?mode=ro", uri=True)
    known = {fp for (fp,) in con.execute("SELECT file_path FROM images")}
    con.close()
    rows, skipped = [], 0
    with open(config.CAPTIONS_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["file_path"] not in known:
                skipped += 1
                continue
            r["id"] = sample_id(r["file_path"])
            r["width"], r["height"] = int(r["width"]), int(r["height"])
            r["content_rating"] = "general"
            rows.append(r)
    if skipped:
        print(f"captions: skipped {skipped:,} rows whose image is not in the gathered manifest.", flush=True)
    adult_rows = _load_adult_captions()
    rows.extend(adult_rows)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample ids found across general and adult training manifests")
    if adult_rows:
        print(f"adult captions: included {len(adult_rows):,} fully attested records.", flush=True)
    _captions_cache = rows
    return rows


def assign_bucket(width: int, height: int, bucket_list: list[tuple[int, int]]) -> tuple[int, int]:
    ar = math.log(width / height)
    return min(bucket_list, key=lambda b: abs(ar - math.log(b[0] / b[1])))


def is_val(sid: int, per_mille: int) -> bool:
    return sid % 1000 < per_mille


# -- shard writing ----------------------------------------------------------

def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def write_shard(prefix: Path, arrays: dict[str, np.ndarray], ids: np.ndarray) -> None:
    for suffix, arr in arrays.items():
        _save_npy_atomic(prefix.with_name(f"{prefix.name}.{suffix}.npy"), arr)
    _save_npy_atomic(prefix.with_name(f"{prefix.name}.ids.npy"), ids.astype(np.int64))


def committed_shards(directory: Path, pattern: str) -> list[Path]:
    """Shard prefixes (path without '.ids.npy') whose ids file exists."""
    if not directory.exists():
        return []
    return sorted(p.with_name(p.name[: -len(".ids.npy")]) for p in directory.glob(pattern + ".ids.npy"))


def done_ids(directory: Path, pattern: str) -> set[int]:
    out: set[int] = set()
    for prefix in committed_shards(directory, pattern):
        out.update(np.load(prefix.with_name(prefix.name + ".ids.npy")).tolist())
    return out


def next_shard_index(directory: Path, stem: str) -> int:
    existing = [int(p.name[len(stem) + 1:].split(".")[0]) for p in directory.glob(f"{stem}_*.ids.npy")]
    return max(existing, default=-1) + 1


# -- training dataset -------------------------------------------------------

class TextStore:
    """Random access to caption embeddings by sample id, via memmapped shards."""

    def __init__(self, directory: Path = config.TEXT_DIR):
        prefixes = committed_shards(directory, "shard_*")
        if not prefixes:
            raise FileNotFoundError(f"No text shards in {directory}; run preprocess.py text first.")
        ids, shard, row = [], [], []
        for si, prefix in enumerate(prefixes):
            sids = np.load(prefix.with_name(prefix.name + ".ids.npy"))
            ids.append(sids)
            shard.append(np.full(len(sids), si, dtype=np.int32))
            row.append(np.arange(len(sids), dtype=np.int32))
        ids = np.concatenate(ids)
        order = np.argsort(ids)
        self.ids, self.shard, self.row = ids[order], np.concatenate(shard)[order], np.concatenate(row)[order]
        self.prefixes = prefixes
        self.null = np.load(directory / "null.npy")
        self._maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __contains__(self, sid: int) -> bool:
        i = np.searchsorted(self.ids, sid)
        return i < len(self.ids) and self.ids[i] == sid

    def get(self, sid: int) -> np.ndarray:
        i = np.searchsorted(self.ids, sid)
        si, r = int(self.shard[i]), int(self.row[i])
        if si not in self._maps:
            p = self.prefixes[si]
            self._maps[si] = (np.load(p.with_name(p.name + ".emb.npy"), mmap_mode="r"),
                              np.load(p.with_name(p.name + ".off.npy")))
        emb, off = self._maps[si]
        return np.array(emb[off[r]:off[r + 1]])

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_maps"] = {}  # memmaps are reopened in each DataLoader worker
        return s


SIDE_WORDS = re.compile(r"\b(left|right|leftmost|rightmost)\b", re.I)


class LatentTextDataset(torch.utils.data.Dataset):
    """(latent, caption embedding) pairs for one resolution and split.
    Items in the same bucket share a latent shape; batch them with BucketBatchSampler."""

    def __init__(self, resolution: int, split: str, val_per_mille: int, max_samples: int | None = None,
                 single_bucket: bool = False, hflip: bool = False,
                 caption_variants: tuple[tuple[str, float], ...] = ()):
        # Multi-length captions: "detailed" is the primary store (every sample has one);
        # the others are drawn by weight when the sample has a caption of that length.
        if caption_variants:
            names = [v for v, _ in caption_variants]
            if "detailed" not in names:
                names, caption_variants = ["detailed"] + names, (("detailed", 0.0),) + tuple(caption_variants)
            empty = [v for v in names if v != "detailed" and not committed_shards(config.TEXT_VARIANT_DIRS[v], "shard_*")]
            if empty:  # e.g. no short captions encoded yet: train on the lengths that exist
                print(f"caption variants without encoded captions (skipped): {', '.join(empty)}", flush=True)
                caption_variants = tuple((v, w) for v, w in caption_variants if v not in empty)
                names = [v for v, _ in caption_variants]
            self.variant_stores = [TextStore(config.TEXT_VARIANT_DIRS[v]) for v in names]
            w = np.array([wt for _, wt in caption_variants], dtype=np.float64)
            self.variant_weights = (w / w.sum()).tolist()
            self.text = self.variant_stores[names.index("detailed")]
        else:
            self.variant_stores, self.variant_weights = [], []
            self.text = TextStore()
        prefixes = committed_shards(config.latent_dir(resolution), "*")
        if not prefixes:
            raise FileNotFoundError(f"No latent shards for {resolution}px; run preprocess.py latents --res {resolution}.")
        self.prefixes = prefixes
        self.items: list[tuple[int, int, int]] = []  # (shard index, row, sample id)
        self.bucket_of_shard: list[str] = []
        for si, prefix in enumerate(prefixes):
            self.bucket_of_shard.append(prefix.name.rsplit("_", 1)[0])
            for r, sid in enumerate(np.load(prefix.with_name(prefix.name + ".ids.npy")).tolist()):
                if (split == "val") == is_val(sid, val_per_mille) and sid in self.text:
                    self.items.append((si, r, sid))
        if single_bucket:  # keep only the most common bucket (so a tiny subset still forms full batches)
            top = Counter(self.bucket_of_shard[si] for si, _, _ in self.items).most_common(1)[0][0]
            self.items = [it for it in self.items if self.bucket_of_shard[it[0]] == top]
        if max_samples is not None and len(self.items) > max_samples:
            self.items = random.Random(0).sample(self.items, max_samples)
        self._maps: dict[tuple[int, str], np.ndarray] = {}
        # Flip augmentation: shards that have a mirrored .flip.npy, and ids whose caption
        # names a side (flipping would make it wrong).
        self.hflip = hflip
        self.has_flip = [hflip and p.with_name(p.name + ".flip.npy").exists() for p in prefixes]
        self.no_flip = np.array(sorted(r["id"] for r in load_captions()
                                       if any(SIDE_WORDS.search(r.get(c) or "") for c in config.CAPTION_COLUMNS.values()))
                                if hflip else [], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def bucket(self, idx: int) -> str:
        return self.bucket_of_shard[self.items[idx][0]]

    def _flippable(self, si: int, sid: int) -> bool:
        if not self.has_flip[si]:
            return False
        i = np.searchsorted(self.no_flip, sid)
        return not (i < len(self.no_flip) and self.no_flip[i] == sid)

    def __getitem__(self, idx: int):
        si, r, sid = self.items[idx]
        kind = "flip" if self.hflip and random.random() < 0.5 and self._flippable(si, sid) else "lat"
        if (si, kind) not in self._maps:
            p = self.prefixes[si]
            self._maps[(si, kind)] = np.load(p.with_name(f"{p.name}.{kind}.npy"), mmap_mode="r")
        latent = torch.from_numpy(np.array(self._maps[(si, kind)][r]))
        store = self.text
        if self.variant_stores:
            pick = random.choices(self.variant_stores, weights=self.variant_weights)[0]
            if sid in pick:
                store = pick
        return latent, torch.from_numpy(store.get(sid)), sid

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_maps"] = {}
        return s


def category_of_ids() -> dict[int, str]:
    """CLIP's category for each sample (a better read than the collection folder),
    falling back to the collected category when an image wasn't scored."""
    cats = {r["id"]: r["top_category"] for r in load_captions()}
    if config.CLIP_LABELS_DB.exists():
        con = sqlite3.connect(f"file:{config.CLIP_LABELS_DB.as_posix()}?mode=ro", uri=True)
        for fp, cat in con.execute("SELECT file_path, clip_top_category FROM clip_labels"):
            sid = sample_id(fp)
            if sid in cats:
                cats[sid] = cat
        con.close()
    return cats


def adult_ids() -> set[int]:
    return {r["id"] for r in load_captions() if r.get("content_rating") == "adult"}


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yields index lists that share one aspect bucket, in globally shuffled order.

    Each epoch draws len(dataset) samples. When adult_fraction is set, that
    share is drawn explicitly from verified-adult ids and the remainder from
    general ids; category_count ** -alpha weights each pool. Otherwise normal
    category-balanced sampling is used. Epoch e's order depends only on
    (seed, e); start_batch skips ahead when resuming mid-epoch."""

    def __init__(self, dataset: LatentTextDataset, batch_size: int, alpha: float, seed: int,
                 epoch: int = 0, start_batch: int = 0, adult_fraction: float | None = None):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch
        self.buckets = np.array([dataset.bucket(i) for i in range(len(dataset))])
        if adult_fraction is not None and not 0 <= adult_fraction <= 1:
            raise ValueError("adult_fraction must be between 0 and 1, or None")
        self.adult_fraction = adult_fraction
        known_adult = adult_ids() if adult_fraction else set()
        self.adult_mask = np.array([sid in known_adult for _, _, sid in dataset.items], dtype=bool)
        if adult_fraction and not self.adult_mask.any():
            raise RuntimeError(
                "adult_sample_fraction is enabled, but no prepared adult samples are available; "
                "run preprocess.py text and preprocess.py latents after adding the adult manifest"
            )
        if adult_fraction and self.adult_mask.all() and adult_fraction < 1:
            raise RuntimeError("adult_sample_fraction requires at least one general-audience sample")
        if alpha > 0:
            cats = category_of_ids()
            labels = [cats.get(sid, "?") for _, _, sid in dataset.items]
            counts = Counter(labels)
            w = np.array([counts[c] ** -alpha for c in labels], dtype=np.float64)
            self.weights = w / w.sum()
        else:
            self.weights = None

    def batches(self, epoch: int) -> list[list[int]]:
        if getattr(self, "_cached", (None,))[0] == epoch:
            return self._cached[1]
        rng = np.random.default_rng(self.seed + epoch)
        n = len(self.dataset)
        if self.adult_fraction:
            n_adult = min(n, max(1, round(n * self.adult_fraction)))
            adult_pool = np.flatnonzero(self.adult_mask)
            general_pool = np.flatnonzero(~self.adult_mask)
            if self.weights is None:
                adult_p = general_p = None
            else:
                adult_p = self.weights[adult_pool] / self.weights[adult_pool].sum()
                general_p = self.weights[general_pool] / self.weights[general_pool].sum()
            draw = np.concatenate([
                rng.choice(adult_pool, size=n_adult, replace=n_adult > len(adult_pool), p=adult_p),
                rng.choice(general_pool, size=n - n_adult, replace=n - n_adult > len(general_pool), p=general_p),
            ])
            rng.shuffle(draw)
        else:
            draw = rng.choice(n, size=n, replace=True, p=self.weights) if self.weights is not None else rng.permutation(n)
        out = []
        for b in np.unique(self.buckets):
            members = draw[self.buckets[draw] == b]
            out += [members[i:i + self.batch_size].tolist()
                    for i in range(0, len(members) - self.batch_size + 1, self.batch_size)]
        rng.shuffle(out)
        self._cached = (epoch, out)
        return out

    def __len__(self):
        return len(self.batches(self.epoch)) - self.start_batch

    def __iter__(self):
        yield from self.batches(self.epoch)[self.start_batch:]


class Collate:
    """Pads caption embeddings to the longest in the batch and applies
    caption dropout (the caption becomes the empty one) for CFG training."""

    def __init__(self, null_embedding: np.ndarray, dropout: float):
        self.null = torch.from_numpy(null_embedding)
        self.dropout = dropout

    def __call__(self, batch):
        latents = torch.stack([b[0] for b in batch])
        texts = [self.null if self.dropout and random.random() < self.dropout else b[1] for b in batch]
        length = max(t.shape[0] for t in texts)
        emb = torch.zeros(len(texts), length, texts[0].shape[1], dtype=texts[0].dtype)
        mask = torch.zeros(len(texts), length, dtype=torch.bool)
        for i, t in enumerate(texts):
            emb[i, :t.shape[0]] = t
            mask[i, :t.shape[0]] = True
        return latents, emb, mask, torch.tensor([b[2] for b in batch])
