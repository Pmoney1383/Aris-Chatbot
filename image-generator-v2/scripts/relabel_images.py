"""
Relabels collected images with CLIP: for every image in the manifest, records
which catalogue subcategory it actually looks like, and how well it matches
the subcategory it was collected under.

Search sources match queries loosely (Pixabay especially), so the collected
label is often wrong even when the image is fine. Use these labels, not the
manifest's, when balancing or filtering the training set.

Writes to dataset/clip_labels.sqlite3 (a separate file, so this can run while
gather_images.py is still collecting). Incremental: already-scored images are
skipped, so run it again any time to label new arrivals. Ctrl+C is safe.

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe relabel_images.py              # score new images
    ../.venv/Scripts/python.exe relabel_images.py --report     # summary only
    ../.venv/Scripts/python.exe relabel_images.py --workers 4  # quieter (less CPU)

Columns in clip_labels:
    clip_top_category / clip_subcategory  best-matching catalogue label
    clip_score        similarity to that best label
    margin            best minus second-best similarity (low = ambiguous)
    label_rank        where the collected label ranks among all labels
                      (1 = CLIP agrees; NULL for PD12M's "unmatched")
    label_score       similarity to the collected label
The adult category is never used as a label.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import open_clip
import torch
from PIL import Image

import config
import console

TEMPLATES = ["a photo of {}.", "a picture of {}.", "{}"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS clip_labels (
    sha256 TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    source_name TEXT NOT NULL,
    top_category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    clip_top_category TEXT NOT NULL,
    clip_subcategory TEXT NOT NULL,
    clip_score REAL NOT NULL,
    margin REAL NOT NULL,
    label_rank INTEGER,
    label_score REAL,
    model TEXT NOT NULL,
    scored_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_sub ON clip_labels(clip_subcategory);
CREATE INDEX IF NOT EXISTS idx_clip_source ON clip_labels(source_name);
"""


def catalogue_labels():
    """(subcategories, subcategory -> top_category), adult rows excluded."""
    top_of = {}
    with open(config.CATALOGUE_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["top_category"] not in config.ADULT_CATEGORY_NAMES:
                top_of.setdefault(r["subcategory"], r["top_category"])
    subs = sorted(top_of)
    return subs, top_of


def text_embeddings(model, tokenizer, subs, device):
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        per_template = []
        for t in TEMPLATES:
            e = model.encode_text(tokenizer([t.format(s) for s in subs]).to(device)).float()
            per_template.append(e / e.norm(dim=-1, keepdim=True))
        e = torch.stack(per_template).mean(0)
    return e / e.norm(dim=-1, keepdim=True)


def make_loader(preprocess):
    def load(path):
        try:
            img = Image.open(config.ROOT / path)
            if img.format == "JPEG":
                img.draft("RGB", (448, 448))  # decode at reduced scale; CLIP only needs 224px
            return preprocess(img.convert("RGB"))
        except Exception:
            return None  # missing/unreadable file: skipped now, retried next run
    return load


def report(out: sqlite3.Connection):
    n = out.execute("SELECT COUNT(*) FROM clip_labels").fetchone()[0]
    print("=" * 72)
    print(f"CLIP LABELS: {n:,} images scored")
    print("=" * 72)
    print(f"\n{'source':<14}{'n':>9}{'agrees (top1)':>15}{'top5':>8}{'top20':>8}")
    for src, cnt, t1, t5, t20 in out.execute(
        "SELECT source_name, COUNT(label_rank), "
        "AVG(label_rank<=1), AVG(label_rank<=5), AVG(label_rank<=20) "
        "FROM clip_labels WHERE label_rank IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"{src:<14}{cnt:>9,}{t1:>15.0%}{t5:>8.0%}{t20:>8.0%}")
    print(f"\n{'top category':<32}{'collected as':>14}{'CLIP says':>12}")
    collected = dict(out.execute("SELECT top_category, COUNT(*) FROM clip_labels GROUP BY 1"))
    clip = dict(out.execute("SELECT clip_top_category, COUNT(*) FROM clip_labels GROUP BY 1"))
    for cat in sorted(set(collected) | set(clip), key=lambda c: -clip.get(c, 0)):
        print(f"{cat:<32}{collected.get(cat, 0):>14,}{clip.get(cat, 0):>12,}")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="print the summary and exit")
    # OpenAI's CLIP weights were trained with QuickGELU; plain "ViT-L-14" loads
    # them with standard GELU and degrades the embeddings.
    p.add_argument("--model", default="ViT-L-14-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=6, help="image-decoding threads (more = faster, louder)")
    args = p.parse_args()

    out = sqlite3.connect(config.CLIP_LABELS_DB)
    out.executescript(SCHEMA)
    if args.report:
        report(out)
        return

    manifest = sqlite3.connect(f"file:{config.MANIFEST_DB.as_posix()}?mode=ro", uri=True)
    done = {r[0] for r in out.execute("SELECT sha256 FROM clip_labels")}
    todo = [r for r in manifest.execute(
        "SELECT sha256, file_path, source_name, top_category, subcategory FROM images ORDER BY id")
        if r[0] not in done]
    manifest.close()
    print(f"{len(todo):,} images to score ({len(done):,} already scored).", flush=True)
    if not todo:
        report(out)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    subs, top_of = catalogue_labels()
    sub_index = {s: i for i, s in enumerate(subs)}
    text = text_embeddings(model, tokenizer, subs, device)
    model_name = f"{args.model}/{args.pretrained}"
    print(f"Scoring against {len(subs)} subcategories with {model_name} on {device}.", flush=True)

    load = make_loader(preprocess)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    start, scored, skipped = time.time(), 0, 0
    pool = ThreadPoolExecutor(args.workers)
    try:
        nxt = pool.map(load, [r[1] for r in batches[0]])
        for bi, batch in enumerate(batches):
            tensors = list(nxt)
            if bi + 1 < len(batches):  # decode the next batch while the GPU works on this one
                nxt = pool.map(load, [r[1] for r in batches[bi + 1]])
            keep = [(r, t) for r, t in zip(batch, tensors) if t is not None]
            skipped += len(batch) - len(keep)
            if not keep:
                continue
            ims = torch.stack([t for _, t in keep]).to(device)
            with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
                f = model.encode_image(ims).float()
            f = f / f.norm(dim=-1, keepdim=True)
            sims = f @ text.T
            top2 = sims.topk(2, dim=1)
            now = time.time()
            rows = []
            for (sha, fp, src, top, sub, ), s, vals, idxs in zip(
                    [r for r, _ in keep], sims, top2.values.tolist(), top2.indices.tolist()):
                best = subs[idxs[0]]
                li = sub_index.get(sub)
                rank = int((s > s[li]).sum()) + 1 if li is not None else None
                label_score = float(s[li]) if li is not None else None
                rows.append((sha, fp, src, top, sub, top_of[best], best, vals[0], vals[0] - vals[1],
                             rank, label_score, model_name, now))
            out.executemany("INSERT OR REPLACE INTO clip_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            out.commit()
            scored += len(rows)
            rate = scored / max(time.time() - start, 1e-6)
            console.live(f"scored {scored:,}/{len(todo):,} | {rate:,.0f} img/s | "
                         f"ETA {(len(todo) - scored) / max(rate, 1e-6) / 60:,.1f} min | skipped {skipped}")
    except KeyboardInterrupt:
        console.log("Ctrl+C: stopping; everything scored so far is saved.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        console.end_live()
    print(f"Scored {scored:,} images this run ({skipped} unreadable/missing, will retry next run).")
    report(out)


if __name__ == "__main__":
    main()
