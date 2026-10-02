"""
Captioning for Phase 1.5 with Florence-2-large (local, free).

Writes to the manifest (dataset/manifest.sqlite3):
    caption         detailed caption  (Florence <DETAILED_CAPTION>, label-prefixed for labeled images)
    caption_medium  short sentence    (Florence <CAPTION>, label-prefixed)
    caption_short   the label itself  ("Ferrari 488") for labeled Phase 1.5 images
    grayscale       1 if the image is black-and-white (pixel saturation), else 0
    caption_source  "florence" | "batch_api"

Florence can't be told what an image is, and it misnames fine-grained dishes
(lahmacun -> "pizza with pepperoni"), so Phase 1.5 food and drink images are
NOT captioned here: --export-food-batch writes them as API requests with the
label as a hint, and --import-food-batch reads the results back.

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe caption_images.py uncaptioned         # images with no caption yet
    ../.venv/Scripts/python.exe caption_images.py medium              # optional: Florence medium captions for already-captioned images
                                                                       # (export-csv otherwise uses each caption's first sentence)
    ../.venv/Scripts/python.exe caption_images.py viewpoints          # CLIP view tags for Transportation & Vehicles
    ../.venv/Scripts/python.exe caption_images.py export-food-batch   # -> dataset/food_caption_requests.jsonl
    ../.venv/Scripts/python.exe caption_images.py import-food-batch results.jsonl
    ../.venv/Scripts/python.exe caption_images.py export-csv          # -> dataset/captions.csv for preprocessing
    add --limit N to try a small batch first

Resumable: only rows still missing the field being written are processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import config
import manifest_db

FLORENCE_ID = "florence-community/Florence-2-large"
BATCH = 16
NUM_BEAMS = 1   # greedy: beam 3 ran at 1.8 img/s next to the collector + flips (~32 h for 205k images)
FOOD_DRINK = ("Food & Cooking", "Drinks & Beverages")
GRAYSCALE_SATURATION = 6.0   # mean (max-min channel) spread on a 64x64 thumbnail below this = B&W
BW_WORDS = re.compile(r"black[- ]and[- ]white|black & white|monochrom|grayscale|greyscale|sepia|b&w", re.I)
P15 = config.PHASE15_ID_PREFIX


def is_labeled(catalogue_id: str) -> bool:
    return catalogue_id.startswith(P15)


def with_label(label: str | None, text: str) -> str:
    text = text.strip()
    return f"{label}. {text}" if label else text


def first_sentence(text: str) -> str:
    """The opening sentence of a multi-sentence caption (its summary), or "" if it has only one."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if len(parts) > 1 and len(parts[0]) >= 20 else ""


def load(path: str):
    try:
        img = Image.open(config.ROOT / path)
        img.draft("RGB", (768, 768))  # JPEG: decode at reduced scale; Florence resizes to 768 anyway
        img = img.convert("RGB")
    except Exception:
        return None, None
    thumb = np.asarray(img.resize((64, 64)), dtype=np.float32)
    return img, int((thumb.max(2) - thumb.min(2)).mean() < GRAYSCALE_SATURATION)


class Florence:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(FLORENCE_ID)
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_ID, torch_dtype=torch.float16).to("cuda").eval()

    def run(self, images, task: str) -> list[str]:
        inputs = self.proc(text=[task] * len(images), images=images, return_tensors="pt").to("cuda", self.torch.float16)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=96, num_beams=NUM_BEAMS, do_sample=False)
        texts = self.proc.batch_decode(ids, skip_special_tokens=True)
        return [self.proc.post_process_generation(t, task=task, image_size=im.size)[task].strip()
                for t, im in zip(texts, images)]


def caption_pass(db: manifest_db.ManifestDB, mode: str, limit: int | None, until: str | None = None):
    """uncaptioned: one Florence <DETAILED_CAPTION> pass; the medium caption is its first
    sentence (a second <CAPTION> pass doubled the cost: the vision encoder runs per task).
    Labeled Phase 1.5 images go first. `until` ("HH:MM", local) stops the pass cleanly at
    that time; the rest is picked up by the next run."""
    if mode == "uncaptioned":
        where = "caption IS NULL AND NOT (catalogue_id LIKE ? AND top_category IN (?, ?))"
        params = [P15 + "%", *FOOD_DRINK]
    else:  # medium: already captioned (PD12M, Megalith, API) but no medium caption yet
        where, params = "caption IS NOT NULL AND caption_medium IS NULL", []
    with db.lock:
        rows = db.conn.execute(f"SELECT id, file_path, catalogue_id, subcategory, caption FROM images WHERE {where} "
                               f"ORDER BY (catalogue_id LIKE ?) DESC, id" + (f" LIMIT {int(limit)}" if limit else ""),
                               params + [P15 + "%"]).fetchall()
    stop_at = None
    if until:
        h, m = (int(x) for x in until.split(":"))
        stop_at = time.mktime(time.localtime()[:3] + (h, m, 0, 0, 0, -1))
    print(f"{mode}: {len(rows):,} images to caption" + (f" (stopping at {until})" if until else ""), flush=True)
    if not rows:
        return
    florence = Florence()
    pool = ThreadPoolExecutor(8)
    start, done = time.time(), 0
    for i in range(0, len(rows), BATCH):
        if stop_at and time.time() >= stop_at:
            print(f"{mode}: reached {until}, stopping after {done:,} images.", flush=True)
            break
        batch = rows[i:i + BATCH]
        loaded = list(pool.map(load, [r["file_path"] for r in batch]))
        ok = [(r, img, gray) for r, (img, gray) in zip(batch, loaded) if img is not None]
        if not ok:
            continue
        imgs = [img for _, img, _ in ok]
        if mode == "uncaptioned":
            detailed = florence.run(imgs, "<DETAILED_CAPTION>")
            medium = [first_sentence(d) for d in detailed]
        else:
            medium, detailed = florence.run(imgs, "<CAPTION>"), [None] * len(ok)
        with db.lock:
            for (r, _, gray), med, det in zip(ok, medium, detailed):
                label = r["subcategory"] if is_labeled(r["catalogue_id"]) else None
                if mode == "uncaptioned":
                    db.conn.execute("UPDATE images SET caption=?, caption_medium=?, caption_short=?, grayscale=?, "
                                    "caption_source='florence' WHERE id=?",
                                    (with_label(label, det), with_label(label, med) if med else None, label, gray,
                                     r["id"]))
                else:
                    db.conn.execute("UPDATE images SET caption_medium=?, grayscale=COALESCE(grayscale, ?) WHERE id=?",
                                    (with_label(label, med), gray, r["id"]))
            db.conn.commit()
        done += len(batch)
        if (i // BATCH) % 20 == 0:
            rate = done / max(time.time() - start, 1e-6)
            print(f"{mode}: {done:,}/{len(rows):,} | {rate:,.1f} img/s | ETA {(len(rows) - done) / rate / 3600:,.1f} h",
                  flush=True)
    pool.shutdown()
    print(f"{mode}: done.", flush=True)


# -- vehicle viewpoints -------------------------------------------------------
# Captions rarely say which side of a vehicle is shown, so the model sees one name
# attached to every angle and blends them (a front from one view, a tail from
# another). CLIP zero-shot tags each Transportation image with its view; the tag
# goes into the captions only when CLIP is confident.
VEHICLE_CATEGORY = "Transportation & Vehicles"
VIEWPOINTS = {  # tag -> CLIP text prompts (averaged)
    "front view": ["a photo of the front of a {}, front view", "a {} seen from straight ahead, head-on"],
    "side view": ["a side view of a {}, profile shot", "a {} seen from the side"],
    "rear view": ["a photo of the back of a {}, rear view", "a {} seen from behind"],
    "front three-quarter view": ["a front three-quarter view of a {}", "a {} seen from the front corner at an angle"],
    "rear three-quarter view": ["a rear three-quarter view of a {}", "a {} seen from the back corner at an angle"],
    "top-down view": ["an aerial top-down view of a {}", "a {} seen from directly above"],
    "interior": ["the interior of a {}, seats and dashboard", "inside a {}, cabin interior"],
    "close-up detail": ["a close-up detail of a {}, such as a wheel, badge or headlight", "a macro shot of part of a {}"],
}
VIEW_NOUNS = ("car", "vehicle", "train", "airplane", "boat", "motorcycle", "bus", "truck")
VIEW_MIN_CONF = 0.45   # below this, the image is left untagged rather than mis-tagged


def viewpoint_pass(db: manifest_db.ManifestDB, limit: int | None):
    import open_clip
    import torch
    with db.lock:
        rows = db.conn.execute("SELECT id, file_path FROM images WHERE top_category=? AND viewpoint IS NULL ORDER BY id"
                               + (f" LIMIT {int(limit)}" if limit else ""), (VEHICLE_CATEGORY,)).fetchall()
    print(f"viewpoints: {len(rows):,} transportation images", flush=True)
    if not rows:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    tags = list(VIEWPOINTS)
    with torch.inference_mode():
        text = []
        for tag in tags:
            prompts = [t.format(n) for t in VIEWPOINTS[tag] for n in VIEW_NOUNS]
            e = model.encode_text(tok(prompts).to(device)).float()
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            text.append(e / e.norm())
        text = torch.stack(text)

    def prep(path):
        try:
            return preprocess(Image.open(config.ROOT / path).convert("RGB"))
        except Exception:
            return None

    pool = ThreadPoolExecutor(8)
    start, done = time.time(), 0
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        ims = list(pool.map(prep, [r["file_path"] for r in batch]))
        keep = [(r, im) for r, im in zip(batch, ims) if im is not None]
        if not keep:
            continue
        with torch.inference_mode():
            f = model.encode_image(torch.stack([im for _, im in keep]).to(device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ text.T).softmax(dim=-1).cpu().numpy()
        with db.lock:
            for (r, _), p in zip(keep, probs):
                j = int(p.argmax())
                db.conn.execute("UPDATE images SET viewpoint=?, viewpoint_conf=? WHERE id=?", (tags[j], float(p[j]), r["id"]))
            db.conn.commit()
        done += len(batch)
        if (i // 64) % 20 == 0:
            print(f"viewpoints: {done:,}/{len(rows):,} | {done / max(time.time() - start, 1e-6):,.0f} img/s", flush=True)
    pool.shutdown()
    print("viewpoints: done.", flush=True)


def apply_viewpoint(view: str | None, conf: float | None, label: str | None, caps: list[str]) -> list[str]:
    """Insert a confident viewpoint into the three caption lengths.
    Labeled:   "Bugatti Chiron. The image…"  -> "Bugatti Chiron, side view. The image…"
               interiors / details get their own name: "Bugatti Chiron interior. …"
    Unlabeled: "Side view. The image shows a red car…"."""
    if not view or conf is None or conf < VIEW_MIN_CONF:
        return caps
    out = []
    for c in caps:
        if not c:
            out.append(c)
        elif label and c.startswith(label):
            named = f"{label} interior" if view == "interior" else f"{label}, {view}"
            out.append(named + c[len(label):])
        else:
            out.append(f"{view[0].upper()}{view[1:]}. {c}")
    return out


def export_food_batch(db: manifest_db.ManifestDB):
    out = config.ROOT / "dataset" / "food_caption_requests.jsonl"
    with db.lock:
        rows = db.conn.execute("SELECT sha256, file_path, subcategory, top_category FROM images WHERE caption IS NULL "
                               "AND catalogue_id LIKE ? AND top_category IN (?, ?)", (P15 + "%", *FOOD_DRINK)).fetchall()
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            kind = "dish" if r["top_category"] == FOOD_DRINK[0] else "drink"
            f.write(json.dumps({
                "sha256": r["sha256"], "file_path": r["file_path"], "label": r["subcategory"],
                "prompt": (f"This photo shows the {kind} \"{r['subcategory']}\". Describe the image in one or two "
                           f"sentences: the {kind}'s appearance, how it is served, and the setting. Only describe what "
                           f"is visible. Then, on a new line starting with 'SHORT:', give a 6-12 word caption."),
            }) + "\n")
    print(f"wrote {len(rows):,} requests to {out}")


NOT_VISIBLE = re.compile(r"^\s*NOT_VISIBLE|\bno [^.]{0,40}\b(is|are) visible|\bnot (be )?visible\b|\bisn't visible\b", re.I)


def import_food_batch(db: manifest_db.ManifestDB, path: str):
    """Reads JSONL lines {"sha256": ..., "caption": "<detailed>\\nSHORT: <short>"}.
    Images whose caption says the dish isn't in the photo are left uncaptioned, which keeps
    them out of captions.csv (they're mislabeled: a motorboat filed under "Mochi")."""
    n = skipped = 0
    with open(path, encoding="utf-8") as f, db.lock:
        for line in f:
            d = json.loads(line)
            text = d["caption"].strip()
            if NOT_VISIBLE.search(text):
                skipped += 1
                continue
            detailed, _, short = text.partition("SHORT:")
            row = db.conn.execute("SELECT subcategory, file_path FROM images WHERE sha256=?", (d["sha256"],)).fetchone()
            if not row:
                continue
            _, gray = load(row["file_path"])
            db.conn.execute("UPDATE images SET caption=?, caption_medium=?, caption_short=?, grayscale=?, "
                            "caption_source='batch_api' WHERE sha256=?",
                            (with_label(row["subcategory"], detailed), with_label(row["subcategory"], short.strip() or detailed),
                             row["subcategory"], gray, d["sha256"]))
            n += 1
        db.conn.commit()
    print(f"imported {n:,} captions; skipped {skipped:,} whose image doesn't show the labeled dish")


def export_csv(db: manifest_db.ManifestDB):
    """captions.csv for training/preprocess.py. A grayscale image whose caption doesn't
    already say so gets "Black-and-white photograph." in front of every caption length."""
    out = config.ROOT / "dataset" / "captions.csv"
    if out.exists():  # keep whatever was there before (it may hold hand-made edits)
        backup = out.with_name(f"captions.backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        out.replace(backup)
        print(f"previous captions.csv saved as {backup.name}")
    with db.lock:
        rows = db.conn.execute("SELECT sha256, catalogue_id, file_path, top_category, subcategory, source_name, width, "
                               "height, caption, caption_medium, caption_short, grayscale, viewpoint, viewpoint_conf "
                               "FROM images "
                               "WHERE caption IS NOT NULL AND NOT (top_category=? AND coalesce(no_vehicle_prob, 0) >= ?) "
                               "ORDER BY id", (VEHICLE_CATEGORY, config.NO_VEHICLE_DROP_PROB)).fetchall()
        dropped = db.conn.execute("SELECT count(*) FROM images WHERE caption IS NOT NULL AND top_category=? "
                                  "AND no_vehicle_prob >= ?", (VEHICLE_CATEGORY, config.NO_VEHICLE_DROP_PROB)).fetchone()[0]
    print(f"left out {dropped:,} Transportation images with no vehicle (vehicle_probe.py)")
    # Short labels for general images: only where CLIP agreed with the collected subcategory.
    agreed = set()
    if config.CLIP_LABELS_DB.exists():
        import sqlite3
        con = sqlite3.connect(f"file:{config.CLIP_LABELS_DB.as_posix()}?mode=ro", uri=True)
        agreed = {s for (s,) in con.execute("SELECT sha256 FROM clip_labels WHERE label_rank = 1")}
        con.close()
    tagged = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "top_category", "subcategory", "source_name", "width", "height",
                    "caption", "caption_medium", "caption_short"])
        for r in rows:
            medium = r["caption_medium"] or first_sentence(r["caption"])
            short = r["caption_short"] or (r["subcategory"] if r["sha256"] in agreed and "unmatched" not in r["subcategory"] else "")
            caps = [r["caption"], medium, short]
            label = r["subcategory"] if is_labeled(r["catalogue_id"]) else None
            caps = apply_viewpoint(r["viewpoint"], r["viewpoint_conf"], label, caps)
            if r["grayscale"] and not BW_WORDS.search(r["caption"]):
                caps = ["Black-and-white photograph. " + c if c else "" for c in caps]
                tagged += 1
            w.writerow([r["file_path"], r["top_category"], r["subcategory"], r["source_name"], r["width"], r["height"], *caps])
    print(f"wrote {len(rows):,} rows to {out} ({tagged:,} tagged black-and-white)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["uncaptioned", "medium", "viewpoints", "export-food-batch", "import-food-batch",
                                      "export-csv"])
    p.add_argument("path", nargs="?", help="results JSONL for import-food-batch")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--until", default=None, help='uncaptioned/medium: stop cleanly at this local time, "HH:MM"')
    args = p.parse_args()
    db = manifest_db.ManifestDB()
    try:
        if args.stage in ("uncaptioned", "medium"):
            caption_pass(db, args.stage, args.limit, args.until)
        elif args.stage == "viewpoints":
            viewpoint_pass(db, args.limit)
        elif args.stage == "export-food-batch":
            export_food_batch(db)
        elif args.stage == "import-food-batch":
            import_food_batch(db, args.path)
        else:
            export_csv(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
