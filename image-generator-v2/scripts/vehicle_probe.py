"""
Vehicle viewpoint / no-vehicle probe for Transportation & Vehicles images.

A linear (logistic-regression) probe on frozen CLIP ViT-L-14 image features,
trained from hand labels. Zero-shot CLIP prompts were too unreliable for
viewpoints; hand labels let the probe learn what the dataset actually holds.

Label codes: F front, FQ front three-quarter, S side, R rear, RQ rear
three-quarter, T top-down, I interior, D close-up detail, V a vehicle with no
usable viewpoint (balloons, drawings, tiny distant vehicles, vehicles in a busy
scene), N no vehicle in the picture (houses, crowds, empty streets, animals).
Only N is dropped: V images are still good vehicle data.

Today only the N probability is used: export-csv in caption_images.py drops
Transportation images with no_vehicle_prob >= config.NO_VEHICLE_DROP_PROB.
Viewpoint tags stay out of captions until the probe is accurate enough.

Files (dataset/vehicle_probe/):
    feats.npy, feats_sha.json   CLIP features of every Transportation image (cache)
    labels.csv                  sha256,code   the hand labels
    pending.json, sheet_*.png   the batch currently being labeled
    probe.pt                    trained weights + class list

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe vehicle_probe.py features          # resumable; GPU-capped to run beside training
    ../.venv/Scripts/python.exe vehicle_probe.py sheets --n 600    # contact sheets of images to label next
    ../.venv/Scripts/python.exe vehicle_probe.py merge LABELS.txt  # "idx CODE" lines for pending.json -> labels.csv
    ../.venv/Scripts/python.exe vehicle_probe.py train             # cross-validated report, then fit on all labels
    ../.venv/Scripts/python.exe vehicle_probe.py apply             # writes images.no_vehicle_prob
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

import config
import manifest_db

VEHICLE_CATEGORY = "Transportation & Vehicles"
CODES = ("F", "FQ", "S", "R", "RQ", "T", "I", "D", "V", "N")
DIR = config.ROOT / "dataset" / "vehicle_probe"
CLIP_MODEL = "ViT-L-14-quickgelu"
GPU_FRACTION = 0.15   # ~2.4 GB next to DiT training (measured peak 1.6 GB)
LOGIT_SCALE = 20.0    # features are unit-norm; scale so weight decay acts sensibly
SHEET_COLS, SHEET_ROWS, TILE = 6, 5, 216


def vehicle_rows(db):
    with db.lock:
        return db.conn.execute("SELECT sha256, file_path FROM images WHERE top_category=? ORDER BY id",
                               (VEHICLE_CATEGORY,)).fetchall()


def load_features():
    shas = json.loads((DIR / "feats_sha.json").read_text())
    return shas, np.load(DIR / "feats.npy")


def load_labels() -> dict[str, str]:
    path = DIR / "labels.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["sha256"]: r["code"] for r in csv.DictReader(f)}


# -- features ------------------------------------------------------------------

def features(db):
    import open_clip
    import torch
    DIR.mkdir(parents=True, exist_ok=True)
    rows = vehicle_rows(db)
    shas, feats = (load_features() if (DIR / "feats.npy").exists() else ([], np.zeros((0, 768), np.float32)))
    have = set(shas)
    todo = [r for r in rows if r["sha256"] not in have]
    print(f"features: {len(have):,} cached, {len(todo):,} to compute", flush=True)
    if not todo:
        return
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai", device="cuda")
    model.eval().half()

    def prep(path):
        try:
            return preprocess(Image.open(config.ROOT / path).convert("RGB"))
        except Exception:
            return None

    shas, chunks = list(shas), [feats]
    pool = ThreadPoolExecutor(8)
    start = time.time()

    def save():
        np.save(DIR / "feats.npy", np.concatenate(chunks))
        (DIR / "feats_sha.json").write_text(json.dumps(shas))

    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        ims = list(pool.map(prep, [r["file_path"] for r in batch]))
        keep = [(r, im) for r, im in zip(batch, ims) if im is not None]
        if keep:
            with torch.inference_mode():
                f = model.encode_image(torch.stack([im for _, im in keep]).cuda().half()).float()
                chunks.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            shas += [r["sha256"] for r, _ in keep]
        if (i // 64) % 50 == 49:
            save()
            done = i + len(batch)
            print(f"features: {done:,}/{len(todo):,} | {done / (time.time() - start):,.0f} img/s", flush=True)
    pool.shutdown()
    save()
    print(f"features: done, {len(shas):,} cached", flush=True)


# -- probe ---------------------------------------------------------------------

def fit(X, y, k, wd):
    import torch
    W = torch.zeros(X.shape[1], k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    counts = torch.bincount(y, minlength=k).float()
    weight = len(y) / (k * counts.clamp(min=1))   # class-balanced
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W * LOGIT_SCALE + b, y, weight=weight) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def predict(X, W, b):
    return (X @ W * LOGIT_SCALE + b).softmax(-1)


def stratified_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def labeled_matrix():
    import torch
    shas, feats = load_features()
    pos = {s: i for i, s in enumerate(shas)}
    labels = {s: c for s, c in load_labels().items() if s in pos}
    keys = list(labels)
    classes = [c for c in CODES if c in set(labels.values())]
    X = torch.tensor(feats[[pos[s] for s in keys]])
    y = torch.tensor([classes.index(labels[s]) for s in keys])
    return X, y, classes


def train():
    import torch
    X, y, classes = labeled_matrix()
    K, n = len(classes), len(y)
    yn = y.numpy()
    print(f"train: {n} labels | " + ", ".join(f"{c} {(yn == i).sum()}" for i, c in enumerate(classes)))
    best = None
    for wd in (3e-3, 1e-2, 3e-2, 1e-1):
        P, accs = torch.zeros(n, K), []
        for seed in (0, 1, 2):
            fold = stratified_folds(yn, 5, seed)
            for f in range(5):
                tr, te = torch.tensor(fold != f), torch.tensor(fold == f)
                p = predict(X[te], *fit(X[tr], y[tr], K, wd))
                if seed == 0:
                    P[te] = p
                accs.append((p.argmax(1) == y[te]).float().mean().item())
        if best is None or np.mean(accs) > best[0]:
            best = (float(np.mean(accs)), wd, P)
    acc, wd, P = best
    pred = P.argmax(1)
    print(f"\nviewpoint ({K}-way): 5-fold x3 accuracy {acc:.3f} (weight decay {wd})")
    for i, c in enumerate(classes):
        m = y == i
        print(f"  {c:<3} recall {(pred[m] == i).float().mean():.2f}  ({m.sum().item()})")
    conf = P.max(1).values
    for th in (0.6, 0.7, 0.8, 0.9):
        m = conf >= th
        print(f"  conf>={th}: coverage {m.float().mean():.2f}, accuracy {(pred[m] == y[m]).float().mean():.3f}")

    # The no-vehicle filter: precision matters most (a dropped real vehicle is lost data).
    if "N" in classes:
        ni = classes.index("N")
        pn, is_n = P[:, ni], y == ni
        print("\nno-vehicle filter (cross-validated):")
        for th in (0.5, 0.6, 0.7, 0.8, 0.9):
            drop = pn >= th
            prec = (drop & is_n).sum().item() / max(drop.sum().item(), 1)
            rec = (drop & is_n).sum().item() / max(is_n.sum().item(), 1)
            print(f"  drop if p(N)>={th}: precision {prec:.2f}, recall {rec:.2f}, drops {drop.float().mean():.1%} of images")

    W, b = fit(X, y, K, wd)
    torch.save({"W": W, "b": b, "classes": classes, "wd": wd, "cv_accuracy": acc, "n_labels": n,
                "clip_model": CLIP_MODEL}, DIR / "probe.pt")
    print(f"\nsaved {DIR / 'probe.pt'}")


def apply(db):
    import torch
    probe = torch.load(DIR / "probe.pt")
    shas, feats = load_features()
    P = predict(torch.tensor(feats), probe["W"], probe["b"])
    pn = P[:, probe["classes"].index("N")].numpy()
    with db.lock:
        db.conn.executemany("UPDATE images SET no_vehicle_prob=? WHERE sha256=?",
                            [(float(p), s) for s, p in zip(shas, pn)])
        db.conn.commit()
    drop = (pn >= config.NO_VEHICLE_DROP_PROB).sum()
    print(f"apply: scored {len(shas):,} images; {drop:,} ({drop / len(shas):.1%}) at p(N) >= "
          f"{config.NO_VEHICLE_DROP_PROB} will be left out of captions.csv")


# -- labeling sheets -----------------------------------------------------------

def sheets(db, n: int, seed: int):
    """Pick the next images to label. With a probe: half from the classes it is
    weakest on (predicted rare views, balanced), a quarter of the least
    confident, a quarter random. Without one: random."""
    labels = load_labels()
    paths = {r["sha256"]: r["file_path"] for r in vehicle_rows(db)}
    have_feats = (DIR / "feats.npy").exists()
    shas, feats = load_features() if have_feats else (list(paths), None)
    pool = np.array([i for i, s in enumerate(shas) if s not in labels and s in paths])
    rng = np.random.default_rng(seed)
    if have_feats and (DIR / "probe.pt").exists():
        import torch
        probe = torch.load(DIR / "probe.pt")
        P = predict(torch.tensor(feats[pool]), probe["W"], probe["b"])
        pred, conf = P.argmax(1).numpy(), P.max(1).values.numpy()
        classes = probe["classes"]
        common = {classes.index(c) for c in ("FQ", "N", "V") if c in classes}
        rare = [k for k in range(len(classes)) if k not in common]
        chosen: list[int] = []
        per = n // 2 // max(len(rare), 1)
        for k in rare:
            idx = np.where(pred == k)[0]
            chosen += list(rng.permutation(idx)[:per])
        rest = np.setdiff1d(np.arange(len(pool)), chosen)
        chosen += list(rest[np.argsort(conf[rest])][: n // 4])
        rest = np.setdiff1d(np.arange(len(pool)), chosen)
        chosen += list(rng.permutation(rest)[: n - len(chosen)])
        picked = pool[rng.permutation(np.array(chosen))]
    else:
        picked = rng.permutation(pool)[:n]
    items = [[shas[i], paths[shas[i]]] for i in picked]
    for old in DIR.glob("sheet_*.png"):
        old.unlink()
    (DIR / "pending.json").write_text(json.dumps(items))
    per_sheet = SHEET_COLS * SHEET_ROWS
    for s in range(0, len(items), per_sheet):
        sheet = Image.new("RGB", (SHEET_COLS * (TILE + 4), SHEET_ROWS * (TILE + 20)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (_, path) in enumerate(items[s:s + per_sheet]):
            x, y = (j % SHEET_COLS) * (TILE + 4), (j // SHEET_COLS) * (TILE + 20)
            try:
                im = Image.open(config.ROOT / path).convert("RGB")
                im.thumbnail((TILE, TILE))
                sheet.paste(im, (x + (TILE - im.width) // 2, y + 18 + (TILE - im.height) // 2))
            except Exception:
                pass
            draw.rectangle([x, y, x + 34, y + 15], fill="yellow")
            draw.text((x + 3, y + 2), str(s + j), fill="black")
        sheet.save(DIR / f"sheet_{s // per_sheet:02d}.png")
    print(f"sheets: {len(items)} images on {(len(items) + per_sheet - 1) // per_sheet} sheets in {DIR}")


def merge(path: str):
    items = json.loads((DIR / "pending.json").read_text())
    labels = load_labels()
    added = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        i, code = line.split()
        if code not in CODES:
            raise SystemExit(f"bad code {code!r} on line: {line.strip()}")
        labels[items[int(i)][0]] = code
        added += 1
    with open(DIR / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "code"])
        w.writerows(labels.items())
    print(f"merge: {added} labels from {path}; {len(labels)} total")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["features", "sheets", "merge", "train", "apply"])
    p.add_argument("path", nargs="?", help="label file for merge")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.stage == "merge":
        return merge(args.path)
    if args.stage == "train":
        return train()
    db = manifest_db.ManifestDB()
    try:
        if args.stage == "features":
            features(db)
        elif args.stage == "sheets":
            sheets(db, args.n, args.seed)
        else:
            apply(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
