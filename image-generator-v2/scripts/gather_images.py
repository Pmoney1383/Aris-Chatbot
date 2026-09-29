"""
Data-gathering script for the text-to-image dataset.

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe gather_images.py                  # run, resumable
    ../.venv/Scripts/python.exe gather_images.py --status         # progress/ETA report and exit
    ../.venv/Scripts/python.exe gather_images.py --workers 24
    ../.venv/Scripts/python.exe gather_images.py --category "Food & Cooking" --limit 200
    ../.venv/Scripts/python.exe gather_images.py --bing             # also use Bing (off by default)
    ../.venv/Scripts/python.exe gather_images.py --retry-exhausted
    ../.venv/Scripts/python.exe gather_images.py --pd12m-only       # or --no-pd12m
    ../.venv/Scripts/python.exe gather_images.py --megalith-only    # or --no-megalith

Two kinds of work run side by side:
  - search rows: catalogue rows queried against the sources in SOURCE_ORDER
  - PD12M bulk import (pd12m.py): public-domain images streamed from the
    PD12M metadata, filtered by size/caption before download

NUM_WORKERS threads each take one catalogue row at a time (rows are shuffled,
so if you stop early the dataset is still balanced across categories). Per
row, sources are queried in SOURCE_ORDER; a source whose rate-limit slot is
far away is skipped for now rather than waited on. Each candidate goes
through:

  1. skip if its URL was already tried (any row, any run) or its host is a
     stock-photo site (config.BLOCKED_IMAGE_DOMAINS)
  2. download (paced per host, 429 -> host cooldown)
  3. reject: below the source's min short side, exact sha256 duplicate,
     corrupt/oversized, near-duplicate pHash, watermark detected
  4. downscale so the long edge is <= SOURCE_DOWNLOAD_MAX_EDGE (never below
     the min short side), encode JPEG, reject if > MAX_OUTPUT_JPEG_BYTES
  5. record in dataset/manifest.sqlite3 and write to
     dataset/final/<category>/<subcategory>/<sha>.jpg; the first image per
     subcategory is also copied to dataset/test-image/ for spot checks.

The "Adult 18+ Safety Research" category is never collected here: its rows
are marked gated_skip and no source is ever queried for them.

Ctrl+C stops cleanly (finishes in-flight images, commits the manifest).
Re-running continues where it left off.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import random
import threading
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests
from PIL import Image, ImageOps

import config
import console
import manifest_db
import ratelimit
import sources as sources_mod

try:
    import imagehash
except ImportError:
    imagehash = None

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message="Palette images with Transparency")
# Real high-res photography can exceed Pillow's 89-megapixel default; allow
# up to 300MP. Bigger than that is rejected (DecompressionBombError), not a crash.
Image.MAX_IMAGE_PIXELS = 300_000_000


def safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in s).strip().replace(" ", "_")


def load_catalogue():
    with open(config.CATALOGUE_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_adult(top_category: str) -> bool:
    return top_category in config.ADULT_CATEGORY_NAMES and not config.ADULT_COLLECTION_ENABLED


class PHashIndex:
    """Approximate near-duplicate index: bucket 64-bit pHashes by their top 16
    bits and compare Hamming distance only within a bucket."""

    def __init__(self, existing_hex):
        self.lock = threading.Lock()
        self.buckets: dict[int, list[int]] = collections.defaultdict(list)
        for h in existing_hex:
            if h:
                v = int(h, 16)
                self.buckets[v >> 48].append(v)

    def try_add(self, v: int) -> bool:
        """Atomically: False if a near-duplicate is already indexed, else index
        v and return True. Check and insert under one lock, so two workers
        holding near-identical images can't both pass."""
        with self.lock:
            bucket = self.buckets[v >> 48]
            if any(bin(v ^ o).count("1") <= config.PHASH_MAX_DISTANCE for o in bucket):
                return False
            bucket.append(v)
            return True

    def remove(self, v: int):
        """Undo try_add for an image that was rejected by a later check."""
        with self.lock:
            bucket = self.buckets.get(v >> 48)
            if bucket and v in bucket:
                bucket.remove(v)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.accepted = 0
        self.candidates = 0
        self.rejects = collections.Counter()
        self.by_source = collections.Counter()
        self.history = collections.deque()  # (timestamp, accepted) samples for rate

    def accept(self, source_name):
        with self.lock:
            self.accepted += 1
            self.candidates += 1
            self.by_source[source_name] += 1

    def reject(self, reason):
        with self.lock:
            self.candidates += 1
            self.rejects[reason] += 1


class Context:
    def __init__(self, db, srcs, phash_index, seen, detector, limit):
        self.db = db
        self.sources = srcs
        self.phash_index = phash_index
        self.seen = seen
        self.seen_lock = threading.Lock()
        self.detector = detector
        self.limit = limit
        self.stats = Stats()
        self.stop = threading.Event()
        self.local = threading.local()
        self.cursor_lock = threading.Lock()
        self.cursors: dict[tuple[str, str], dict] = {}
        self.bulks = []  # running bulk importers (PD12M, Megalith)
        self.initial_total = 0  # images already in the manifest when this run started

    # Page cursors for by_subcategory sources, shared by all rows of one
    # subcategory so each row fetches a different page.
    MAX_SHARED_PAGES = 50

    def claim_page(self, key):
        with self.cursor_lock:
            c = self.cursors.setdefault(key, {"next": 1, "returned": [], "exhausted": False})
            if c["returned"]:
                return c["returned"].pop()
            if c["exhausted"] or c["next"] > self.MAX_SHARED_PAGES:
                return None
            c["next"] += 1
            return c["next"] - 1

    def return_page(self, key, page):
        with self.cursor_lock:
            self.cursors[key]["returned"].append(page)

    def exhaust(self, key):
        with self.cursor_lock:
            self.cursors[key]["exhausted"] = True

    def session(self) -> requests.Session:
        s = getattr(self.local, "session", None)
        if s is None:
            s = requests.Session()
            self.local.session = s
        return s

    def claim_url(self, url: str) -> bool:
        """True if this URL hasn't been tried before (and marks it tried)."""
        h = manifest_db.url_hash(url)
        with self.seen_lock:
            if h in self.seen:
                return False
            self.seen.add(h)
        self.db.add_seen_url(h)
        return True


def download_bytes(ctx: Context, url: str):
    domain = ratelimit.domain_of(url)
    if not ratelimit.reserve(domain):
        return None
    ua = config.USER_AGENT if domain.endswith("wikimedia.org") else config.BROWSER_USER_AGENT
    for attempt in range(config.DOWNLOAD_RETRY_COUNT + 1):
        try:
            with ctx.session().get(url, timeout=config.REQUEST_TIMEOUT_SECONDS, stream=True,
                                   headers={"User-Agent": ua}) as r:
                if r.status_code == 429:
                    ratelimit.cool_down(domain, ratelimit.parse_retry_after(r.headers.get("Retry-After")))
                    return None
                if r.status_code != 200:
                    return None
                ctype = r.headers.get("Content-Type", "")
                if ctype and not ctype.startswith("image/") and "octet-stream" not in ctype:
                    return None
                length = r.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > config.MAX_DOWNLOAD_BYTES:
                    return None
                buf = bytearray()
                try:
                    for chunk in r.iter_content(64 * 1024):
                        buf.extend(chunk)
                        if len(buf) > config.MAX_DOWNLOAD_BYTES:
                            return None
                finally:
                    ratelimit.record_bytes(domain, len(buf))
                return bytes(buf)
        except Exception:
            if attempt < config.DOWNLOAD_RETRY_COUNT:
                time.sleep(0.5)
    return None


def fit_size(w: int, h: int, min_short: int) -> tuple[int, int]:
    """Shrink so the long edge is <= SOURCE_DOWNLOAD_MAX_EDGE, but never push
    the short side below min_short. Never upscales."""
    long_edge, short_edge = max(w, h), min(w, h)
    scale = min(1.0, max(config.SOURCE_DOWNLOAD_MAX_EDGE / long_edge, min_short / short_edge))
    return max(1, round(w * scale)), max(1, round(h * scale))


def process_candidate(ctx: Context, cand, row, source_name: str) -> bool:
    db = ctx.db
    cid = row["catalogue_id"]
    min_short = config.SOURCE_MIN_SHORT_SIDE.get(source_name, config.MIN_SHORT_SIDE)

    def reject(reason):
        db.log_reject(cid, reason, cand.url)
        ctx.stats.reject(reason)
        return False

    data = download_bytes(ctx, cand.url)
    if data is None:
        return reject("download_failed")

    sha256 = hashlib.sha256(data).hexdigest()
    if db.sha256_exists(sha256):
        return reject("duplicate_sha256")

    try:
        img = Image.open(io.BytesIO(data))
        orig_w, orig_h = img.size  # header only, no decode yet
        if min(orig_w, orig_h) < min_short:
            return reject("below_min_resolution")
        img.verify()
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            # decode at reduced scale: much faster for big JPEGs
            img.draft("RGB", fit_size(orig_w, orig_h, min_short))
        # Phone photos store rotation as an EXIF flag; our re-encode drops
        # EXIF, so bake the rotation into the pixels first.
        img = ImageOps.exif_transpose(img.convert("RGB"))
        target = fit_size(img.width, img.height, min_short)
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
    except Image.DecompressionBombError:
        return reject("oversized_image")
    except Exception:  # Pillow raises a zoo of types on malformed files
        return reject("corrupt_image")

    phash_int = phash_hex = None
    if imagehash is not None:
        phash_hex = str(imagehash.phash(img))
        phash_int = int(phash_hex, 16)
        if not ctx.phash_index.try_add(phash_int):
            return reject("near_duplicate_phash")

    # The pHash is now reserved in the index; release it if any later step
    # rejects the image (or raises), so it doesn't block a future good copy.
    stored = False
    try:
        stored = _check_and_store(ctx, img, sha256, phash_hex, cand, row, source_name, reject)
    finally:
        if not stored and phash_int is not None:
            ctx.phash_index.remove(phash_int)
    return stored


def _check_and_store(ctx: Context, img, sha256, phash_hex, cand, row, source_name, reject) -> bool:
    db = ctx.db
    cid = row["catalogue_id"]
    wm_score = None
    if ctx.detector is not None:
        wm_score = ctx.detector.score(img)
        if wm_score >= config.WATERMARK_THRESHOLD:
            return reject("watermark")

    encoded = io.BytesIO()
    img.save(encoded, format="JPEG", quality=config.OUTPUT_JPEG_QUALITY, optimize=True)
    if encoded.tell() > config.MAX_OUTPUT_JPEG_BYTES:
        return reject("output_too_large")

    cat_slug, sub_slug = safe_slug(row["top_category"]), safe_slug(row["subcategory"])
    dest_dir = config.FINAL_DIR / cat_slug / sub_slug
    dest_path = dest_dir / f"{sha256[:16]}.jpg"

    inserted = db.try_insert_image(
        row["target"],
        catalogue_id=cid,
        top_category=row["top_category"],
        subcategory=row["subcategory"],
        file_path=str(dest_path.relative_to(config.ROOT)),
        source_name=source_name,
        source_url=cand.source_url,
        source_domain=cand.source_domain,
        license=cand.license,
        sha256=sha256,
        phash=phash_hex,
        width=img.width,
        height=img.height,
        watermark_score=wm_score,
        caption=cand.caption,
        adult_label="sfw",
        safety_status="unreviewed",
        downloaded_at=time.time(),
    )
    if not inserted:
        return reject("duplicate_or_row_full")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(encoded.getvalue())
    if db.claim_sample(row["top_category"], row["subcategory"]):
        config.TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        (config.TEST_IMAGE_DIR / f"{cat_slug}__{sub_slug}.jpg").write_bytes(encoded.getvalue())

    ctx.stats.accept(source_name)
    if ctx.limit and ctx.stats.accepted >= ctx.limit:
        ctx.stop.set()
    return True


def process_row(ctx: Context, row):
    db = ctx.db
    cid = row["catalogue_id"]
    if is_adult(row["top_category"]):
        db.mark_row_status(cid, "gated_skip")
        return
    if ctx.stop.is_set():
        return

    target = row["target"]
    accepted = db.row_accepted(cid)
    if accepted >= target:
        db.mark_row_status(cid, "done")
        return
    db.mark_row_status(cid, "in_progress")

    state = {
        s.name: {"page": 1, "pages": 0, "stagnant": 0,
                 "done": s.categories is not None and row["top_category"] not in s.categories}
        for s in ctx.sources
    }
    busy_passes = 0
    while not ctx.stop.is_set() and accepted < target:
        accepted = db.row_accepted(cid)  # the Megalith filler also credits this row
        if accepted >= target:
            break
        active =[s for s in ctx.sources if not state[s.name]["done"]]
        if not active:
            break
        answered = False
        for source in active:
            if ctx.stop.is_set() or accepted >= target:
                break
            st = state[source.name]
            if source.by_subcategory:
                cursor_key = (source.name, row["subcategory"])
                page = ctx.claim_page(cursor_key)
                if page is None:
                    st["done"] = True
                    continue
                cands = source.search(row["subcategory"], page)
                if cands is sources_mod.BUSY:
                    ctx.return_page(cursor_key, page)
                    continue
                if not cands:
                    ctx.exhaust(cursor_key)
            else:
                cands = source.search(row["query_seed"], st["page"])
                if cands is sources_mod.BUSY:
                    continue
                st["page"] += 1
            answered = True
            st["pages"] += 1
            got = 0
            for cand in cands:
                if ctx.stop.is_set() or accepted >= target:
                    break
                if sources_mod.is_blocked_domain(cand.url):
                    ctx.stats.reject("blocked_stock_domain")
                    continue
                if ratelimit.is_cooling(ratelimit.domain_of(cand.url)):
                    continue  # try this URL again on a later page/run rather than burning it
                if not ctx.claim_url(cand.url):
                    ctx.stats.reject("url_already_tried")
                    continue
                if process_candidate(ctx, cand, row, source.name):
                    accepted += 1
                    got += 1
            st["stagnant"] = 0 if got else st["stagnant"] + 1
            if (not cands or st["stagnant"] >= config.MAX_STAGNANT_PAGES
                    or st["pages"] >= config.MAX_PAGES_PER_SOURCE):
                st["done"] = True
        if answered:
            busy_passes = 0
        else:
            busy_passes += 1
            if busy_passes >= 30:
                break  # remaining sources stayed rate-limited; leave the row for a later run
            time.sleep(1.0)

    if db.row_accepted(cid) >= target:
        db.mark_row_status(cid, "done")
    elif all(st["done"] for st in state.values()):
        db.mark_row_status(cid, "exhausted")
    else:
        db.mark_row_status(cid, "pending")


def fmt_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "?"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def _rates(ctx: Context):
    """Sample the accepted counter and return (per-min over the last minute,
    per-min over the last 10 minutes)."""
    s = ctx.stats
    now = time.time()
    with s.lock:
        s.history.append((now, s.accepted))
        while s.history and now - s.history[0][0] > 600:
            s.history.popleft()
        accepted = s.accepted
        t10, a10 = s.history[0]
        t1, a1 = next(((t, a) for t, a in s.history if now - t <= 60), (now, accepted))
    per_min = lambda t, a: (accepted - a) / ((now - t) / 60) if now - t > 1 else 0.0
    return per_min(t1, a1), per_min(t10, a10)


def live_line(ctx: Context, start: float, rows_total: int, done_futures: list) -> str:
    rate_now, rate_avg = _rates(ctx)
    s = ctx.stats
    with s.lock:
        accepted, rejected = s.accepted, s.candidates - s.accepted
    total = ctx.initial_total + accepted
    eta = max(config.TOTAL_TARGET_IMAGES - total, 0) / rate_avg * 60 if rate_avg > 0 else None
    pd = "".join(f" | {b.name} {b.total:,}" for b in ctx.bulks)
    return (f"[{fmt_duration(time.time() - start)}] {total:,} total (+{accepted:,}) | "
            f"{rate_now:,.0f}/min now, {rate_avg:,.0f} avg | 500k ETA {fmt_duration(eta)}{pd} | "
            f"rows {done_futures[0]:,}/{rows_total:,} | rejected {rejected:,}")


def detail_report(ctx: Context, start: float) -> str:
    s = ctx.stats
    with s.lock:
        top_rejects = ", ".join(f"{k} {v:,}" for k, v in s.rejects.most_common(5))
        by_source = ", ".join(f"{k} {v:,}" for k, v in s.by_source.most_common())
        accepted, candidates = s.accepted, s.candidates
    bulk = "".join(f"\n    {b.status_line()}" for b in ctx.bulks)
    return (f"--- [{fmt_duration(time.time() - start)}] +{accepted:,} accepted of {candidates:,} candidates this run\n"
            f"    sources: {by_source or '-'}\n"
            f"    top rejects: {top_rejects or '-'}{bulk}")


def progress_loop(ctx: Context, start: float, rows_total: int, done_futures: list):
    """Redraw the live line every second; print the full breakdown above it
    every 5 minutes (every 30s, with no live line, when output isn't a terminal)."""
    detail_every = 300 if console.IS_TTY else 30
    last_detail = time.time()
    while not ctx.stop.wait(1.0):
        line = live_line(ctx, start, rows_total, done_futures)
        console.live(line)
        if time.time() - last_detail >= detail_every:
            if not console.IS_TTY:
                console.log(line)
            console.log(detail_report(ctx, start))
            last_detail = time.time()


def print_status(db: manifest_db.ManifestDB):
    total, by_status = db.totals()
    rate = db.rate_per_hour()
    print("=" * 64)
    print("DATA GATHERING STATUS")
    print("=" * 64)
    print(f"Accepted images total : {total:,}")
    print("Rows by status        : " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_status.items())))
    print(f"Recent rate           : " + (f"{rate:,.0f} img/hr ({rate / 60:,.0f} img/min)" if rate else "n/a (not enough recent activity)"))
    for name, target in (("500k target", config.TOTAL_TARGET_IMAGES), ("1M stretch", config.STRETCH_TARGET_IMAGES)):
        remaining = max(target - total, 0)
        eta = f"ETA ~{fmt_duration(remaining / rate * 3600)}" if rate else "ETA unknown"
        print(f"  -> {name}: {remaining:,} remaining, {eta}")
    print("\nBy source:")
    for r in db.source_totals():
        print(f"  {r['source_name']:<32} {r['n']:>9,}")
    print("\nRejects:")
    for r in db.reject_totals():
        print(f"  {r['reason']:<32} {r['n']:>9,}")
    print("\nBy category:")
    for r in db.per_category_totals():
        print(f"  {r['top_category']:<32} {r['n']:>9,}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="print progress/ETA and exit")
    parser.add_argument("--category", type=str, default=None, help="only process this top_category")
    parser.add_argument("--limit", type=int, default=None, help="stop after accepting this many images this run")
    parser.add_argument("--per-row-target", type=int, default=None, help="override config.IMAGES_PER_ROW_TARGET")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--bing", action="store_true",
                        help="also use Bing web image search (off by default: results are often off-topic)")
    parser.add_argument("--no-watermark", action="store_true", help="skip watermark detection")
    parser.add_argument("--retry-exhausted", action="store_true",
                        help="also revisit rows that ran out of candidates in an earlier run")
    parser.add_argument("--no-pd12m", action="store_true", help="don't run the PD12M bulk import")
    parser.add_argument("--pd12m-only", action="store_true", help="run only the PD12M bulk import, no search rows")
    parser.add_argument("--pd12m-workers", type=int, default=config.PD12M_WORKERS)
    parser.add_argument("--no-megalith", action="store_true", help="don't run the Megalith gap filler")
    parser.add_argument("--megalith-only", action="store_true", help="run only the Megalith gap filler, no search rows")
    parser.add_argument("--megalith-workers", type=int, default=config.MEGALITH_WORKERS)
    args = parser.parse_args()

    if args.bing:
        config.ENABLE_BING = True
    if args.per_row_target:
        config.IMAGES_PER_ROW_TARGET = args.per_row_target

    if args.status:
        if not config.MANIFEST_DB.exists():
            print("No manifest yet -- nothing collected.")
            return
        db = manifest_db.ManifestDB(readonly=True)
        print_status(db)
        db.close()
        return

    db = manifest_db.ManifestDB()

    catalogue = load_catalogue()
    db.upsert_rows(
        (r["catalogue_id"], r["top_category"], r["subcategory"], r["query_seed"], r["source_tier"],
         config.IMAGES_PER_ROW_TARGET, is_adult(r["top_category"]))
        for r in catalogue
    )

    srcs = [s for s in sources_mod.build_sources() if s.available()]
    print(f"Active sources: {[s.name for s in srcs]}", flush=True)
    if not srcs:
        print("No sources available (check API keys in .env). Aborting.")
        db.close()
        return

    detector = None
    if config.WATERMARK_ENABLED and not args.no_watermark:
        import watermark
        detector = watermark.WatermarkDetector()
        print(f"Watermark detector loaded on {detector.device} (reject >= {config.WATERMARK_THRESHOLD})", flush=True)

    print("Loading dedup indexes...", flush=True)
    ctx = Context(db, srcs, PHashIndex(db.all_phashes()), db.load_seen_urls(), detector, args.limit)
    ctx.initial_total = db.totals()[0]

    rows = []
    if not (args.pd12m_only or args.megalith_only):
        rows = [r for r in db.pending_rows(args.category, args.retry_exhausted) if not is_adult(r["top_category"])]
        random.Random(0).shuffle(rows)
    print(f"{len(rows):,} rows to work through with {args.workers} workers.", flush=True)

    if config.PD12M_ENABLED and not args.no_pd12m and not args.category and not args.megalith_only:
        import pd12m
        ctx.bulks.append(pd12m.PD12MImporter(ctx, pd12m.CaptionMatcher(catalogue), process_candidate, args.pd12m_workers))
        print(f"PD12M bulk import running with {args.pd12m_workers} workers.", flush=True)
    if config.MEGALITH_ENABLED and not args.no_megalith and not args.category and not args.pd12m_only:
        import megalith
        ctx.bulks.append(megalith.MegalithImporter(ctx, process_candidate, args.megalith_workers))
        print(f"Megalith gap filler running with {args.megalith_workers} workers.", flush=True)
    for b in ctx.bulks:
        b.start()

    start = time.time()
    done_count = [0]
    progress = threading.Thread(target=progress_loop, args=(ctx, start, len(rows), done_count), daemon=True)
    progress.start()

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = {pool.submit(process_row, ctx, row) for row in rows}
    try:
        # Poll with a timeout rather than blocking in as_completed(), so
        # Ctrl+C is delivered promptly on Windows.
        while pending and not ctx.stop.is_set():
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                done_count[0] += 1
                try:
                    fut.result()
                except Exception:
                    # One bad row must never take down the run.
                    console.log("Row failed (continuing):\n" + traceback.format_exc())
        if ctx.bulks and not ctx.stop.is_set():
            console.log("Search rows finished; waiting for the bulk importers...")
            for b in ctx.bulks:
                while not b.done.wait(1.0):
                    pass
    except KeyboardInterrupt:
        console.log("Ctrl+C: finishing in-flight images and saving...")
        ctx.stop.set()
    finally:
        ctx.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for b in ctx.bulks:
            b.join(timeout=120)
        progress.join(timeout=5)
        console.live(live_line(ctx, start, len(rows), done_count))
        console.end_live()
        print(detail_report(ctx, start))
        db.close()

    print(f"\nDone this run: accepted {ctx.stats.accepted:,} images. Run with --status for a full report.", flush=True)


if __name__ == "__main__":
    main()
