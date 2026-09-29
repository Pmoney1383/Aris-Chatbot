"""
PD12M bulk importer: streams Spawning's PD12M metadata shards from Hugging
Face (public, no login) and feeds qualifying images through the same
download/dedup/watermark/encode pipeline as the search workers.

Per shard (100k rows, ~19MB parquet):
  - keep rows with short side >= MIN_SHORT_SIDE and long side <= PD12M_MAX_LONG_EDGE
  - skip captions matching UNSAFE_CAPTION_PATTERN (before download)
  - assign a catalogue (category, subcategory) by matching subcategory names
    in the caption; unmatched rows go to PD12M_UNMATCHED_CATEGORY
  - enforce PD12M_MAX_IMAGES / PD12M_MAX_PER_SUBCATEGORY / PD12M_UNMATCHED_MAX
The PD12M caption is stored in the manifest's caption column.

A shard is recorded as done only after every queued row has been processed,
so an interrupted shard is re-scanned next run (already-tried URLs are
skipped quickly via the seen-URL index).
"""

from __future__ import annotations

import collections
import io
import queue
import random
import re
import threading
import time
import traceback

import pyarrow.parquet as pq
import requests

import config
import console
import ratelimit
from sources import Candidate

DATASET = "pd12m"


class CaptionMatcher:
    """Maps a caption to the first catalogue subcategory named in it (longest
    names first, so "sports car" wins over "car"). Adult rows are excluded."""

    def __init__(self, catalogue_rows):
        self.mapping: dict[str, tuple[str, str]] = {}
        for r in catalogue_rows:
            if r["top_category"] in config.ADULT_CATEGORY_NAMES:
                continue
            self.mapping.setdefault(r["subcategory"].lower(), (r["top_category"], r["subcategory"]))
        phrases = sorted(self.mapping, key=len, reverse=True)
        self.rx = re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")(?:e?s)?\b", re.I)

    def match(self, caption: str):
        m = self.rx.search(caption or "")
        return self.mapping[m.group(1).lower()] if m else None


class PD12MImporter:
    name = DATASET

    def __init__(self, ctx, matcher: CaptionMatcher, process_fn, workers: int):
        self.ctx = ctx
        self.matcher = matcher
        self.process_fn = process_fn
        self.workers = workers
        self.q: queue.Queue = queue.Queue(maxsize=workers * 50)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.counts = collections.Counter(ctx.db.subcategory_counts(DATASET))
        self.total = sum(self.counts.values())
        self.unsafe = re.compile(config.UNSAFE_CAPTION_PATTERN, re.I)
        self.skipped = collections.Counter()
        self.shards_total = 0
        self.shards_done = 0
        self.current_shard = None
        self.threads: list[threading.Thread] = []

    # -- public -------------------------------------------------------------

    def start(self):
        self.threads = [threading.Thread(target=self._feeder, name="pd12m-feeder", daemon=True)]
        self.threads += [threading.Thread(target=self._worker, name=f"pd12m-{i}", daemon=True)
                         for i in range(self.workers)]
        for t in self.threads:
            t.start()

    def join(self, timeout=None):
        for t in self.threads:
            t.join(timeout)

    def status_line(self) -> str:
        with self.lock:
            skipped = ", ".join(f"{k} {v:,}" for k, v in self.skipped.most_common(3))
            return (f"pd12m: {self.total:,}/{config.PD12M_MAX_IMAGES:,} kept | shards "
                    f"{self.shards_done}/{self.shards_total} | queue {self.q.qsize()} | "
                    f"pre-download skips: {skipped or '-'}")

    # -- internals ----------------------------------------------------------

    def _full(self) -> bool:
        with self.lock:
            return self.total >= config.PD12M_MAX_IMAGES

    def _has_room(self, key) -> bool:
        with self.lock:
            if self.total >= config.PD12M_MAX_IMAGES:
                return False
            cap = config.PD12M_UNMATCHED_MAX if key[0] == config.PD12M_UNMATCHED_CATEGORY else config.PD12M_MAX_PER_SUBCATEGORY
            return self.counts[key] < cap

    def _skip(self, reason):
        with self.lock:
            self.skipped[reason] += 1

    def _shard_names(self) -> list[str]:
        r = requests.get(f"https://huggingface.co/api/datasets/{config.PD12M_HF_REPO}", timeout=60)
        r.raise_for_status()
        return sorted(s["rfilename"] for s in r.json().get("siblings", [])
                      if s["rfilename"].startswith("metadata/") and s["rfilename"].endswith(".parquet"))

    def _load_shard(self, name: str):
        url = f"https://huggingface.co/datasets/{config.PD12M_HF_REPO}/resolve/main/{name}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                cols = ["url", "caption", "width", "height", "mime_type", "license", "source"]
                return pq.read_table(io.BytesIO(r.content), columns=cols).to_pylist()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def _feeder(self):
        ctx = self.ctx
        try:
            names = self._shard_names()
            done = ctx.db.done_shards(DATASET)
            todo = [n for n in names if n not in done]
            random.Random(0).shuffle(todo)  # shards are grouped by source; shuffle for variety
            with self.lock:
                self.shards_total = len(names)
                self.shards_done = len(names) - len(todo)
            for name in todo:
                if ctx.stop.is_set() or self._full():
                    break
                self.current_shard = name
                rows = self._load_shard(name)
                random.shuffle(rows)
                complete = True
                for r in rows:
                    if ctx.stop.is_set() or self._full():
                        complete = False
                        break
                    w, h = int(r["width"] or 0), int(r["height"] or 0)
                    if min(w, h) < config.MIN_SHORT_SIDE or max(w, h) > config.PD12M_MAX_LONG_EDGE:
                        continue
                    if r["mime_type"] not in ("image/jpeg", "image/png", "image/webp"):
                        continue
                    caption = r["caption"] or ""
                    if self.unsafe.search(caption):
                        self._skip("unsafe_caption")
                        continue
                    key = self.matcher.match(caption) or (config.PD12M_UNMATCHED_CATEGORY, "unmatched")
                    if not self._has_room(key):
                        self._skip("category_cap")
                        continue
                    cand = Candidate(r["url"], r["source"] or "pd12m", r["url"], r["license"], caption)
                    while not ctx.stop.is_set():
                        try:
                            self.q.put((cand, key), timeout=1)
                            break
                        except queue.Full:
                            pass
                # Wait for this shard's queued rows to finish before marking it done.
                while self.q.unfinished_tasks and not ctx.stop.is_set():
                    time.sleep(1)
                if complete and not ctx.stop.is_set():
                    ctx.db.mark_shard_done(DATASET, name)
                    with self.lock:
                        self.shards_done += 1
                    console.log(f"pd12m: finished {name} ({self.shards_done}/{self.shards_total})")
        except Exception:
            console.log("pd12m feeder stopped on error:\n" + traceback.format_exc())
        finally:
            self.done.set()

    def _worker(self):
        ctx = self.ctx
        while not (self.done.is_set() and self.q.empty()):
            try:
                cand, key = self.q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if ctx.stop.is_set() or not self._has_room(key):
                    continue
                if ratelimit.is_cooling(ratelimit.domain_of(cand.url)):
                    continue
                if not ctx.claim_url(cand.url):
                    ctx.stats.reject("url_already_tried")
                    continue
                row = {"catalogue_id": "PD12M", "top_category": key[0], "subcategory": key[1],
                       "target": 10 ** 12}
                if self.process_fn(ctx, cand, row, DATASET):
                    with self.lock:
                        self.counts[key] += 1
                        self.total += 1
            except Exception:
                console.log("pd12m worker error (continuing):\n" + traceback.format_exc())
            finally:
                self.q.task_done()
