"""
Megalith-10m gap filler: streams the CaptionEmporium captioned copy of
Megalith-10m (~10M CC0 / public-domain Flickr photos) from Hugging Face and
uses it only to fill catalogue rows the search workers haven't finished.

A row is open while its status is pending / in_progress / exhausted and it is
under target. A photo is kept only if its caption names an open row's
subcategory; it is credited to that row (so the search workers skip rows it
fills) and marked done when the row reaches target. Unmatched photos are
never downloaded. Captions are the InternVL2 short summaries.

Only the needed parquet columns are read (HfFileSystem range reads), so a
~300MB shard costs a few MB. Shards are recorded done like PD12M's.
"""

from __future__ import annotations

import collections
import queue
import random
import re
import threading
import time
import traceback

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

import config
import console
import ratelimit
from megalith_aliases import ALIASES
from sources import Candidate

DATASET = "megalith"
COLUMNS = ["url_highres", "url_source", "width", "height", config.MEGALITH_CAPTION_COLUMN]
OPEN_STATUSES = ("pending", "in_progress", "exhausted")


class AliasMatcher:
    """Maps a caption to open (top_category, subcategory) keys. Exact
    subcategory names are tried first, then the megalith_aliases patterns;
    within each pass the earliest mention in the caption wins."""

    def __init__(self, keys):
        self.keys = list(keys)
        order = sorted(range(len(self.keys)), key=lambda i: len(self.keys[i][1]), reverse=True)
        exact = [rf"(?P<k{i}>{re.escape(self.keys[i][1])}(?:e?s)?)" for i in order]
        alias = [rf"(?P<k{i}>(?:{'|'.join(ALIASES[self.keys[i][1]])})(?:e?s)?)"
                 for i in order if ALIASES.get(self.keys[i][1])]
        self.passes = [re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for p in (exact, alias) if p]

    def candidates(self, caption: str):
        for rx in self.passes:
            for m in rx.finditer(caption, 0, config.MEGALITH_MATCH_MAX_CHARS):
                yield self.keys[int(m.lastgroup[1:])]


class PatternMatcher:
    """Phase 1.5 scene rows: each open key has its own caption regex (the
    row's query_seed); keys are yielded in order of where they first match."""

    def __init__(self, keys, patterns: dict):
        self.keys = list(keys)
        self.rx = {k: re.compile(patterns[k], re.I) for k in self.keys}

    def candidates(self, caption: str):
        head = caption[:config.PHASE15_SCENE_MATCH_CHARS]  # the main subject comes first
        hits = [(m.start(), k) for k in self.keys if (m := self.rx[k].search(head))]
        for _, k in sorted(hits):
            yield k


class MegalithImporter:
    name = DATASET

    def __init__(self, ctx, process_fn, workers: int, phase15: bool = False):
        """phase15=False fills the main catalogue's open rows by subcategory name
        and aliases; phase15=True fills only the Phase 1.5 scene rows, matching
        each row's caption regex, with its own shard progress so every shard
        is scanned again."""
        self.ctx = ctx
        self.phase15 = phase15
        self.shard_key = config.PHASE15_MEGALITH_DATASET if phase15 else DATASET
        self.name = self.shard_key
        self.process_fn = process_fn
        self.workers = workers
        self.q: queue.Queue = queue.Queue(maxsize=workers * 50)
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.unsafe = re.compile(config.UNSAFE_CAPTION_PATTERN, re.I)
        self.skipped = collections.Counter()
        self.total = 0
        self.shards_total = 0
        self.shards_done = 0
        self.threads: list[threading.Thread] = []

        # (top_category, subcategory) -> {catalogue_id: [row, remaining]}
        self.open: dict[tuple[str, str], dict[str, list]] = collections.defaultdict(dict)
        self.patterns: dict[tuple[str, str], str] = {}
        for r in ctx.db.pending_rows(include_exhausted=True):
            if r["top_category"] in config.ADULT_CATEGORY_NAMES:
                continue
            is_p15 = r["catalogue_id"].startswith(config.PHASE15_ID_PREFIX)
            if is_p15 != phase15 or (phase15 and r["source_tier"] != "megalith"):
                continue
            if phase15:
                self.patterns[(r["top_category"], r["subcategory"])] = r["query_seed"]
            room = r["target"] - r["accepted_count"]
            if room > 0:
                self.open[(r["top_category"], r["subcategory"])][r["catalogue_id"]] = [dict(r), room]
        self.matcher = self._make_matcher(list(self.open)) if self.open else None
        # Set when a subcategory closes or reopens. The regex only reports one
        # subcategory per caption position, so a closed one would keep hiding
        # others that share its words ("parrot" in Pets and Birds) until the
        # matcher is rebuilt from the open set.
        self.matcher_stale = False

    def _make_matcher(self, keys):
        return PatternMatcher(keys, self.patterns) if self.phase15 else AliasMatcher(keys)

    # -- public -------------------------------------------------------------

    def start(self):
        self.threads = [threading.Thread(target=self._feeder, name="megalith-feeder", daemon=True)]
        self.threads += [threading.Thread(target=self._worker, name=f"megalith-{i}", daemon=True)
                         for i in range(self.workers)]
        for t in self.threads:
            t.start()

    def join(self, timeout=None):
        for t in self.threads:
            t.join(timeout)

    def status_line(self) -> str:
        with self.lock:
            skipped = ", ".join(f"{k} {v:,}" for k, v in self.skipped.most_common(3))
            open_rows = sum(len(v) for v in self.open.values())
            return (f"{self.name}: {self.total:,} kept | open rows {open_rows:,} | shards "
                    f"{self.shards_done}/{self.shards_total} | queue {self.q.qsize()} | "
                    f"pre-download skips: {skipped or '-'}")

    # -- internals ----------------------------------------------------------

    def _finished(self) -> bool:
        with self.lock:
            return not self.open

    def _pick_key(self, caption: str):
        """First subcategory named near the start of the caption (where the
        main subject is) that still has an open row."""
        if self.matcher_stale:
            with self.lock:
                keys = list(self.open)
                self.matcher_stale = False
            if keys:
                self.matcher = self._make_matcher(keys)
        for key in self.matcher.candidates(caption):
            with self.lock:
                if key in self.open:
                    return key
        return None

    def _claim_row(self, key):
        """Reserve one slot in an open row of `key`; returns the row or None."""
        with self.lock:
            rows = self.open.get(key)
            if not rows:
                return None
            cid = random.choice(list(rows))
            rows[cid][1] -= 1
            row = rows[cid][0]
            if rows[cid][1] <= 0:
                del rows[cid]
                if not rows:
                    del self.open[key]
                    self.matcher_stale = True
            return row

    def _release_row(self, key, row):
        """Give back a slot reserved by _claim_row after a rejected download."""
        room = row["target"] - self.ctx.db.row_accepted(row["catalogue_id"])
        with self.lock:
            if key not in self.open:
                self.matcher_stale = True
            rows = self.open.setdefault(key, {})
            if room <= 0:
                rows.pop(row["catalogue_id"], None)
                if not rows:
                    del self.open[key]
                return
            entry = rows.setdefault(row["catalogue_id"], [row, 0])
            entry[1] = min(entry[1] + 1, room)

    def _skip(self, reason):
        with self.lock:
            self.skipped[reason] += 1

    def _feeder(self):
        ctx = self.ctx
        try:
            if self.matcher is None:
                console.log("megalith: no open rows to fill")
                return
            fs = HfFileSystem()
            repo = f"datasets/{config.MEGALITH_HF_REPO}"
            names = sorted(p.removeprefix(repo + "/") for p in fs.glob(f"{repo}/train/*.parquet"))
            done = ctx.db.done_shards(self.shard_key)
            todo = [n for n in names if n not in done]
            random.Random(0).shuffle(todo)
            with self.lock:
                self.shards_total = len(names)
                self.shards_done = len(names) - len(todo)
            min_short = config.SOURCE_MIN_SHORT_SIDE[DATASET]
            for name in todo:
                if ctx.stop.is_set() or self._finished():
                    break
                with fs.open(f"{repo}/{name}", "rb") as fh:
                    rows = pq.read_table(fh, columns=COLUMNS).to_pylist()
                random.shuffle(rows)
                complete = True
                for r in rows:
                    if ctx.stop.is_set() or self._finished():
                        complete = False
                        break
                    if min(int(r["width"] or 0), int(r["height"] or 0)) < min_short:
                        continue
                    caption = r[config.MEGALITH_CAPTION_COLUMN] or ""
                    if self.unsafe.search(caption):
                        self._skip("unsafe_caption")
                        continue
                    key = self._pick_key(caption)
                    if key is None:
                        self._skip("no_open_subcategory")
                        continue
                    cand = Candidate(r["url_highres"], "flickr.com", r["url_source"], "flickr_public_domain", caption)
                    while not ctx.stop.is_set():
                        try:
                            self.q.put((cand, key), timeout=1)
                            break
                        except queue.Full:
                            pass
                while self.q.unfinished_tasks and not ctx.stop.is_set():
                    time.sleep(1)
                if complete and not ctx.stop.is_set():
                    ctx.db.mark_shard_done(self.shard_key, name)
                    with self.lock:
                        self.shards_done += 1
                    console.log(f"megalith: finished {name} ({self.shards_done}/{self.shards_total})")
        except Exception:
            console.log("megalith feeder stopped on error:\n" + traceback.format_exc())
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
                if ctx.stop.is_set() or ratelimit.is_cooling(ratelimit.domain_of(cand.url)):
                    continue
                row = self._claim_row(key)
                if row is None:
                    continue
                if not ctx.claim_url(cand.url):
                    ctx.stats.reject("url_already_tried")
                    self._release_row(key, row)
                    continue
                if self.process_fn(ctx, cand, row, DATASET):
                    with self.lock:
                        self.total += 1
                    if ctx.db.row_accepted(row["catalogue_id"]) >= row["target"]:
                        ctx.db.mark_row_status(row["catalogue_id"], "done")
                else:
                    self._release_row(key, row)
            except Exception:
                console.log("megalith worker error (continuing):\n" + traceback.format_exc())
            finally:
                self.q.task_done()


class MegalithGeneralImporter(MegalithImporter):
    """General expansion (gather_images.py --expand): keeps Megalith photos that
    aren't tied to catalogue rows, filed like PD12M -- under the first catalogue
    subcategory named in the caption, or "General (Megalith)" -- with per-key
    and total caps. Its own shard progress, so every shard is scanned."""

    CATALOGUE_ID = "MEGALITH-GEN"

    def __init__(self, ctx, process_fn, workers: int, catalogue):
        super().__init__(ctx, process_fn, workers, phase15=True)  # phase15=True -> no main-catalogue rows loaded
        from pd12m import CaptionMatcher
        self.open = {}
        self.matcher = CaptionMatcher(catalogue)
        self.shard_key = self.name = config.MEGALITH_GENERAL_DATASET
        with ctx.db.lock:
            rows = ctx.db.conn.execute(
                "SELECT top_category, subcategory, COUNT(*) FROM images WHERE catalogue_id=? GROUP BY 1, 2",
                (self.CATALOGUE_ID,)).fetchall()
        self.counts = collections.Counter({(a, b): n for a, b, n in rows})
        self.total = sum(self.counts.values())

    def status_line(self) -> str:
        with self.lock:
            skipped = ", ".join(f"{k} {v:,}" for k, v in self.skipped.most_common(3))
            return (f"{self.name}: {self.total:,}/{config.MEGALITH_GENERAL_MAX_IMAGES:,} kept | shards "
                    f"{self.shards_done}/{self.shards_total} | queue {self.q.qsize()} | pre-download skips: {skipped or '-'}")

    def _cap(self, key) -> int:
        return (config.MEGALITH_GENERAL_UNMATCHED_MAX if key[0] == config.MEGALITH_GENERAL_UNMATCHED_CATEGORY
                else config.MEGALITH_GENERAL_MAX_PER_SUBCATEGORY)

    def _finished(self) -> bool:
        with self.lock:
            return self.total >= config.MEGALITH_GENERAL_MAX_IMAGES

    def _pick_key(self, caption: str):
        key = self.matcher.match(caption) or (config.MEGALITH_GENERAL_UNMATCHED_CATEGORY, "unmatched")
        with self.lock:
            return key if self.counts[key] < self._cap(key) else None

    def _claim_row(self, key):
        with self.lock:
            if self.counts[key] >= self._cap(key) or self.total >= config.MEGALITH_GENERAL_MAX_IMAGES:
                return None
            self.counts[key] += 1  # reserved; given back by _release_row if the download is rejected
        return {"catalogue_id": self.CATALOGUE_ID, "top_category": key[0], "subcategory": key[1], "target": 10 ** 12}

    def _release_row(self, key, row):
        with self.lock:
            self.counts[key] -= 1
