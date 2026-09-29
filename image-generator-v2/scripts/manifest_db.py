"""SQLite-backed manifest + resumable per-row progress tracking.

One connection shared by all worker threads, guarded by a lock. Writes are
committed every config.DB_COMMIT_EVERY operations (and on close) rather than
one at a time; a crash can lose at most that many recent records, whose files
are simply re-fetched on the next run.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    catalogue_id TEXT PRIMARY KEY,
    top_category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    query_seed TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    target INTEGER NOT NULL,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'  -- pending | in_progress | done | exhausted | gated_skip
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalogue_id TEXT NOT NULL,
    top_category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    license TEXT,
    sha256 TEXT NOT NULL UNIQUE,
    phash TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    watermark_score REAL,
    caption TEXT,
    objects TEXT,
    style TEXT,
    setting TEXT,
    lighting TEXT,
    quality_score REAL,
    adult_label TEXT,
    safety_status TEXT NOT NULL DEFAULT 'unreviewed',
    downloaded_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rejects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalogue_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    url TEXT,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS seen_urls (
    url_hash BLOB PRIMARY KEY
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bulk_shards (
    dataset TEXT NOT NULL,
    shard TEXT NOT NULL,
    PRIMARY KEY (dataset, shard)
);

CREATE TABLE IF NOT EXISTS subcategory_samples (
    top_category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    PRIMARY KEY (top_category, subcategory)
);

CREATE INDEX IF NOT EXISTS idx_images_catalogue ON images(catalogue_id);
CREATE INDEX IF NOT EXISTS idx_images_category ON images(top_category);
CREATE INDEX IF NOT EXISTS idx_images_downloaded ON images(downloaded_at);
CREATE INDEX IF NOT EXISTS idx_rejects_catalogue ON rejects(catalogue_id);
"""


def url_hash(url: str) -> bytes:
    return hashlib.sha1(url.encode("utf-8", "replace")).digest()


class ManifestDB:
    def __init__(self, path=config.MANIFEST_DB, readonly=False):
        self.path = path
        self.lock = threading.RLock()
        self._pending_writes = 0
        if readonly:
            # For --status alongside a running gather: never takes a write lock.
            self.conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            return
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(images)")}
        if "watermark_score" not in cols:
            self.conn.execute("ALTER TABLE images ADD COLUMN watermark_score REAL")

    def _wrote(self, n=1):
        self._pending_writes += n
        if self._pending_writes >= config.DB_COMMIT_EVERY:
            self.conn.commit()
            self._pending_writes = 0

    def commit(self):
        with self.lock:
            self.conn.commit()
            self._pending_writes = 0

    def upsert_rows(self, rows):
        """rows: iterable of (catalogue_id, top_category, subcategory, query_seed, source_tier, target, gated)."""
        with self.lock:
            self.conn.executemany(
                """INSERT INTO rows (catalogue_id, top_category, subcategory, query_seed, source_tier, target, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(catalogue_id) DO UPDATE SET
                       top_category=excluded.top_category,
                       subcategory=excluded.subcategory,
                       query_seed=excluded.query_seed,
                       source_tier=excluded.source_tier,
                       target=excluded.target,
                       status=CASE
                           WHEN excluded.status='gated_skip' THEN 'gated_skip'
                           WHEN rows.status='done' AND rows.accepted_count < excluded.target THEN 'pending'
                           WHEN rows.status IN ('done', 'exhausted') THEN rows.status
                           ELSE 'pending' END
                """,
                [(cid, tc, sub, q, tier, target, "gated_skip" if gated else "pending")
                 for cid, tc, sub, q, tier, target, gated in rows],
            )
            self.conn.commit()

    def pending_rows(self, category=None, include_exhausted=False):
        excluded = ("done", "gated_skip") if include_exhausted else ("done", "gated_skip", "exhausted")
        sql = f"SELECT * FROM rows WHERE status NOT IN ({','.join('?' * len(excluded))})"
        params = list(excluded)
        if category:
            sql += " AND top_category=?"
            params.append(category)
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def mark_row_status(self, catalogue_id, status):
        with self.lock:
            self.conn.execute("UPDATE rows SET status=? WHERE catalogue_id=?", (status, catalogue_id))
            self._wrote()

    def load_seen_urls(self) -> set[bytes]:
        with self.lock:
            return {r[0] for r in self.conn.execute("SELECT url_hash FROM seen_urls")}

    def add_seen_url(self, h: bytes):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO seen_urls (url_hash) VALUES (?)", (h,))
            self._wrote()

    def sha256_exists(self, sha256):
        with self.lock:
            return self.conn.execute("SELECT 1 FROM images WHERE sha256=?", (sha256,)).fetchone() is not None

    def all_phashes(self):
        with self.lock:
            return [r[0] for r in self.conn.execute("SELECT phash FROM images WHERE phash IS NOT NULL")]

    def try_insert_image(self, target: int, **fields) -> bool:
        """Insert the image only if its row is still under target and its
        sha256 is new. Returns False (inserting nothing) otherwise."""
        with self.lock:
            row = self.conn.execute(
                "SELECT accepted_count FROM rows WHERE catalogue_id=?", (fields["catalogue_id"],)
            ).fetchone()
            if row is not None and row[0] >= target:
                return False
            cols = ", ".join(fields.keys())
            placeholders = ", ".join("?" for _ in fields)
            try:
                self.conn.execute(f"INSERT INTO images ({cols}) VALUES ({placeholders})", tuple(fields.values()))
            except sqlite3.IntegrityError:
                return False
            self.conn.execute(
                "UPDATE rows SET accepted_count = accepted_count + 1 WHERE catalogue_id=?",
                (fields["catalogue_id"],),
            )
            self._wrote(2)
            return True

    def done_shards(self, dataset) -> set[str]:
        with self.lock:
            return {r[0] for r in self.conn.execute("SELECT shard FROM bulk_shards WHERE dataset=?", (dataset,))}

    def mark_shard_done(self, dataset, shard):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO bulk_shards (dataset, shard) VALUES (?, ?)", (dataset, shard))
            self.conn.commit()
            self._pending_writes = 0

    def subcategory_counts(self, source_name) -> dict[tuple[str, str], int]:
        with self.lock:
            return {
                (r[0], r[1]): r[2]
                for r in self.conn.execute(
                    "SELECT top_category, subcategory, COUNT(*) FROM images WHERE source_name=? "
                    "GROUP BY top_category, subcategory", (source_name,))
            }

    def row_accepted(self, catalogue_id) -> int:
        with self.lock:
            r = self.conn.execute("SELECT accepted_count FROM rows WHERE catalogue_id=?", (catalogue_id,)).fetchone()
            return r[0] if r else 0

    def log_reject(self, catalogue_id, reason, url=None):
        with self.lock:
            self.conn.execute(
                "INSERT INTO rejects (catalogue_id, reason, url, ts) VALUES (?, ?, ?, ?)",
                (catalogue_id, reason, url, time.time()),
            )
            self._wrote()

    def claim_sample(self, top_category, subcategory) -> bool:
        """True exactly once per (category, subcategory): the caller should
        write the sample image."""
        with self.lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO subcategory_samples (top_category, subcategory) VALUES (?, ?)",
                (top_category, subcategory),
            )
            self._wrote()
            return cur.rowcount == 1

    def totals(self):
        with self.lock:
            total_images = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            by_status = dict(self.conn.execute("SELECT status, COUNT(*) FROM rows GROUP BY status").fetchall())
            return total_images, by_status

    def per_category_totals(self):
        with self.lock:
            return self.conn.execute(
                "SELECT top_category, COUNT(*) as n FROM images GROUP BY top_category ORDER BY n DESC"
            ).fetchall()

    def reject_totals(self):
        with self.lock:
            return self.conn.execute(
                "SELECT reason, COUNT(*) as n FROM rejects GROUP BY reason ORDER BY n DESC"
            ).fetchall()

    def source_totals(self):
        with self.lock:
            return self.conn.execute(
                "SELECT source_name, COUNT(*) as n FROM images GROUP BY source_name ORDER BY n DESC"
            ).fetchall()

    def images_since(self, ts):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM images WHERE downloaded_at >= ?", (ts,)).fetchone()[0]

    def rate_per_hour(self, window_s=3600):
        """Accepted images/hour over the last window, measured over the time
        actually covered (so a 5-minute-old run isn't divided by a full hour)."""
        with self.lock:
            n, first, last = self.conn.execute(
                "SELECT COUNT(*), MIN(downloaded_at), MAX(downloaded_at) FROM images WHERE downloaded_at >= ?",
                (time.time() - window_s,),
            ).fetchone()
        if not n or last - first < 30:
            return None
        return n / ((last - first) / 3600)

    def close(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()
