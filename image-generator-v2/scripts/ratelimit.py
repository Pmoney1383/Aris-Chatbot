"""Thread-safe per-key pacing and 429 cooldowns, shared by sources.py (search
API calls) and gather_images.py (image downloads) across all worker threads."""

import collections
import threading
import time
import urllib.parse

import config

_lock = threading.Lock()
_cooldowns: dict[str, float] = {}
_next_slot: dict[str, float] = {}

_WINDOWS = {"hour": 3600, "day": 86400}


class _ByteWindow:
    """Rolling byte totals for one host over the last hour and day."""

    def __init__(self):
        self.events = {w: collections.deque() for w in _WINDOWS}
        self.totals = {w: 0 for w in _WINDOWS}

    def prune(self, now):
        for w, span in _WINDOWS.items():
            q = self.events[w]
            while q and now - q[0][0] > span:
                self.totals[w] -= q.popleft()[1]

    def add(self, now, n):
        for w in _WINDOWS:
            self.events[w].append((now, n))
            self.totals[w] += n


_bytes: dict[str, _ByteWindow] = collections.defaultdict(_ByteWindow)


def _over_budget(key: str, now: float) -> bool:
    budget = config.DOWNLOAD_BYTE_BUDGETS.get(key)
    if not budget:
        return False
    win = _bytes[key]
    win.prune(now)
    return any(win.totals[w] >= limit for w, limit in budget.items())


def record_bytes(key: str, n: int) -> None:
    if key in config.DOWNLOAD_BYTE_BUDGETS:
        with _lock:
            _bytes[key].add(time.time(), n)


def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def interval_for(key: str) -> float:
    return config.DOMAIN_MIN_INTERVAL_SECONDS.get(key, config.DEFAULT_MIN_INTERVAL_SECONDS)


def is_cooling(key: str) -> bool:
    """True while `key` is backing off after a 429 or is over its byte budget."""
    with _lock:
        now = time.time()
        return now < _cooldowns.get(key, 0) or _over_budget(key, now)


def cool_down(key: str, seconds: float) -> None:
    with _lock:
        _cooldowns[key] = max(_cooldowns.get(key, 0), time.time() + seconds)


def parse_retry_after(header_value, default=30.0, cap=900.0) -> float:
    if not header_value:
        return default
    try:
        return min(float(header_value), cap)
    except ValueError:
        return default


def reserve(key: str, max_wait: float | None = None) -> bool:
    """Claim the next request slot for `key`, sleeping until it arrives.

    Slots are handed out under a lock, so concurrent workers queue up at
    exactly one request per interval instead of all firing at once. With
    `max_wait`, returns False without claiming anything if the slot is
    further away than that -- the caller should go do something else."""
    interval = interval_for(key)
    with _lock:
        now = time.time()
        if now < _cooldowns.get(key, 0) or _over_budget(key, now):
            return False
        slot = max(now, _next_slot.get(key, 0))
        if max_wait is not None and slot - now > max_wait:
            return False
        _next_slot[key] = slot + interval
    wait = slot - time.time()
    if wait > 0:
        time.sleep(wait)
    return True
