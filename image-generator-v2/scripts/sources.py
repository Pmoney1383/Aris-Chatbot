"""
Image-search connectors used by gather_images.py.

Every connector implements:

    search(query: str, page: int) -> list[Candidate] | None

A list (possibly empty) is a real answer for that page. None means the source
is busy (its rate-limit slot is too far away, or it is cooling down after a
429) -- the caller should try another source and come back later, without
counting this as an empty page.

Connectors that need an API key report available() -> False when the key is
missing, so gather_images.py just skips them.

IMPORTANT: there is deliberately no adult/NSFW connector here. The
"Adult 18+ Safety Research" category is blocked in gather_images.py before
any connector is ever called for it. Bing requests always force SafeSearch
strict on top of that.
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests

import config
import ratelimit

_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": config.USER_AGENT})
        _thread_local.session = s
    return s


BUSY = None


def paced_get(url, rate_key, **kwargs):
    """GET that waits for `rate_key`'s next slot (up to MAX_SEARCH_WAIT_SECONDS)
    and turns a 429 into a cooldown for that key. Returns a Response, or BUSY
    if the key had no slot available or got rate-limited."""
    if not ratelimit.reserve(rate_key, max_wait=config.MAX_SEARCH_WAIT_SECONDS):
        return BUSY
    try:
        r = _session().get(url, timeout=config.REQUEST_TIMEOUT_SECONDS, **kwargs)
    except Exception:
        return requests.Response()  # status_code None -> treated as an empty page
    if r.status_code == 429:
        ratelimit.cool_down(rate_key, ratelimit.parse_retry_after(r.headers.get("Retry-After")))
        return BUSY
    return r


@dataclass
class Candidate:
    url: str
    source_domain: str
    source_url: str
    license: Optional[str] = None
    caption: Optional[str] = None


class BaseSource:
    name = "base"
    # If set, the source is only queried for rows in these top categories.
    categories: Optional[frozenset] = None
    # Subject-specific sources are searched by subcategory ("lion"), not the
    # full query phrase ("lion close-up portrait"). All rows of a subcategory
    # then share one page cursor, so they walk through different result
    # pages instead of each re-fetching page 1.
    by_subcategory = False

    def available(self) -> bool:
        return True

    def search(self, query: str, page: int):
        raise NotImplementedError


def _json_or_none(r):
    if r is BUSY:
        return BUSY
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except Exception:
        return {}


class OpenverseSource(BaseSource):
    """https://api.openverse.org -- CC-licensed aggregate. No key needed, but
    anonymous limits are low; not in the default SOURCE_ORDER."""

    name = "openverse"

    def search(self, query, page):
        data = _json_or_none(paced_get(
            "https://api.openverse.org/v1/images/", "openverse_api",
            params={"q": query, "page": page, "page_size": config.SOURCE_PAGE_SIZE["openverse"]},
        ))
        if data is BUSY:
            return BUSY
        out = []
        for item in data.get("results", []):
            url = item.get("url")
            if not url:
                continue
            landing = item.get("foreign_landing_url", url)
            out.append(Candidate(url, ratelimit.domain_of(landing) or "openverse.org", landing, item.get("license")))
        return out


class WikimediaCommonsSource(BaseSource):
    """Wikimedia Commons search API. No key needed. Asks for a server-side
    thumbnail capped at SOURCE_DOWNLOAD_MAX_EDGE wide so we don't pull
    multi-MB originals (the API returns the original when it's smaller)."""

    name = "wikimedia"

    def search(self, query, page):
        per_page = config.SOURCE_PAGE_SIZE["wikimedia"]
        data = _json_or_none(paced_get(
            "https://commons.wikimedia.org/w/api.php", "wikimedia_api",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrlimit": per_page,
                "gsroffset": (page - 1) * per_page,
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": config.SOURCE_DOWNLOAD_MAX_EDGE,
            },
        ))
        if data is BUSY:
            return BUSY
        out = []
        for page_data in data.get("query", {}).get("pages", {}).values():
            infos = page_data.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/") or mime == "image/svg+xml":
                continue
            url = info.get("thumburl") or info["url"]
            out.append(Candidate(url, "commons.wikimedia.org", info.get("descriptionurl", info["url"]), "wikimedia_commons"))
        return out


class PexelsSource(BaseSource):
    name = "pexels"

    def available(self):
        return bool(config.PEXELS_API_KEY)

    def search(self, query, page):
        r = paced_get(
            "https://api.pexels.com/v1/search", "pexels_api",
            params={"query": query, "page": page, "per_page": config.SOURCE_PAGE_SIZE["pexels"]},
            headers={"Authorization": config.PEXELS_API_KEY},
        )
        if r is not BUSY and r.headers.get("X-Ratelimit-Remaining") == "0":
            # Hourly/monthly quota used up: back off until Pexels says it resets.
            try:
                reset_in = float(r.headers.get("X-Ratelimit-Reset", "0")) - time.time()
            except ValueError:
                reset_in = 3600
            ratelimit.cool_down("pexels_api", max(reset_in, 60))
        data = _json_or_none(r)
        if data is BUSY:
            return BUSY
        out = []
        for p in data.get("photos", []):
            base_url = p.get("src", {}).get("original")
            if not base_url:
                continue
            # Pexels' CDN resizes on the fly: bound the long edge server-side
            # instead of downloading a 20-30MB original.
            edge = config.SOURCE_DOWNLOAD_MAX_EDGE
            url = f"{base_url}?auto=compress&cs=tinysrgb&fit=max&w={edge}&h={edge}"
            out.append(Candidate(url, "pexels.com", p.get("url", base_url), "pexels_free_license"))
        return out


class PixabaySource(BaseSource):
    name = "pixabay"

    def available(self):
        return bool(config.PIXABAY_API_KEY)

    def search(self, query, page):
        data = _json_or_none(paced_get(
            "https://pixabay.com/api/", "pixabay_api",
            params={
                "key": config.PIXABAY_API_KEY,
                "q": query[:100],
                "image_type": "photo",
                "safesearch": "true",
                "per_page": config.SOURCE_PAGE_SIZE["pixabay"],
                "page": page,
            },
        ))
        if data is BUSY:
            return BUSY
        out = []
        for h in data.get("hits", []):
            url = h.get("largeImageURL")
            if not url:
                continue
            out.append(Candidate(url, "pixabay.com", h.get("pageURL", url), "pixabay_content_license"))
        return out


class UnsplashSource(BaseSource):
    name = "unsplash"

    def available(self):
        return bool(config.UNSPLASH_ACCESS_KEY)

    def search(self, query, page):
        data = _json_or_none(paced_get(
            "https://api.unsplash.com/search/photos", "unsplash_api",
            params={"query": query, "page": page, "per_page": config.SOURCE_PAGE_SIZE["unsplash"],
                    "content_filter": "high"},
            headers={"Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}"},
        ))
        if data is BUSY:
            return BUSY
        out = []
        for item in data.get("results", []):
            raw = item.get("urls", {}).get("raw")
            if not raw:
                continue
            edge = config.SOURCE_DOWNLOAD_MAX_EDGE
            url = f"{raw}&fm=jpg&q=85&fit=max&w={edge}&h={edge}"
            out.append(Candidate(url, "unsplash.com", item.get("links", {}).get("html", raw), "unsplash_license"))
        return out


class BingSource(BaseSource):
    """
    Bing image search results page, parsed for each result's original image
    URL ("murl"). Always requested with SafeSearch strict (adlt=strict) and
    restricted to large photos. Results come from arbitrary websites, so they
    are lower-trust: stock-photo hosts are skipped before download
    (config.BLOCKED_IMAGE_DOMAINS) and everything still passes the watermark,
    resolution and dedup checks.

    Fragile by nature: if Bing changes its markup, this returns [] and the
    other sources carry on.
    """

    name = "bing"
    _murl_re = re.compile(r'"murl":"(https?://[^"]+)"')

    def available(self):
        return config.ENABLE_BING

    def search(self, query, page):
        per_page = config.SOURCE_PAGE_SIZE["bing"]
        r = paced_get(
            "https://www.bing.com/images/async", "bing_search",
            params={
                "q": query,
                "first": (page - 1) * per_page,
                "count": per_page,
                "adlt": "strict",
                "qft": "+filterui:photo-photo+filterui:imagesize-large",
            },
            headers={"User-Agent": config.BROWSER_USER_AGENT},
        )
        if r is BUSY:
            return BUSY
        if r.status_code != 200:
            return []
        out = []
        seen = set()
        for raw in self._murl_re.findall(html.unescape(r.text)):
            try:
                url = json.loads(f'"{raw}"')
            except ValueError:
                url = raw
            if url in seen:
                continue
            seen.add(url)
            out.append(Candidate(url, ratelimit.domain_of(url) or "unknown", url, None))
        return out


class INaturalistSource(BaseSource):
    """iNaturalist research-grade observations (no key). The subcategory is
    resolved to a taxon first ("lion" -> Panthera leo); a plain text search
    matches loosely ("lion" also hits dandelions, "lion's tooth").

    Photo downloads are also capped by bytes (config.DOWNLOAD_BYTE_BUDGETS)
    to stay under iNaturalist's media guidance."""

    name = "inaturalist"
    categories = frozenset({
        "Animals & Wildlife", "Birds", "Marine Life", "Insects & Small Creatures",
        "Reptiles & Amphibians", "Plants & Botany",
    })
    by_subcategory = True
    MAX_RESULTS = 10_000  # API refuses page * per_page beyond this

    def __init__(self):
        self._taxa: dict[str, Optional[int]] = {}
        self._lock = threading.Lock()

    def _taxon_id(self, name: str):
        with self._lock:
            if name in self._taxa:
                return self._taxa[name]
        data = _json_or_none(paced_get(
            "https://api.inaturalist.org/v1/taxa/autocomplete", "inat_api",
            params={"q": name, "per_page": 10, "is_active": "true"},
        ))
        if data is BUSY:
            return BUSY
        # Autocomplete ranks loosely ("moose" -> Mosses first), so only accept
        # a taxon whose common name is the subcategory itself (or its plural,
        # or starts with it: "frog" -> "Frogs and Toads").
        q = name.lower()
        forms = {q, q + "s", q + "es"} | ({q[:-1] + "ies"} if q.endswith("y") else set())
        taxon = None
        for t in data.get("results") or []:
            common = (t.get("preferred_common_name") or "").lower()
            if common in forms or any(common.startswith(f + " ") for f in forms) or t.get("name", "").lower() == q:
                taxon = t["id"]
                break
        with self._lock:
            self._taxa[name] = taxon
        return taxon

    def search(self, query, page):
        per_page = config.SOURCE_PAGE_SIZE["inaturalist"]
        if page * per_page > self.MAX_RESULTS:
            return []
        taxon = self._taxon_id(query)
        if taxon is BUSY:
            return BUSY
        if taxon is None:
            return []
        data = _json_or_none(paced_get(
            "https://api.inaturalist.org/v1/observations", "inat_api",
            params={
                "taxon_id": taxon, "photos": "true", "quality_grade": "research",
                "photo_license": "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd",
                "per_page": per_page, "page": page, "order_by": "votes",
            },
        ))
        if data is BUSY:
            return BUSY
        out = []
        for obs in data.get("results", []):
            photos = obs.get("photos") or []
            if not photos or not photos[0].get("url"):
                continue  # first photo only; the rest are usually the same subject
            p = photos[0]
            dims = p.get("original_dimensions") or {}
            if dims and min(dims.get("width", 0), dims.get("height", 0)) < config.MIN_SHORT_SIDE:
                continue  # skip before download
            url = p["url"].replace("/square.", "/original.")
            out.append(Candidate(url, "inaturalist.org", obs.get("uri", url), p.get("license_code")))
        return out


class NasaImagesSource(BaseSource):
    """NASA Image and Video Library (no key). NASA imagery is generally not
    copyrighted; a minority of items are third-party."""

    name = "nasa"
    categories = frozenset({"Space & Astronomy", "Weather & Sky"})
    by_subcategory = True

    def search(self, query, page):
        data = _json_or_none(paced_get(
            "https://images-api.nasa.gov/search", "nasa_api",
            params={"q": query, "media_type": "image", "page": page, "page_size": config.SOURCE_PAGE_SIZE["nasa"]},
        ))
        if data is BUSY:
            return BUSY
        out = []
        for item in data.get("collection", {}).get("items", []):
            links = item.get("links") or []
            meta = (item.get("data") or [{}])[0]
            if not links or "~thumb" not in links[0].get("href", ""):
                continue
            url = links[0]["href"].replace("~thumb", "~orig")
            nasa_id = meta.get("nasa_id", "")
            out.append(Candidate(url, "nasa.gov", f"https://images.nasa.gov/details/{nasa_id}", "nasa_media"))
        return out


class CommonsCategorySource(BaseSource):
    """Walks one Wikimedia Commons category (found by searching the row's
    query) and its subcategories, breadth-first. Every file in that tree is
    labeled by the concept, which makes this the source of specific, labeled
    photos (a car model, a dish) at full resolution.

    Subcategories are followed only while their names still contain the
    resolved category's name and none of COMMONS_SKIP_SUBCATEGORY_WORDS, so
    "Pizza" reaches "Pizza Margherita in Naples" but not "Pizza boxes".
    Each search() call returns the next batch; [] once the tree is exhausted."""

    name = "commons_cat"
    MAX_DEPTH = 3

    def __init__(self):
        self._walks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _api(self, params):
        return _json_or_none(paced_get("https://commons.wikimedia.org/w/api.php", "wikimedia_api",
                                       params={**params, "format": "json"}))

    def _resolve(self, query):
        data = self._api({"action": "query", "list": "search", "srsearch": query, "srnamespace": 14, "srlimit": 10})
        if data is BUSY:
            return BUSY
        titles = [h["title"][len("Category:"):] for h in data.get("query", {}).get("search", [])]
        q = query.lower()
        exact = [t for t in titles if t.lower() == q]
        return (exact or titles or [None])[0]

    @staticmethod
    def _stem(title: str) -> str:
        return re.sub(r"\s*\(.*?\)\s*", " ", title).strip().lower()

    def _next_category(self, walk):
        """Advance to the next queued category; False when the tree is exhausted. Caller holds the lock."""
        while walk["queue"] and walk["queue"][0][0] in walk["seen"]:
            walk["queue"].pop(0)
        if not walk["queue"]:
            return False
        walk["current"] = walk["queue"].pop(0)
        walk["seen"].add(walk["current"][0])
        walk["cont"] = None
        return True

    def _follow(self, sub: str, root_stem: str) -> bool:
        low = sub.lower()
        return (re.search(r"\b" + re.escape(root_stem) + r"\b", low) is not None
                and not any(re.search(r"\b" + re.escape(w), low) for w in config.COMMONS_SKIP_SUBCATEGORY_WORDS))

    def search(self, query, page):
        with self._lock:
            walk = self._walks.get(query)
        if walk is None:
            root = self._resolve(query)
            if root is BUSY:
                return BUSY
            walk = {"root_stem": self._stem(root) if root else "", "queue": [(root, 0)] if root else [],
                    "seen": set(), "cont": None, "current": None}
            with self._lock:
                walk = self._walks.setdefault(query, walk)
        min_short = config.SOURCE_MIN_SHORT_SIDE.get(self.name, config.MIN_SHORT_SIDE)
        # Category pages that hold only subcategories yield no files; keep walking
        # (bounded) so the caller never mistakes such a page for an exhausted source.
        for _ in range(config.COMMONS_MAX_CALLS_PER_SEARCH):
            with self._lock:
                if walk["current"] is None and not self._next_category(walk):
                    return []
                (cat, depth), cont = walk["current"], walk["cont"]
            data = self._api({"action": "query", "generator": "categorymembers", "gcmtitle": "Category:" + cat,
                              "gcmtype": "file|subcat", "gcmlimit": config.SOURCE_PAGE_SIZE["commons_cat"],
                              "prop": "imageinfo", "iiprop": "url|size|mime",
                              "iiurlwidth": config.SOURCE_DOWNLOAD_MAX_EDGE, **(cont or {})})
            if data is BUSY:
                return BUSY
            out, subcats = [], []
            for p in data.get("query", {}).get("pages", {}).values():
                if p.get("ns") == 14:
                    sub = p["title"][len("Category:"):]
                    if depth < self.MAX_DEPTH and self._follow(sub, walk["root_stem"]):
                        subcats.append((sub, depth + 1))
                    continue
                info = (p.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < min_short:
                    continue  # skip before download
                url = info.get("thumburl") or info.get("url")
                if url:
                    out.append(Candidate(url, "commons.wikimedia.org", info.get("descriptionurl", url),
                                         "wikimedia_commons"))
            with self._lock:
                walk["queue"].extend(subcats)
                if "continue" in data:
                    walk["cont"] = data["continue"]
                else:
                    walk["current"] = None
            if out:
                return out
        return []


class INatSpeciesSource(INaturalistSource):
    """iNaturalist by exact scientific name, downloading the 'large' rendition
    (1024px long edge) instead of the original: labeled species photos above
    the 512px floor at a fraction of the bytes."""

    name = "inat_species"
    categories = None
    by_subcategory = False  # search by the row's query (scientific name), not its label (common name)

    def search(self, query, page):
        per_page = config.SOURCE_PAGE_SIZE["inaturalist"]
        if page * per_page > self.MAX_RESULTS:
            return []
        taxon = self._taxon_id(query)
        if taxon is BUSY:
            return BUSY
        if taxon is None:
            return []
        data = _json_or_none(paced_get(
            "https://api.inaturalist.org/v1/observations", "inat_api",
            params={
                "taxon_id": taxon, "photos": "true", "quality_grade": "research", "captive": "false",
                "photo_license": "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd",
                "per_page": per_page, "page": page, "order_by": "votes",
            },
        ))
        if data is BUSY:
            return BUSY
        min_short = config.SOURCE_MIN_SHORT_SIDE.get(self.name, config.MIN_SHORT_SIDE)
        out = []
        for obs in data.get("results", []):
            p = (obs.get("photos") or [{}])[0]
            if not p.get("url"):
                continue
            dims = p.get("original_dimensions") or {}
            w, h = dims.get("width", 0), dims.get("height", 0)
            if dims and min(w, h) * min(1.0, 1024 / max(w, h, 1)) < min_short:
                continue  # the 1024px rendition would fall below the floor
            url = p["url"].replace("/square.", "/large.")
            out.append(Candidate(url, "inaturalist.org", obs.get("uri", url), p.get("license_code")))
        return out


def build_phase15_sources() -> dict[str, BaseSource]:
    """Phase 1.5 connectors by name, in the order they are tried (see config.PHASE15_TIER_SOURCES)."""
    return {"commons_cat": CommonsCategorySource(), "wikimedia": WikimediaCommonsSource(),
            "inat_species": INatSpeciesSource(), "nasa": NasaImagesSource()}


def build_sources() -> list[BaseSource]:
    registry = {
        "openverse": OpenverseSource(),
        "wikimedia": WikimediaCommonsSource(),
        "pexels": PexelsSource(),
        "pixabay": PixabaySource(),
        "unsplash": UnsplashSource(),
        "bing": BingSource(),
        "inaturalist": INaturalistSource(),
        "nasa": NasaImagesSource(),
    }
    return [registry[name] for name in config.SOURCE_ORDER if name in registry]


def is_blocked_domain(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(b in host for b in config.BLOCKED_IMAGE_DOMAINS)
