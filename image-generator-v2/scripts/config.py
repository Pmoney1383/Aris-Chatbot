"""Single source of truth for the data-gathering pipeline's settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency) -- KEY=VALUE per line,
    '#' comments, blank lines ignored. Never overrides a var already set in
    the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

CATALOGUE_CSV = ROOT / "dataset" / "image_scraping_catalogue.csv"
QUARANTINE_DIR = ROOT / "dataset" / "quarantine"
FINAL_DIR = ROOT / "dataset" / "final"
TEST_IMAGE_DIR = ROOT / "dataset" / "test-image"
MANIFEST_DB = ROOT / "dataset" / "manifest.sqlite3"
CLIP_LABELS_DB = ROOT / "dataset" / "clip_labels.sqlite3"  # written by relabel_images.py
# vehicle_probe.py: Transportation images whose probe p(no vehicle) is at least this
# are left out of captions.csv. Set from the cross-validated precision in `train`.
# 2026-10-01, 1,260 labels: at 0.8 the CV drops were 55 N, 7 V (vehicle, no usable view),
# 1 real side view -- 87% strict N precision, ~98% "not a useful vehicle photo".
NO_VEHICLE_DROP_PROB = 0.8
# A running gather_images.py stops gracefully (saving everything) when this file appears.
STOP_COLLECTION_FILE = ROOT / "dataset" / "STOP_COLLECTION"
LOG_DIR = ROOT / "logs"

# Overall dataset target. Script keeps running (across resumes) until this
# many *accepted* (post dedup/corrupt/resolution-check) images exist, or the
# catalogue's per-row targets are exhausted.
TOTAL_TARGET_IMAGES = 500_000
STRETCH_TARGET_IMAGES = 1_000_000

# Roughly spread the target evenly across catalogue rows. Adult category rows
# are excluded from this budget (see ADULT_CATEGORY_NAME below) since that
# category is gated off from automated collection entirely.
IMAGES_PER_ROW_TARGET = 120  # 5920 SFW rows * 120 ~= 710k ceiling if every row hits target

MIN_SHORT_SIDE = 1080
# Pixabay's standard API key only exposes largeImageURL (long edge capped at
# 1280px), so a 3:2 photo arrives at 1280x853. Holding it to 1080 rejected
# nearly every Pixabay image; 800 admits 3:2 and 4:3 photos. width/height are
# stored in the manifest, so training can still filter these out later.
SOURCE_MIN_SHORT_SIDE = {"pixabay": 800, "megalith": 640,
                         # Phase 1.5 sources: 512 = the phase-2 training resolution
                         "commons_cat": 512, "inat_species": 512, "nasa": 512, "wikimedia": 512}
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024  # skip absurdly large files
MAX_OUTPUT_JPEG_BYTES = int(1.5 * 1024 * 1024)  # reject anything whose re-encoded JPEG exceeds this
OUTPUT_JPEG_QUALITY = 92

# Long edge every saved image is capped at. Requested server-side where the
# source supports it (Pexels CDN params, Wikimedia thumbnails) so we never
# download a huge original, and applied locally to everything else (Bing
# results) before encoding, so big originals get downscaled rather than
# tripping MAX_OUTPUT_JPEG_BYTES.
SOURCE_DOWNLOAD_MAX_EDGE = 1920
REQUEST_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRY_COUNT = 1  # a single retry; on 429 we cool the domain down instead of busy-retrying

# Concurrency: each worker owns one catalogue row at a time. Most time is
# network wait, and pacing below is enforced per domain across all workers,
# so more workers mostly means more hosts being downloaded from in parallel.
NUM_WORKERS = 16
DB_COMMIT_EVERY = 50

# Minimum seconds between two requests sharing a rate key. API calls use an
# explicit key (e.g. "pexels_api") so they don't share a budget with the same
# host's image downloads; downloads are keyed by hostname.
DOMAIN_MIN_INTERVAL_SECONDS = {
    "pexels_api": 18.5,      # Pexels free tier: 200 requests/hour
    "pixabay_api": 0.65,     # Pixabay: 100 requests/minute
    "wikimedia_api": 1.5,         # Wikimedia: 200 req/min total per client (API + media); 40/min here
    "openverse_api": 2.0,
    "unsplash_api": 75.0,    # Unsplash demo tier: 50 requests/hour
    "bing_search": 2.0,
    "inat_api": 1.1,         # iNaturalist asks for ~1 request/second
    "nasa_api": 0.5,
    "images.pexels.com": 0.05,
    "cdn.pixabay.com": 0.05,
    "pixabay.com": 0.1,
    "upload.wikimedia.org": 0.4,  # + 150/min downloads = 190/min, under the 200/min limit
    "inaturalist-open-data.s3.amazonaws.com": 0.3,
    "static.inaturalist.org": 0.5,
    "images-assets.nasa.gov": 0.2,
    "pd12m.s3.us-west-2.amazonaws.com": 0.02,
    # Flickr's CDN blocked this IP (403) after ~517k Megalith downloads at 0.05s/host;
    # 0.5s/host (~20 downloads/s over all hosts) is the gentler pace.
    **{f"farm{i}.staticflickr.com": 0.5 for i in (*range(1, 10), 66)},
    "live.staticflickr.com": 0.5,
}
DEFAULT_MIN_INTERVAL_SECONDS = 0.5

# Rolling download-volume caps per host, in bytes. iNaturalist's guidance is
# to stay under 5 GB/hour and 24 GB/day of media; these leave a margin.
DOWNLOAD_BYTE_BUDGETS = {
    "inaturalist-open-data.s3.amazonaws.com": {"hour": 4.0e9, "day": 20.0e9},
    "static.inaturalist.org": {"hour": 4.0e9, "day": 20.0e9},
}
# A worker won't sit waiting for a busy search API longer than this; it tries
# another source instead, so a slow API (Pexels, 1 call per 18.5s) doesn't
# stall every worker queued behind it.
MAX_SEARCH_WAIT_SECONDS = 3.0

# Search results per API call -- each source's documented maximum, so every
# rate-limited call returns as many candidates as possible.
SOURCE_PAGE_SIZE = {
    "pexels": 80,
    "pixabay": 200,
    "wikimedia": 50,
    "openverse": 20,
    "unsplash": 30,
    "bing": 100,
    "inaturalist": 200,
    "nasa": 100,
    "commons_cat": 50,   # the API scales thumbnails for at most 50 files per request
}
# Stop querying a source for a row after this many consecutive pages that
# produced no accepted image.
MAX_STAGNANT_PAGES = 2
# Hard cap on pages per source per row, so a row can't spin forever.
MAX_PAGES_PER_SOURCE = 10
# Per-source overrides: Commons pages are small (50 files), so a concept needs more of them.
SOURCE_MAX_PAGES = {"commons_cat": 60}

# Stock-photo / preview hosts whose images are watermarked by design. Web
# search results from these are skipped before download.
BLOCKED_IMAGE_DOMAINS = (
    "shutterstock.com", "gettyimages.", "istockphoto.com", "alamy.com",
    "dreamstime.com", "123rf.com", "depositphotos.com", "stock.adobe.com",
    "ftcdn.net", "vecteezy.com", "canstockphoto.", "bigstockphoto.com",
    "stockphoto.com", "agefotostock.com", "superstock.com", "pond5.com",
    "freepik.com", "vectorstock.com", "colourbox.com", "photodune.net",
    "pinimg.com", "pinterest.",
    # e-commerce product listings (off-topic product shots, text overlays)
    "media-amazon.com", "ssl-images-amazon.com", "ebayimg.com", "walmartimages.com",
    "etsystatic.com", "alicdn.com", "cdn.shopify.com", "bigcommerce.com",
)

# Watermark rejection (LAION-5B watermark detector, EfficientNet-B3).
WATERMARK_MODEL_PATH = ROOT / "models" / "watermark_model_v1.pt"
WATERMARK_ENABLED = True
# Probability above which an image counts as watermarked. LAION used 0.8 for
# its dataset; 0.5 is stricter -- it drops some clean images too, in exchange
# for letting fewer watermarked ones through.
WATERMARK_THRESHOLD = 0.5

# Category name(s) that must NEVER be routed through generic web/image-search
# scraping. The gathering script hard-skips these unless an explicit, named,
# licensed/age-verified connector is wired in sources.py and enabled here.
ADULT_CATEGORY_NAMES = {"Adult 18+ Safety Research"}
ADULT_COLLECTION_ENABLED = False  # do not flip without wiring a real verified-adult connector

# API keys (all optional — sources without a key are simply skipped).
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # openai_food_batch.py (food & drink captions)

# Order sources are tried per row. Subject-specific sources (iNaturalist,
# NASA -- each limited to its own categories, see sources.py) go first because
# their results match the subject most precisely; then the keyed stock APIs,
# Wikimedia, and finally Bing if enabled. Openverse is off: its anonymous
# limits Cloudflare-blocked us during testing. The Met's API is not used: its
# search matches any metadata loosely and has no working title filter, and
# PD12M already includes the Met's public-domain collection with captions.
SOURCE_ORDER = ["inaturalist", "nasa", "pexels", "pixabay", "wikimedia", "unsplash", "bing"]

# PD12M bulk import (Spawning's 12.4M public-domain/CC0 image-caption set,
# https://huggingface.co/datasets/Spawning/PD12M). Runs alongside the search
# workers; image sizes and captions come in the metadata, so undersized and
# unsafe-captioned images are skipped before download.
PD12M_ENABLED = True
PD12M_WORKERS = 8
PD12M_HF_REPO = "Spawning/PD12M"
# Raised from 400k / 1.5k / 150k for the Phase 1.5 expansion (more unique images ->
# fewer epochs over the same pictures); PD12M_EXCLUDE_PATTERN keeps the new ones photographic.
PD12M_MAX_IMAGES = 1_000_000          # total cap
PD12M_MAX_PER_SUBCATEGORY = 4_000     # per caption-matched catalogue subcategory
PD12M_UNMATCHED_MAX = 400_000         # captions matching no subcategory go to "General (PD12M)"
# Captions matching this are skipped before download: museum objects, artwork, documents,
# maps (PD12M has many). Applies to shards read from now on; already-imported images stay.
PD12M_EXCLUDE_PATTERN = (
    r"\b(painting|drawing|illustration|engraving|etching|lithograph|woodcut|print|manuscript|document|"
    r"page|text|letter|handwritten|handwriting|map|poster|stamp|coin|medal|banknote|herbarium|specimen|"
    r"fossil|sculpture|statue|figurine|artifact|artefact|museum|plate|portrait of|bust|carving|textile|"
    r"fabric sample|vase|ceramic|pottery|jar|bowl with|object|tool|weapon|sword|jewelry|brooch|ring)\b"
)
PD12M_MAX_LONG_EDGE = 6000            # skip giant originals (bandwidth); plenty remain
PD12M_UNMATCHED_CATEGORY = "General (PD12M)"

# Megalith-10m gap filler (megalith.py): CC0 / public-domain Flickr photos with
# InternVL2 captions, used only for catalogue rows still under target. Its
# download URLs cap the long edge at 1024px, hence the 640 short-side floor in
# SOURCE_MIN_SHORT_SIDE (still above the 512px training resolution).
MEGALITH_ENABLED = True
MEGALITH_WORKERS = 24
MEGALITH_HF_REPO = "CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption"
MEGALITH_CAPTION_COLUMN = "caption_internlm2_short"
# General Megalith expansion (gather_images.py --expand): photos not tied to catalogue rows,
# filed by caption-matched subcategory like PD12M, under these caps.
MEGALITH_GENERAL_MAX_IMAGES = 400_000
MEGALITH_GENERAL_MAX_PER_SUBCATEGORY = 4_000
MEGALITH_GENERAL_UNMATCHED_MAX = 150_000
MEGALITH_GENERAL_UNMATCHED_CATEGORY = "General (Megalith)"
MEGALITH_GENERAL_DATASET = "megalith_gen"   # shard-progress key
# A subcategory must be named within this many leading caption characters.
# Later mentions are usually background ("...standing before a field of
# wildflowers"), which filed photos under the wrong subcategory.
MEGALITH_MATCH_MAX_CHARS = 100

# Captions (PD12M) matching this are skipped before download, keeping nudity
# and sexual content out of the SFW collection.
UNSAFE_CAPTION_PATTERN = (
    r"\b(nude|nudes|naked|nudity|topless|bare[- ]breasted|erotic|sexual|sexy|sex|"
    r"lingerie|underwear|genital\w*|breasts?|buttocks|undress\w*|intimate|seductive)\b"
)

# Bing web image search. Never used for the adult category (rows in
# ADULT_CATEGORY_NAMES are skipped before any source is queried), and always
# requested with SafeSearch strict. Off by default: in testing, most results
# for scripted requests were off-topic (wristwatches for "historic street",
# game boxes for "chef portrait"), and SafeSearch strict still let through an
# unrelated suggestive result. Needs a relevance filter before it's worth
# using. Enable with --bing.
ENABLE_BING = False
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

# Near-duplicate rejection threshold (Hamming distance between 64-bit pHashes).
PHASH_MAX_DISTANCE = 6

USER_AGENT = "ArisImageGatherer/1.0 (research dataset collection; https://github.com/Pmoney1383/Aris-Chatbot)"


# -- Phase 1.5 (scripts/phase15_concepts.py, gather_images.py --phase15) -----
PHASE15_ID_PREFIX = "P15-"
PHASE15_WORKERS = 10
PHASE15_SEARCH_WAIT_SECONDS = 20.0   # workers share the 1 req/s Commons API; let them queue instead of giving up
PHASE15_GPU_MEMORY_FRACTION = 0.12   # caps the watermark detector at ~2 GB so it can run next to DiT training
PHASE15_MEGALITH_DATASET = "megalith_p15"  # separate shard progress: re-scans every shard for the scene patterns
COMMONS_MAX_CALLS_PER_SEARCH = 8
# Commons subcategories whose names contain any of these are not followed
# (packaging, venues, artwork, toys, parts...): they drift away from the concept.
COMMONS_SKIP_SUBCATEGORY_WORDS = (
    "box", "packag", "oven", "restaurant", "shop", "store", "stall", "vendor", "market", "logo", "sign",
    "advert", "poster", "label", "menu", "stamp", "coin", "museum", "drawing", "painting", "illustration",
    "in art", "art ", "sculpture", "statue", "diagram", "map", "toy", "scale model", "lego", "model car",
    "interior", "dashboard", "engine", "wheel", "badge", "emblem", "brochure", "catalog", "manual", "patent",
    "ingredient", "recipe", "cookbook", "machine", "factory", "production", "people eating", "eating",
    "festival", "competition", "video", "screenshot", "wreck", "crash", "accident", "damaged",
)
# concept source -> connectors, tried in order. Commons concepts fall back to Commons full-text
# search when their category tree is small (e.g. many Persian dishes have only a few categorized files).
PHASE15_TIER_SOURCES = {"commons": ("commons_cat", "wikimedia"), "wsearch": ("wikimedia",),
                        "inat": ("inat_species",), "nasa": ("nasa",)}
PHASE15_SCENE_MATCH_CHARS = 140   # scene regexes must match within the start of a Megalith caption
