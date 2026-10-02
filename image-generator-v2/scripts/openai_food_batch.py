"""
Caption the Phase 1.5 food & drink images with the OpenAI Batch API.

Florence misnames fine-grained dishes, so these images are captioned by an API
model that is told which dish it is (caption_images.py export-food-batch writes
dataset/food_caption_requests.jsonl). This script turns that file into Batch API
jobs, runs them, and writes results in the format caption_images.py imports.

Requires OPENAI_API_KEY in image-generator-v2/.env (gitignored).

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe openai_food_batch.py run --limit 100   # pilot: 100 images, prints samples + cost
    ../.venv/Scripts/python.exe openai_food_batch.py run               # everything not captioned yet
    ../.venv/Scripts/python.exe openai_food_batch.py status            # batches submitted so far
    ../.venv/Scripts/python.exe openai_food_batch.py import            # results -> manifest (caption_images import)
Afterwards, to train on them: caption_images.py export-csv, then preprocess.py text / latents / flips
(each only encodes the new rows; set PREP_GENERAL_ONLY=1).

`run` keeps --parallel batch files in flight (default 4, ~2k requests each, well under
a Tier 1 enqueued-token limit) and collects each as it finishes. It is resumable
(Ctrl+C, rerun): images that already have a result, or are in a batch still running,
are skipped, and running batches are picked up again.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

import config

API = "https://api.openai.com/v1"
REQUESTS = config.ROOT / "dataset" / "food_caption_requests.jsonl"
DIR = config.ROOT / "dataset" / "openai_batch"
STATE = DIR / "state.json"
RESULTS = DIR / "results.jsonl"      # {"sha256", "caption"} -- caption_images.py import format
ERRORS = DIR / "errors.jsonl"
ENDPOINT = "/v1/chat/completions"
MAX_SIDE = 512                       # resize before upload: predictable image tokens, small files
JPEG_QUALITY = 85
MAX_FILE_BYTES = 40 * 2 ** 20        # a 150 MB single-request upload was reset mid-transfer; keep files small
MAX_FILE_REQUESTS = 5000
POLL_SECONDS = 60


def api_key() -> str:
    key = getattr(config, "OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set: add a line OPENAI_API_KEY=... to image-generator-v2/.env")
    return key


def headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}"}


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"batches": []}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1))


def done_shas() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["sha256"] for l in open(RESULTS, encoding="utf-8") if l.strip()}


def encode_image(path: str) -> str | None:
    try:
        img = Image.open(config.ROOT / path)
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# About 10% of the pilot's images showed no dish at all (a motorboat filed under "Mochi", a
# manuscript under "Spaghetti bolognese"); caption_images.py import-food-batch skips these.
NOT_VISIBLE_RULE = (" If the photo does not actually show this dish or drink, reply with only the word "
                    "NOT_VISIBLE and nothing else.")


def request_line(item: dict, b64: str, args) -> str:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": item["prompt"] + NOT_VISIBLE_RULE},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": args.detail}},
        ]}],
        "max_completion_tokens": args.max_tokens,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    return json.dumps({"custom_id": item["sha256"], "method": "POST", "url": ENDPOINT, "body": body})


def prepare(args) -> list:
    """Write batch input files for every request without a result or a running batch."""
    if not REQUESTS.exists():
        raise SystemExit(f"{REQUESTS} not found: run caption_images.py export-food-batch first")
    DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    # Skip images with a caption, and images in a batch that hasn't been collected yet.
    # Requests that failed inside a finished batch are retried on the next run.
    in_flight = {sha for b in state["batches"] if not b.get("collected")
                 and b["status"] not in ("failed", "expired", "cancelled") for sha in b.get("shas", [])}
    skip = done_shas() | in_flight
    items = [json.loads(l) for l in open(REQUESTS, encoding="utf-8") if l.strip()]
    items = [it for it in items if it["sha256"] not in skip][: args.limit]
    print(f"prepare: {len(items):,} requests to send ({len(skip):,} done or in flight)")
    files, n = [], len(list(DIR.glob("input_*.jsonl")))
    out, size, shas = None, 0, []

    def close():
        nonlocal out
        if out:
            out.close()
            files.append((path, shas))
            out = None

    with ThreadPoolExecutor(8) as pool:
        for it, b64 in zip(items, pool.map(lambda it: encode_image(it["file_path"]), items)):
            if b64 is None:
                print(f"  unreadable image, skipped: {it['file_path']}")
                continue
            line = request_line(it, b64, args) + "\n"
            if out is None or size + len(line) > MAX_FILE_BYTES or len(shas) >= args.chunk:
                close()
                path = DIR / f"input_{n:03d}.jsonl"
                n += 1
                out, size, shas = open(path, "w", encoding="utf-8"), 0, []
            out.write(line)
            size += len(line)
            shas.append(it["sha256"])
    close()
    for p, s in files:
        print(f"  {p.name}: {len(s):,} requests, {p.stat().st_size / 2 ** 20:,.0f} MB")
    return files


def upload(path) -> str:
    for attempt in range(5):
        try:
            with open(path, "rb") as f:
                r = requests.post(f"{API}/files", headers=headers(), data={"purpose": "batch"},
                                  files={"file": (path.name, f, "application/jsonl")}, timeout=600)
            r.raise_for_status()
            return r.json()["id"]
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"  upload of {path.name} failed ({e.__class__.__name__}); retrying in {30 * (attempt + 1)} s")
            time.sleep(30 * (attempt + 1))


def submit(path, shas, state) -> dict:
    file_id = upload(path)
    r = requests.post(f"{API}/batches", headers=headers(), timeout=60,
                      json={"input_file_id": file_id, "endpoint": ENDPOINT, "completion_window": "24h"})
    r.raise_for_status()
    b = {"input": path.name, "file_id": file_id, "batch_id": r.json()["id"], "status": r.json()["status"],
         "shas": shas, "submitted": time.time()}
    state["batches"].append(b)
    save_state(state)
    print(f"submitted {path.name}: batch {b['batch_id']} ({len(shas):,} requests)")
    return b


def refresh(b, state) -> dict:
    r = requests.get(f"{API}/batches/{b['batch_id']}", headers=headers(), timeout=60)
    r.raise_for_status()
    info = r.json()
    b["status"] = info["status"]
    b["output_file_id"], b["error_file_id"] = info.get("output_file_id"), info.get("error_file_id")
    b["counts"] = info.get("request_counts")
    if info.get("errors"):
        b["errors"] = info["errors"]
    save_state(state)
    return b


def collect(b, state) -> tuple[int, int, int, int]:
    """Append a finished batch's captions to results.jsonl. Returns (ok, failed, in_tokens, out_tokens)."""
    ok = bad = tin = tout = 0
    if b.get("output_file_id"):
        text = requests.get(f"{API}/files/{b['output_file_id']}/content", headers=headers(), timeout=600).text
        with open(RESULTS, "a", encoding="utf-8") as res, open(ERRORS, "a", encoding="utf-8") as err:
            for line in text.splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                resp = d.get("response") or {}
                body = resp.get("body") or {}
                usage = body.get("usage") or {}
                tin += usage.get("prompt_tokens", 0)
                tout += usage.get("completion_tokens", 0)
                try:
                    caption = body["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, TypeError, AttributeError):
                    caption = ""
                if resp.get("status_code") == 200 and caption:
                    res.write(json.dumps({"sha256": d["custom_id"], "caption": caption}) + "\n")
                    ok += 1
                else:
                    err.write(line + "\n")
                    bad += 1
    if b.get("error_file_id"):
        text = requests.get(f"{API}/files/{b['error_file_id']}/content", headers=headers(), timeout=600).text
        with open(ERRORS, "a", encoding="utf-8") as err:
            for line in text.splitlines():
                if line.strip():
                    err.write(line + "\n")
                    bad += 1
    b["collected"] = True
    save_state(state)
    return ok, bad, tin, tout


FINAL = ("completed", "failed", "expired", "cancelled")


def finish(b, state, args) -> None:
    if b["status"] != "completed" and not b.get("output_file_id"):
        print(f"  batch {b['batch_id']} ended as {b['status']}: {b.get('errors')}")
        return
    ok, bad, tin, tout = collect(b, state)
    cost = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
    print(f"  collected {ok:,} captions, {bad:,} errors | tokens in {tin:,} out {tout:,} | "
          f"~${cost:,.2f} at ${args.price_in}/${args.price_out} per 1M")


def run(args) -> None:
    """Keep up to --parallel batches in flight (each ~2k requests, under the queue limit);
    collect each as it finishes. Batches from an earlier run are resumed first."""
    state = load_state()
    active = [b for b in state["batches"] if not b.get("collected") and b["status"] not in FINAL[1:]]
    for b in active:
        print(f"resuming {b['batch_id']} ({b['input']})")
    todo = prepare(args)
    while todo or active:
        while todo and len(active) < args.parallel:
            active.append(submit(*todo.pop(0), state))
        time.sleep(POLL_SECONDS)
        for b in list(active):
            refresh(b, state)
            c = b.get("counts") or {}
            print(f"  {b['input']}: {b['status']} {c.get('completed', 0):,}/{c.get('total', 0):,} "
                  f"(failed {c.get('failed', 0):,})", flush=True)
            if b["status"] in FINAL:
                finish(b, state, args)
                active.remove(b)
    if RESULTS.exists():
        lines = open(RESULTS, encoding="utf-8").read().splitlines()
        print(f"\n{len(lines):,} captions in {RESULTS.name}. Last few:")
        for l in lines[-5:]:
            d = json.loads(l)
            print(f"  {d['sha256'][:12]}: {d['caption'][:300]!r}")
        print("\nNext: openai_food_batch.py import  (then export-csv + preprocess to train on them)")


def status(_args) -> None:
    state = load_state()
    for b in state["batches"]:
        if b["status"] not in ("completed", "failed", "expired", "cancelled"):
            refresh(b, state)
        print(f"{b['input']}: {b['batch_id']} {b['status']} {b.get('counts')} collected={b.get('collected', False)}")
    print(f"{len(done_shas()):,} captions in results.jsonl")


def do_import(_args) -> None:
    import caption_images
    import manifest_db
    db = manifest_db.ManifestDB()
    try:
        caption_images.import_food_batch(db, str(RESULTS))
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["run", "status", "import"])
    p.add_argument("--limit", type=int, default=None, help="only this many images (pilot)")
    p.add_argument("--chunk", type=int, default=MAX_FILE_REQUESTS, help="max requests per batch file")
    p.add_argument("--parallel", type=int, default=4, help="batches in flight at once (mind the queue limit)")
    p.add_argument("--model", default="gpt-6-luna")
    # Images are resized to 512 px first, so "high" cost the same as "low" in the pilot
    # (14,799 input tokens for 50 images either way) and its captions were a little more specific.
    p.add_argument("--detail", default="high", choices=["low", "high", "auto"])
    p.add_argument("--reasoning-effort", default="none", help='"" to leave it out of the request')
    p.add_argument("--max-tokens", type=int, default=200)
    # Batch prices per 1M tokens the user quoted on 2026-10-01 -- only used for the cost printout.
    p.add_argument("--price-in", type=float, default=0.05)
    p.add_argument("--price-out", type=float, default=0.25)
    args = p.parse_args()
    {"run": run, "status": status, "import": do_import}[args.stage](args)


if __name__ == "__main__":
    main()
