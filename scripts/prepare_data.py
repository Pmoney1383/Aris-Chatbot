"""Download and tokenize FineWeb-Edu into uint16 binary shards.

Streams HuggingFaceFW/fineweb-edu (sample-10BT), tokenizes with tiktoken gpt2,
and writes flat uint16 shards to data/shards/ until token_budget is reached.

Resumable: existing full-size shards are skipped, and streaming skips the
documents they contained.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tiktoken
from datasets import load_dataset
from rich.progress import (
    BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from src.config import TrainConfig


def main():
    cfg = TrainConfig()
    cfg.shard_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # document separator

    shard_bytes = cfg.shard_size * 2  # uint16

    # ---- resume: count existing complete shards ----
    shard_idx = 0
    tokens_written = 0
    while True:
        p = cfg.shard_dir / f"shard_{shard_idx:04d}.bin"
        if p.exists() and p.stat().st_size == shard_bytes:
            shard_idx += 1
            tokens_written += cfg.shard_size
        else:
            if p.exists():
                p.unlink()  # partial shard from an interrupted run — redo it
            break

    if tokens_written >= cfg.token_budget:
        print(f"Token budget already met: {tokens_written:,} tokens in "
              f"{shard_idx} shards. Nothing to do.")
        return

    if shard_idx > 0:
        print(f"Resuming: found {shard_idx} complete shards "
              f"({tokens_written:,} tokens). Skipping ahead in the stream...")

    ds = load_dataset(cfg.dataset_name, name=cfg.dataset_split,
                      split="train", streaming=True)

    buffer = np.empty(cfg.shard_size, dtype=np.uint16)
    buf_pos = 0
    tokens_seen_this_run = 0
    tokens_to_skip = tokens_written  # fast-skip docs already in complete shards
    start_time = time.time()

    with Progress(
        TextColumn("[bold]prepare_data"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed:,}/{task.total:,} tokens"),
        TextColumn("shards: {task.fields[shards]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("", total=cfg.token_budget,
                                 completed=tokens_written, shards=shard_idx)

        for doc in ds:
            tokens = enc.encode_ordinary(doc["text"])
            tokens.append(eot)

            if tokens_to_skip > 0:
                tokens_to_skip -= len(tokens)
                continue

            tokens = np.asarray(tokens, dtype=np.uint16)
            pos = 0
            while pos < len(tokens):
                take = min(len(tokens) - pos, cfg.shard_size - buf_pos)
                buffer[buf_pos : buf_pos + take] = tokens[pos : pos + take]
                buf_pos += take
                pos += take

                if buf_pos == cfg.shard_size:
                    out = cfg.shard_dir / f"shard_{shard_idx:04d}.bin"
                    buffer.tofile(out)
                    shard_idx += 1
                    tokens_written += cfg.shard_size
                    buf_pos = 0
                    progress.update(task, completed=tokens_written,
                                    shards=shard_idx)

                    if tokens_written >= cfg.token_budget:
                        break

            tokens_seen_this_run += len(tokens)
            if tokens_written < cfg.token_budget:
                progress.update(task, completed=tokens_written + buf_pos,
                                shards=shard_idx)
            if tokens_written >= cfg.token_budget:
                break

    # write any final partial shard only if we ran out of data before budget
    if tokens_written < cfg.token_budget and buf_pos > 0:
        out = cfg.shard_dir / f"shard_{shard_idx:04d}.bin"
        buffer[:buf_pos].tofile(out)
        tokens_written += buf_pos
        shard_idx += 1

    elapsed = time.time() - start_time
    total_bytes = sum(p.stat().st_size for p in cfg.shard_dir.glob("shard_*.bin"))
    print("\n=== prepare_data complete ===")
    print(f"Total tokens written: {tokens_written:,}")
    print(f"Shards:               {shard_idx}")
    print(f"Disk size:            {total_bytes / 1e9:.2f} GB")
    print(f"Time this run:        {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
