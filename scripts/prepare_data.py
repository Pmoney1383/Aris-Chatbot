"""Download and tokenize the Stage B diversity mix into uint16 binary shards.

Three streaming sources, each written to its own source-tagged shards in
data/shards/ (mixing happens at train time via weighted shard sampling):

- fineweb (80%): HuggingFaceFW/fineweb-edu sample-10BT. Fast-forwards past the
  first `fineweb_skip_tokens` tokens (already consumed by Stage 0) so Stage B
  trains on fresh documents.
- books (10%): emozilla/pg19 (parquet mirror of deepmind/pg19 — Project
  Gutenberg long-form books).
- conv (10%): allenai/soda (everyday dialogue), formatted "Speaker: utterance"
  one turn per line.

Everything is tokenized with tiktoken gpt2 (EOT between documents) and written
as flat uint16 shards named shard_{source}_{index:04d}.bin.

Resumable: existing full-size shards are skipped per source, and streaming
skips the documents they contained.
"""

import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tiktoken
from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from src.config import TrainConfig

console = Console()


@dataclass
class Source:
    name: str                              # shard filename tag
    dataset_name: str
    dataset_config: str | None
    token_budget: int
    extract_text: Callable[[dict], str]
    skip_tokens: int = 0                   # fast-forward (fresh data after Stage 0)


def soda_text(doc: dict) -> str:
    return "\n".join(
        f"{speaker}: {utterance}"
        for speaker, utterance in zip(doc["speakers"], doc["dialogue"])
    )


def build_sources(cfg: TrainConfig) -> list[Source]:
    return [
        Source(
            name="fineweb",
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_split,
            token_budget=cfg.fineweb_token_budget,
            extract_text=lambda doc: doc["text"],
            skip_tokens=cfg.fineweb_skip_tokens,
        ),
        Source(
            name="books",
            dataset_name=cfg.books_dataset_name,
            dataset_config=None,
            token_budget=cfg.books_token_budget,
            extract_text=lambda doc: doc["text"],
        ),
        Source(
            name="conv",
            dataset_name=cfg.conv_dataset_name,
            dataset_config=None,
            token_budget=cfg.conv_token_budget,
            extract_text=soda_text,
        ),
    ]


def existing_complete_shards(shard_dir: Path, source: str, shard_bytes: int) -> int:
    """Count leading complete shards for a source; delete a trailing partial."""
    idx = 0
    while True:
        p = shard_dir / f"shard_{source}_{idx:04d}.bin"
        if p.exists() and p.stat().st_size == shard_bytes:
            idx += 1
        else:
            if p.exists():
                p.unlink()  # partial shard from an interrupted run — redo it
            return idx


def token_stream(src: Source, enc) -> Iterator[list[int]]:
    """Yield tokenized documents (EOT-terminated) from a streaming dataset."""
    ds = load_dataset(
        src.dataset_name, name=src.dataset_config, split="train", streaming=True
    )
    eot = enc.eot_token
    for doc in ds:
        text = src.extract_text(doc)
        if not text:
            continue
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        yield tokens


def prepare_source(src: Source, cfg: TrainConfig, enc, progress: Progress) -> int:
    shard_bytes = cfg.shard_size * 2  # uint16

    shard_idx = existing_complete_shards(cfg.shard_dir, src.name, shard_bytes)
    tokens_written = shard_idx * cfg.shard_size

    task = progress.add_task(
        src.name, total=src.token_budget, completed=tokens_written, shards=shard_idx
    )
    if tokens_written >= src.token_budget:
        progress.console.print(
            f"[green]{src.name}[/]: budget already met "
            f"({tokens_written:,} tokens in {shard_idx} shards)."
        )
        return tokens_written
    if shard_idx > 0:
        progress.console.print(
            f"[yellow]{src.name}[/]: resuming past {shard_idx} complete shards."
        )

    buffer = np.empty(cfg.shard_size, dtype=np.uint16)
    buf_pos = 0
    # fast-forward: Stage 0's consumed tokens, plus tokens already in shards
    tokens_to_skip = src.skip_tokens + tokens_written

    for tokens in token_stream(src, enc):
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
                out = cfg.shard_dir / f"shard_{src.name}_{shard_idx:04d}.bin"
                buffer.tofile(out)
                shard_idx += 1
                tokens_written += cfg.shard_size
                buf_pos = 0
                progress.update(task, completed=tokens_written, shards=shard_idx)
                if tokens_written >= src.token_budget:
                    return tokens_written

        progress.update(task, completed=tokens_written + buf_pos, shards=shard_idx)

    # stream exhausted before budget: flush the final partial shard
    if buf_pos > 0:
        out = cfg.shard_dir / f"shard_{src.name}_{shard_idx:04d}.bin"
        buffer[:buf_pos].tofile(out)
        tokens_written += buf_pos
        shard_idx += 1
        progress.update(task, completed=tokens_written, shards=shard_idx)
    progress.console.print(
        f"[yellow]{src.name}[/]: stream exhausted at {tokens_written:,} tokens "
        f"(budget was {src.token_budget:,})."
    )
    return tokens_written


def main():
    cfg = TrainConfig()
    cfg.shard_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    sources = build_sources(cfg)
    start_time = time.time()
    totals = {}

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed:,}/{task.total:,} tokens"),
        TextColumn("shards: {task.fields[shards]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for src in sources:
            if src.skip_tokens > 0:
                progress.console.print(
                    f"[bold]{src.name}[/]: skipping first "
                    f"{src.skip_tokens:,} tokens (used in Stage 0)..."
                )
            totals[src.name] = prepare_source(src, cfg, enc, progress)

    elapsed = time.time() - start_time
    total_tokens = sum(totals.values())
    total_bytes = sum(p.stat().st_size for p in cfg.shard_dir.glob("shard_*.bin"))
    console.print("\n=== prepare_data complete ===")
    for name, n in totals.items():
        console.print(f"{name:>8}: {n:,} tokens")
    console.print(f"Total tokens written: {total_tokens:,}")
    console.print(f"Disk size:            {total_bytes / 1e9:.2f} GB")
    console.print(f"Time this run:        {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
