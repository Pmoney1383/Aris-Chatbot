"""Download and tokenize SFT conversation data into paired token/mask shards.

Sources:
- OpenAssistant/oasst1 + oasst2: human-written conversation trees. Each tree is
  linearized by walking from the root and picking the best-ranked reply at
  every level (rank 0 = highest human rating), English only, no deleted
  messages.
- allenai/WildChat-1M (gated — needs `huggingface-cli login`): English,
  multi-turn (>= 2 assistant turns), non-toxic, non-redacted conversations,
  streamed up to `wildchat_token_budget` tokens.

Every conversation is rendered with the chat template in src/chat_format.py
and tokenized with tiktoken gpt2. Output in data/sft_shards/:

    shard_sft_0000.bin       flat uint16 token ids
    shard_sft_mask_0000.bin  uint8 loss mask, same length (1 = assistant token)

Conversations are packed back-to-back into shards (a window at train time may
span two conversations; the mask keeps the loss correct).
"""

import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.chat_format import encode_conversation
from src.config import SFTConfig

console = Console()

Turns = list[tuple[str, str]]  # [(role, text), ...]


# =========================================================
# SOURCES
# =========================================================

def oasst_conversations(dataset_name: str) -> Iterator[Turns]:
    """Linearize OASST message trees: root -> best-ranked child -> ... -> leaf."""
    ds = load_dataset(dataset_name, split="train")

    children: dict[str, list[dict]] = {}
    roots: list[dict] = []
    for msg in ds:
        if msg.get("deleted"):
            continue
        if msg.get("lang") != "en":
            continue
        parent = msg.get("parent_id")
        if parent is None:
            roots.append(msg)
        else:
            children.setdefault(parent, []).append(msg)

    role_map = {"prompter": "user", "assistant": "assistant"}

    for root in roots:
        turns: Turns = []
        msg = root
        while msg is not None:
            role = role_map.get(msg["role"])
            if role is None:
                break
            turns.append((role, msg["text"]))
            replies = children.get(msg["message_id"], [])
            # rank 0 is the top human-ranked reply; unranked sorts last
            replies.sort(key=lambda m: m["rank"] if m.get("rank") is not None else 1e9)
            msg = replies[0] if replies else None

        # must end on an assistant turn and contain at least one exchange
        while turns and turns[-1][0] != "assistant":
            turns.pop()
        if len(turns) >= 2 and turns[0][0] == "user":
            yield turns


def wildchat_conversations(dataset_name: str) -> Iterator[Turns]:
    """Stream WildChat-1M: English, >=2 assistant turns, non-toxic."""
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for row in ds:
        if row.get("language") != "English":
            continue
        if row.get("turn", 0) < 2:  # `turn` = number of user/assistant exchanges
            continue
        if row.get("toxic") or row.get("redacted"):
            continue
        turns: Turns = [
            ("user" if m["role"] == "user" else "assistant", m["content"] or "")
            for m in row["conversation"]
            if m["role"] in ("user", "assistant")
        ]
        if any(m.get("toxic") for m in row["conversation"]):
            continue
        while turns and turns[-1][0] != "assistant":
            turns.pop()
        if len(turns) >= 4 and turns[0][0] == "user":  # 4 turns ~= 2 exchanges
            yield turns


# =========================================================
# SHARD WRITER
# =========================================================

class ShardWriter:
    """Packs (token, mask) streams into paired fixed-size shard files."""

    def __init__(self, shard_dir: Path, shard_size: int):
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.tok_buf = np.empty(shard_size, dtype=np.uint16)
        self.mask_buf = np.empty(shard_size, dtype=np.uint8)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0

    def _flush(self, n: int):
        tok_path = self.shard_dir / f"shard_sft_{self.shard_idx:04d}.bin"
        mask_path = self.shard_dir / f"shard_sft_mask_{self.shard_idx:04d}.bin"
        self.tok_buf[:n].tofile(tok_path)
        self.mask_buf[:n].tofile(mask_path)
        self.shard_idx += 1

    def add(self, ids: list[int], mask: list[int]):
        ids = np.asarray(ids, dtype=np.uint16)
        mask = np.asarray(mask, dtype=np.uint8)
        self.total_tokens += len(ids)
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, self.shard_size - self.pos)
            self.tok_buf[self.pos : self.pos + take] = ids[pos : pos + take]
            self.mask_buf[self.pos : self.pos + take] = mask[pos : pos + take]
            self.pos += take
            pos += take
            if self.pos == self.shard_size:
                self._flush(self.shard_size)
                self.pos = 0

    def close(self):
        if self.pos > 0:
            self._flush(self.pos)
            self.pos = 0


# =========================================================
# MAIN
# =========================================================

def main():
    cfg = SFTConfig()
    cfg.shard_dir.mkdir(parents=True, exist_ok=True)

    # a fresh run replaces any previous shards (small enough to just redo)
    for old in cfg.shard_dir.glob("shard_sft_*.bin"):
        old.unlink()

    writer = ShardWriter(cfg.shard_dir, cfg.shard_size)
    stats: dict[str, dict] = {}

    sources = [
        ("oasst1", lambda: oasst_conversations(cfg.oasst1_dataset), None),
        ("oasst2", lambda: oasst_conversations(cfg.oasst2_dataset), None),
        ("wildchat", lambda: wildchat_conversations(cfg.wildchat_dataset),
         cfg.wildchat_token_budget),
    ]

    for name, make_iter, token_budget in sources:
        console.print(f"\n[bold]{name}[/]: loading...")
        n_convs = 0
        n_tokens = 0
        n_skipped_long = 0
        try:
            for turns in make_iter():
                ids, mask = encode_conversation(turns)
                if len(ids) > cfg.max_conversation_tokens:
                    n_skipped_long += 1
                    continue
                writer.add(ids, mask)
                n_convs += 1
                n_tokens += len(ids)
                if n_convs % 5000 == 0:
                    console.print(
                        f"  {name}: {n_convs:,} conversations, {n_tokens / 1e6:.1f}M tokens"
                    )
                if token_budget is not None and n_tokens >= token_budget:
                    console.print(f"  {name}: token budget reached.")
                    break
        except Exception as e:
            console.print(f"[red]{name} failed: {type(e).__name__}: {e}[/]")
            if name == "wildchat":
                console.print(
                    "[yellow]WildChat-1M is a gated dataset — run "
                    "`huggingface-cli login` after accepting the license on "
                    "https://huggingface.co/datasets/allenai/WildChat-1M[/]"
                )
        stats[name] = {
            "conversations": n_convs,
            "tokens": n_tokens,
            "skipped_long": n_skipped_long,
        }
        console.print(
            f"[green]{name}[/]: {n_convs:,} conversations, {n_tokens:,} tokens "
            f"({n_skipped_long:,} skipped as >{cfg.max_conversation_tokens} tokens)"
        )

    writer.close()

    total_convs = sum(s["conversations"] for s in stats.values())
    total_tokens = sum(s["tokens"] for s in stats.values())

    table = Table(title="prepare_sft_data summary")
    table.add_column("source")
    table.add_column("conversations", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("avg tokens/conv", justify="right")
    for name, s in stats.items():
        avg = s["tokens"] / s["conversations"] if s["conversations"] else 0
        table.add_row(name, f"{s['conversations']:,}", f"{s['tokens']:,}", f"{avg:.0f}")
    avg_all = total_tokens / total_convs if total_convs else 0
    table.add_row("total", f"{total_convs:,}", f"{total_tokens:,}", f"{avg_all:.0f}",
                  style="bold")
    console.print(table)
    console.print(f"Shards written: {writer.shard_idx} pairs in {cfg.shard_dir}")


if __name__ == "__main__":
    main()
