"""Sharded token-stream dataset.

Shards are flat binary files of uint16 token ids written by scripts/prepare_data.py.
ShardedDataset memmaps every shard and yields sliding windows of max_seq_len + 1
tokens across the concatenated stream (input = window[:-1], target = window[1:]).
Windows never cross a shard boundary (shards are document-aligned enough that the
tiny loss at boundaries is irrelevant).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(shard_dir: Path) -> list[Path]:
    return sorted(Path(shard_dir).glob("shard_*.bin"))


def split_shards(shard_dir: Path, val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Split shards into (train, val) — the last N shards become val."""
    shards = list_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(
            f"No shards found in {shard_dir}. Run scripts/prepare_data.py first."
        )
    n_val = max(1, round(len(shards) * val_fraction)) if len(shards) > 1 else 0
    if n_val == 0:
        # single shard: carve val out of the same shard is not supported;
        # use the one shard for both (fine for smoke tests only)
        return shards, shards
    return shards[:-n_val], shards[-n_val:]


class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[Path], seq_len: int, stride: int | None = None):
        """
        seq_len: model context length; each item is (input[seq_len], target[seq_len])
        stride:  step between window starts (default seq_len, i.e. non-overlapping)
        """
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.shard_paths = [Path(p) for p in shard_paths]

        self._sizes = []            # token count per shard
        self._windows_per_shard = []
        for p in self.shard_paths:
            n_tokens = p.stat().st_size // 2  # uint16
            n_windows = max(0, (n_tokens - self.seq_len - 1) // self.stride + 1)
            self._sizes.append(n_tokens)
            self._windows_per_shard.append(n_windows)

        self._cum_windows = np.cumsum([0] + self._windows_per_shard)
        self._mmaps = [None] * len(self.shard_paths)  # lazy, per-worker

    def __len__(self):
        return int(self._cum_windows[-1])

    def _get_mmap(self, shard_idx):
        if self._mmaps[shard_idx] is None:
            self._mmaps[shard_idx] = np.memmap(
                self.shard_paths[shard_idx], dtype=np.uint16, mode="r"
            )
        return self._mmaps[shard_idx]

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self._cum_windows, idx, side="right") - 1)
        local_idx = idx - self._cum_windows[shard_idx]
        start = int(local_idx) * self.stride

        data = self._get_mmap(shard_idx)
        window = data[start : start + self.seq_len + 1].astype(np.int64)

        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:].copy())
        return x, y

    def __getstate__(self):
        # drop open memmaps so DataLoader workers re-open their own handles
        state = self.__dict__.copy()
        state["_mmaps"] = [None] * len(self.shard_paths)
        return state
