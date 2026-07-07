"""Sharded token-stream dataset.

Shards are flat binary files of uint16 token ids written by scripts/prepare_data.py,
named shard_{source}_{index:04d}.bin (e.g. shard_fineweb_0003.bin). ShardedDataset
memmaps every shard and yields sliding windows of max_seq_len + 1 tokens across the
concatenated stream (input = window[:-1], target = window[1:]). Windows never cross
a shard boundary (shards are document-aligned enough that the tiny loss at
boundaries is irrelevant).

For the Stage B diversity mix, make_weighted_sampler() builds a
WeightedRandomSampler whose per-window weights realize the configured
per-source mixing ratio (e.g. 80% fineweb / 10% books / 10% conv) regardless of
how many shards each source has on disk.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def list_shards(shard_dir: Path) -> list[Path]:
    return sorted(Path(shard_dir).glob("shard_*.bin"))


def shard_source(path: Path) -> str:
    """Extract the data source from a shard filename.

    shard_fineweb_0003.bin -> "fineweb"; legacy shard_0003.bin -> "fineweb"
    (Stage 0 shards had no source tag and were all FineWeb-Edu).
    """
    parts = Path(path).stem.split("_")
    return parts[1] if len(parts) == 3 else "fineweb"


def split_shards(
    shard_dir: Path,
    val_fraction: float,
    source_weights: dict[str, float] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Split shards into (train, val), holding out shards per source.

    Each source contributes its last shard(s) to val so validation covers the
    same mix as training. source_weights only needs to be provided to validate
    that every configured source actually has shards on disk.
    """
    shards = list_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(
            f"No shards found in {shard_dir}. Run scripts/prepare_data.py first."
        )

    by_source: dict[str, list[Path]] = {}
    for p in shards:
        by_source.setdefault(shard_source(p), []).append(p)

    if source_weights is not None:
        missing = [s for s, w in source_weights.items() if w > 0 and s not in by_source]
        if missing:
            raise FileNotFoundError(
                f"No shards on disk for source(s) {missing} in {shard_dir}. "
                "Run scripts/prepare_data.py first."
            )

    train_shards: list[Path] = []
    val_shards: list[Path] = []
    for source_shards in by_source.values():
        if len(source_shards) < 2:
            # single shard: use it for both (fine for smoke tests only)
            train_shards.extend(source_shards)
            val_shards.extend(source_shards)
            continue
        n_val = max(1, round(len(source_shards) * val_fraction))
        train_shards.extend(source_shards[:-n_val])
        val_shards.extend(source_shards[-n_val:])
    return sorted(train_shards), sorted(val_shards)


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


def make_weighted_sampler(
    ds: ShardedDataset, source_weights: dict[str, float]
) -> WeightedRandomSampler:
    """Sampler that draws windows so sources appear in source_weights ratio.

    Every window in a source gets weight w_source / n_windows_in_source, so the
    probability of drawing from a source equals its configured weight even when
    shard counts don't match the ratio. Sources on disk that are missing from
    source_weights get weight 0 (never sampled).
    """
    windows_per_source: dict[str, int] = {}
    for path, n_windows in zip(ds.shard_paths, ds._windows_per_shard):
        src = shard_source(path)
        windows_per_source[src] = windows_per_source.get(src, 0) + n_windows

    weights = np.empty(len(ds), dtype=np.float64)
    offset = 0
    for path, n_windows in zip(ds.shard_paths, ds._windows_per_shard):
        src = shard_source(path)
        w = source_weights.get(src, 0.0) / max(1, windows_per_source[src])
        weights[offset : offset + n_windows] = w
        offset += n_windows

    if weights.sum() <= 0:
        raise ValueError(
            "All sampling weights are zero — shard sources on disk don't match "
            f"source_weights keys {sorted(source_weights)}."
        )

    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(ds), replacement=True
    )
