"""Shard-local sampling for the separate BLIP-2 cache speed experiment."""

from __future__ import annotations

import os
import random

from .cache import CachedCaptionDataset


class ShardAwareCachedCaptionDataset(CachedCaptionDataset):
    """Serve shuffled shards contiguously without preloading the full cache.

    NN-Dataset's generic training loader owns a random sampler. PyTorch passes
    each sampled batch to ``__getitems__`` when that hook exists. For batches
    larger than one, this experimental dataset replaces the global random
    indices with a deterministic epoch order: shuffled shards containing
    shuffled local samples. This preserves full sample coverage while keeping
    cache access local. Single-item requests remain ordinary indexed reads so
    the framework's input-shape probe does not consume the epoch cursor.
    """

    def __init__(self, split="train", cache_dir=None, verify=True, seed=None):
        if split != "train":
            raise ValueError("Shard-aware sampling is only valid for the training split.")
        super().__init__(split=split, cache_dir=cache_dir, verify=verify)
        raw_seed = os.environ.get("BLIP2_SHARD_SEED", "42") if seed is None else seed
        try:
            self.seed = int(raw_seed)
        except (TypeError, ValueError) as error:
            raise ValueError(f"BLIP2_SHARD_SEED must be an integer, got {raw_seed!r}.") from error
        self._epoch = 0
        self._cursor = 0
        self._order = self._make_epoch_order()

    def _make_epoch_order(self):
        rng = random.Random(self.seed + self._epoch)
        groups = []
        start = 0
        remaining = len(self)
        for shard in self.shards:
            count = min(int(shard["samples"]), remaining)
            if count <= 0:
                break
            indices = list(range(start, start + count))
            rng.shuffle(indices)
            groups.append(indices)
            start += count
            remaining -= count
        rng.shuffle(groups)
        return [index for group in groups for index in group]

    def _next_epoch(self):
        self._epoch += 1
        self._cursor = 0
        self._order = self._make_epoch_order()

    def __getitems__(self, indices):
        size = len(indices)
        if size <= 1:
            return [self[index] for index in indices]
        if self._cursor == len(self):
            self._next_epoch()
        end = self._cursor + size
        if end > len(self):
            raise RuntimeError("Shard-aware batch crossed the declared epoch length.")
        selected = self._order[self._cursor:end]
        self._cursor = end
        return [self[index] for index in selected]

