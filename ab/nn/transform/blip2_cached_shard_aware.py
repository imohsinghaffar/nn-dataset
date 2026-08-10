"""Transform adapter for the separate shard-aware BLIP-2 experiment."""

from ab.nn.captioning.blip2.cache import CachedCaptionDataset
from ab.nn.captioning.blip2.contract import OPT_VOCAB_SIZE
from ab.nn.captioning.blip2.shard_aware import ShardAwareCachedCaptionDataset


def transform(norm=None):
    del norm
    return None


def get_dataset(split="train", cache_dir=None):
    if split == "train":
        return ShardAwareCachedCaptionDataset(split, cache_dir=cache_dir)
    return CachedCaptionDataset(split, cache_dir=cache_dir)


def get_vocab_size():
    return (OPT_VOCAB_SIZE,)

