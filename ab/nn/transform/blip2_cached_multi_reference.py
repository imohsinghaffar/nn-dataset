"""Transform adapter for ``Blip2Cached_MultiReference``.

The cache and collator remain identical to the verified baseline.  Reference
selection belongs to the separate model so validation retains all references
for the caption metrics.
"""

from ab.nn.captioning.blip2.cache import CachedCaptionDataset
from ab.nn.captioning.blip2.contract import OPT_VOCAB_SIZE


def transform(norm=None):
    del norm
    return None


def get_dataset(split="train", cache_dir=None):
    return CachedCaptionDataset(split=split, cache_dir=cache_dir)


def get_vocab_size():
    return (OPT_VOCAB_SIZE,)
