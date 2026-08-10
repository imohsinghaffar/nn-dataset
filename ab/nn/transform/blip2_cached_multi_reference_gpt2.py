"""Transform adapter for portable cached BLIP-2 + GPT-2 captioning."""

from ab.nn.captioning.blip2.cache import CachedCaptionDataset
from ab.nn.captioning.blip2.gpt2 import GPT2_VOCAB_SIZE, collator


def transform(norm=None):
    del norm
    return None


def get_dataset(split="train", cache_dir=None):
    dataset = CachedCaptionDataset(split=split, cache_dir=cache_dir)
    dataset.collate_fn = collator(dataset.cache_dir)
    return dataset


def get_vocab_size():
    return (GPT2_VOCAB_SIZE,)
