import tempfile
import unittest
from pathlib import Path

import torch

from ab.nn.captioning.blip2.contract import (
    CACHE_VERSION,
    FEATURE_SHAPE,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    atomic_json,
    sha256_file,
)
from ab.nn.captioning.blip2.shard_aware import ShardAwareCachedCaptionDataset


class TestShardAwareCachedCaptionDataset(unittest.TestCase):
    def test_framework_preserves_approach_suffix_in_model_name(self):
        from ab.nn.util.Util import conf_to_names

        self.assertEqual(
            conf_to_names(
                "img-captioning_coco_bleu,meteor,cider_Blip2Cached_ShardAware"
            ),
            (
                "img-captioning",
                "coco",
                "bleu,meteor,cider",
                "Blip2Cached_ShardAware",
            ),
        )

    def make_cache(self, root, counts=(3, 2, 4)):
        shards = []
        offset = 0
        for position, count in enumerate(counts):
            path = root / f"train-{position:05d}.pt"
            captions = [[str(index)] for index in range(offset, offset + count)]
            torch.save(
                {
                    "features": torch.zeros(count, *FEATURE_SHAPE, dtype=torch.float16),
                    "captions": captions,
                    "image_ids": list(range(offset, offset + count)),
                },
                path,
            )
            shards.append({
                "filename": path.name,
                "samples": count,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            offset += count
        atomic_json(root / MANIFEST_NAME, {
            "cache_version": CACHE_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "feature_shape": list(FEATURE_SHAPE),
            "feature_dtype": "float16",
            "projection": {},
            "splits": {"train": {"complete": True, "samples": offset, "shards": shards}},
        })

    def test_epoch_covers_every_sample_once_and_changes_deterministically(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.make_cache(root)
            dataset = ShardAwareCachedCaptionDataset(cache_dir=root, seed=7)
            first_order = list(dataset._order)
            first = dataset.__getitems__([0] * len(dataset))
            self.assertEqual(sorted(int(item[1][0]) for item in first), list(range(9)))
            second = dataset.__getitems__([0] * len(dataset))
            self.assertEqual(sorted(int(item[1][0]) for item in second), list(range(9)))
            self.assertNotEqual(first_order, dataset._order)
            again = ShardAwareCachedCaptionDataset(cache_dir=root, seed=7)
            self.assertEqual(first_order, again._order)

    def test_single_item_shape_probe_does_not_advance_cursor(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.make_cache(root)
            dataset = ShardAwareCachedCaptionDataset(cache_dir=root, seed=7)
            dataset.__getitems__([2])
            self.assertEqual(dataset._cursor, 0)

    def test_transform_keeps_validation_on_the_baseline_dataset(self):
        from ab.nn.captioning.blip2.cache import CachedCaptionDataset
        from ab.nn.transform.blip2_cached_shard_aware import get_dataset

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.make_cache(root, counts=(2,))
            manifest = __import__("json").loads((root / MANIFEST_NAME).read_text())
            train_record = manifest["splits"]["train"]
            source = root / train_record["shards"][0]["filename"]
            target = root / "val-00000.pt"
            target.write_bytes(source.read_bytes())
            manifest["splits"]["val"] = {
                "complete": True,
                "samples": 2,
                "shards": [{**train_record["shards"][0], "filename": target.name}],
            }
            atomic_json(root / MANIFEST_NAME, manifest)
            self.assertIsInstance(get_dataset("train", root), ShardAwareCachedCaptionDataset)
            validation = get_dataset("val", root)
            self.assertIsInstance(validation, CachedCaptionDataset)
            self.assertNotIsInstance(validation, ShardAwareCachedCaptionDataset)


if __name__ == "__main__":
    unittest.main()
