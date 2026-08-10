"""Offline unit tests for BLIP-2 V2 contract and store.
Run without network requests or GPU requirement:
    python -m unittest discover -s tests -p 'test_blip2_*_v2.py' -v
"""

import unittest
from ab.nn.transform.blip2_cache_contract_v2 import (
    CACHE_FORMAT_VERSION,
    DATASET_ID,
    FEATURE_DTYPE,
    FEATURE_MODEL_ID,
    FEATURE_SHAPE,
    GPT2_MODEL_ID,
    CacheManifest,
    normalize_split,
    canonical_shard_name,
)

class TestBlip2ContractV2(unittest.TestCase):

    def test_normalize_split(self):
        self.assertEqual(normalize_split("train"), "train")
        self.assertEqual(normalize_split("val"), "val")
        self.assertEqual(normalize_split("validation"), "val")
        self.assertEqual(normalize_split("VALID"), "val")

    def test_canonical_shard_name(self):
        self.assertEqual(canonical_shard_name("train", 0), "coco_train_00000.pt")
        self.assertEqual(canonical_shard_name("val", 15), "coco_val_00015.pt")

    def test_empty_manifest(self):
        manifest = CacheManifest.empty()
        self.assertEqual(manifest.format_version, CACHE_FORMAT_VERSION)
        self.assertEqual(manifest.dataset, DATASET_ID)
        self.assertEqual(manifest.feature_model, FEATURE_MODEL_ID)
        self.assertEqual(manifest.feature_shape, FEATURE_SHAPE)
        self.assertEqual(manifest.feature_dtype, FEATURE_DTYPE)
        self.assertEqual(manifest.tokenizer, GPT2_MODEL_ID)

if __name__ == "__main__":
    unittest.main()
