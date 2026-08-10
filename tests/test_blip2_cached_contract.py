import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from ab.nn.captioning.blip2.cache import CachedCaptionDataset

from ab.nn.captioning.blip2.contract import (
    CACHE_VERSION,
    FEATURE_SHAPE,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    OPT_VOCAB_SIZE,
    CacheError,
    atomic_json,
    read_manifest,
    resolve_projection_path,
    sha256_file,
)
from ab.nn.tools.build_blip2_cached import _manifest


class TestBlip2CachedContract(unittest.TestCase):
    def test_tested_environment_contract_passes(self):
        from ab.nn.captioning.blip2.environment import validate_environment

        validate_environment()

    def test_environment_contract_rejects_transformers_drift(self):
        from ab.nn.captioning.blip2.environment import (
            EnvironmentCompatibilityError,
            validate_environment,
        )

        def fake_version(package):
            return "5.14.1" if package == "transformers" else "2.9.1"

        with patch("ab.nn.captioning.blip2.environment.version", fake_version):
            with self.assertRaisesRegex(EnvironmentCompatibilityError, "4.57.6"):
                validate_environment()

    def test_projection_paths_are_not_optuna_search_parameters(self):
        from ab.nn.nn.Blip2Cached import supported_hyperparameters

        self.assertEqual(supported_hyperparameters(), {"lr", "batch"})

    def test_builder_rejects_wrong_revision_before_resume(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            manifest = self.manifest()
            manifest["model_revision"] = "wrong-revision"
            atomic_json(root / MANIFEST_NAME, manifest)
            with self.assertRaisesRegex(RuntimeError, "model_revision"):
                _manifest(root)

    def test_builder_rejects_wrong_feature_contract_before_resume(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            manifest = self.manifest()
            manifest["feature_shape"] = [1, 2]
            manifest["feature_dtype"] = "float32"
            atomic_json(root / MANIFEST_NAME, manifest)
            with self.assertRaisesRegex(RuntimeError, "feature_shape, feature_dtype"):
                _manifest(root)

    def test_projection_path_defaults_inside_cache(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value).resolve()
            self.assertEqual(
                resolve_projection_path(root),
                root / "language_projection.pt",
            )

    def test_relative_projection_path_is_portable_inside_cache(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value).resolve()
            self.assertEqual(
                resolve_projection_path(root, "checkpoints/trained_projection.pt"),
                root / "checkpoints" / "trained_projection.pt",
            )

    def test_opt_vocabulary_contract(self):
        from ab.nn.transform.blip2_cached import get_vocab_size

        self.assertEqual(OPT_VOCAB_SIZE, 50272)
        self.assertEqual(get_vocab_size(), (OPT_VOCAB_SIZE,))

    def manifest(self):
        return {
            "cache_version": CACHE_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "feature_shape": list(FEATURE_SHAPE),
            "feature_dtype": "float16",
            "projection": {},
            "splits": {},
        }

    def test_atomic_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            atomic_json(root / MANIFEST_NAME, self.manifest())
            self.assertEqual(read_manifest(root)["model_id"], MODEL_ID)

    def test_rejects_wrong_model(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            manifest = self.manifest()
            manifest["model_id"] = "wrong/model"
            atomic_json(root / MANIFEST_NAME, manifest)
            with self.assertRaises(CacheError):
                read_manifest(root)

    def test_streaming_checksum(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "data.bin"
            path.write_bytes(b"portable-cache")
            expected = hashlib.sha256(b"portable-cache").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_validated_dataset_reads_shard(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            shard = root / "train-00000.pt"
            torch.save(
                {
                    "features": torch.zeros(2, *FEATURE_SHAPE, dtype=torch.float16),
                    "captions": [["one"], ["two"]],
                    "image_ids": [1, 2],
                },
                shard,
            )
            manifest = self.manifest()
            manifest["splits"] = {
                "train": {
                    "complete": True,
                    "samples": 2,
                    "shards": [{
                        "filename": shard.name,
                        "samples": 2,
                        "size_bytes": shard.stat().st_size,
                        "sha256": sha256_file(shard),
                    }],
                }
            }
            atomic_json(root / MANIFEST_NAME, manifest)
            dataset = CachedCaptionDataset("train", str(root))
            feature, captions = dataset[1]
            self.assertEqual(tuple(feature.shape), FEATURE_SHAPE)
            self.assertEqual(captions, ["two"])

    def test_dataset_limit_is_explicit_and_deterministic(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            shard = root / "train-00000.pt"
            torch.save(
                {
                    "features": torch.zeros(2, *FEATURE_SHAPE, dtype=torch.float16),
                    "captions": [["one"], ["two"]],
                    "image_ids": [1, 2],
                },
                shard,
            )
            manifest = self.manifest()
            manifest["splits"] = {
                "train": {
                    "complete": True,
                    "samples": 2,
                    "shards": [{
                        "filename": shard.name,
                        "samples": 2,
                        "size_bytes": shard.stat().st_size,
                        "sha256": sha256_file(shard),
                    }],
                }
            }
            atomic_json(root / MANIFEST_NAME, manifest)
            previous = os.environ.get("BLIP2_TRAIN_LIMIT")
            os.environ["BLIP2_TRAIN_LIMIT"] = "1"
            try:
                dataset = CachedCaptionDataset("train", str(root))
                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset[0][1], ["one"])
            finally:
                if previous is None:
                    os.environ.pop("BLIP2_TRAIN_LIMIT", None)
                else:
                    os.environ["BLIP2_TRAIN_LIMIT"] = previous


if __name__ == "__main__":
    unittest.main()
