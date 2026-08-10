"""Shared, dependency-light contract for BLIP-2 feature-cache v2.

This module contains only paths, metadata models, manifest parsing, and file
integrity helpers. It deliberately imports neither PyTorch nor Transformers so
cache metadata can be inspected in small CI jobs and before a GPU environment
is initialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 2
LEGACY_CACHE_FORMAT_VERSION = 1
MANIFEST_FILENAME = "blip2_cache_manifest_v2.json"

DATASET_ID = "coco-2017"
FEATURE_MODEL_ID = "Salesforce/blip2-opt-2.7b"
FEATURE_SHAPE = (32, 768)
FEATURE_DTYPE = "float16"
GPT2_MODEL_ID = "gpt2"
GPT2_VOCAB_SIZE = 50_257

SUPPORTED_SPLITS = frozenset({"train", "val"})
DEFAULT_EXPECTED_SPLIT_SIZES = {
    "train": 118_287,
    "val": 5_000,
}

_V2_SHARD_PATTERN = re.compile(r"^coco_(train|val)_(\d{5})\.pt$")
_LEGACY_SHARD_PATTERN = re.compile(
    r"^coco_(train|val)_(\d+|final)\.pt$"
)


class CacheContractError(ValueError):
    """Raised when cache metadata violates the public v2 contract."""


def normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    aliases = {"validation": "val", "valid": "val"}
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_SPLITS:
        raise CacheContractError(
            f"Unsupported split {split!r}; expected one of "
            f"{sorted(SUPPORTED_SPLITS)}."
        )
    return normalized


def resolve_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit path, environment override, or repository default."""
    configured = cache_dir or os.environ.get("BLIP2_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    repository_root = Path(__file__).resolve().parents[3]
    return (repository_root / "out" / "cache").resolve()


def manifest_path(cache_dir: str | os.PathLike[str]) -> Path:
    return Path(cache_dir).expanduser().resolve() / MANIFEST_FILENAME


def expected_split_size(split: str) -> int:
    normalized = normalize_split(split)
    variable = f"BLIP2_EXPECTED_{normalized.upper()}_SIZE"
    raw_value = os.environ.get(
        variable,
        str(DEFAULT_EXPECTED_SPLIT_SIZES[normalized]),
    )

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise CacheContractError(
            f"{variable} must be a non-negative integer, got {raw_value!r}."
        ) from error

    if value < 0:
        raise CacheContractError(
            f"{variable} must be non-negative, got {value}."
        )
    return value


def canonical_shard_name(split: str, shard_index: int) -> str:
    normalized = normalize_split(split)
    index = int(shard_index)
    if index < 0:
        raise CacheContractError("shard_index must be non-negative.")
    return f"coco_{normalized}_{index:05d}.pt"


def _validate_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise CacheContractError(f"Invalid SHA-256 digest: {value!r}.")
    return normalized


def _safe_shard_filename(filename: str, split: str, *, legacy: bool) -> str:
    name = str(filename)
    if name != Path(name).name or Path(name).is_absolute():
        raise CacheContractError(
            f"Shard filename must be a safe basename, got {filename!r}."
        )

    pattern = _LEGACY_SHARD_PATTERN if legacy else _V2_SHARD_PATTERN
    match = pattern.fullmatch(name)
    if match is None:
        label = "legacy" if legacy else "v2"
        raise CacheContractError(
            f"Invalid {label} shard filename for split {split!r}: {name!r}."
        )
    if match.group(1) != split:
        raise CacheContractError(
            f"Shard {name!r} belongs to {match.group(1)!r}, not {split!r}."
        )
    return name


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 4 << 20) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ShardRecord:
    filename: str
    sample_count: int
    sha256: str | None
    size_bytes: int | None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        split: str,
        *,
        legacy: bool = False,
    ) -> "ShardRecord":
        if not isinstance(value, dict):
            raise CacheContractError("Each shard record must be an object.")

        filename = _safe_shard_filename(
            value.get("filename", ""),
            split,
            legacy=legacy,
        )
        sample_count = int(value.get("sample_count", 0))
        if sample_count <= 0:
            raise CacheContractError(
                f"Shard {filename!r} must have a positive sample_count."
            )

        raw_size = value.get("size_bytes")
        size_bytes = None if raw_size is None else int(raw_size)
        if size_bytes is not None and size_bytes <= 0:
            raise CacheContractError(
                f"Shard {filename!r} must have a positive size_bytes."
            )

        sha256 = _validate_sha256(value.get("sha256"))
        if not legacy and (sha256 is None or size_bytes is None):
            raise CacheContractError(
                f"V2 shard {filename!r} requires sha256 and size_bytes."
            )

        return cls(
            filename=filename,
            sample_count=sample_count,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitRecord:
    sample_count: int
    shards: tuple[ShardRecord, ...]

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        split: str,
        *,
        legacy: bool = False,
    ) -> "SplitRecord":
        if not isinstance(value, dict):
            raise CacheContractError(
                f"Manifest split {split!r} must be an object."
            )

        raw_shards = value.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise CacheContractError(
                f"Manifest split {split!r} requires a non-empty shard list."
            )

        if legacy and all(isinstance(item, str) for item in raw_shards):
            counts = value.get("shard_sample_counts")
            if not isinstance(counts, list) or len(counts) != len(raw_shards):
                raise CacheContractError(
                    "Legacy manifests do not contain per-shard counts. "
                    "Use the cache store's explicit legacy discovery path."
                )
            raw_shards = [
                {"filename": name, "sample_count": count}
                for name, count in zip(raw_shards, counts, strict=True)
            ]

        shards = tuple(
            ShardRecord.from_dict(item, split, legacy=legacy)
            for item in raw_shards
        )
        filenames = [record.filename for record in shards]
        if len(filenames) != len(set(filenames)):
            raise CacheContractError(
                f"Manifest split {split!r} contains duplicate shard names."
            )

        sample_count = int(value.get("sample_count", 0))
        calculated_count = sum(record.sample_count for record in shards)
        if sample_count != calculated_count:
            raise CacheContractError(
                f"Manifest split {split!r} reports {sample_count} samples, "
                f"but its shards report {calculated_count}."
            )

        return cls(sample_count=sample_count, shards=shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class CacheManifest:
    splits: dict[str, SplitRecord] = field(default_factory=dict)
    format_version: int = CACHE_FORMAT_VERSION
    dataset: str = DATASET_ID
    feature_model: str = FEATURE_MODEL_ID
    feature_shape: tuple[int, int] = FEATURE_SHAPE
    feature_dtype: str = FEATURE_DTYPE
    tokenizer: str = GPT2_MODEL_ID

    @classmethod
    def empty(cls) -> "CacheManifest":
        return cls()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CacheManifest":
        if not isinstance(value, dict):
            raise CacheContractError("Cache manifest must contain an object.")

        format_version = int(value.get("format_version", -1))
        if format_version != CACHE_FORMAT_VERSION:
            raise CacheContractError(
                f"Unsupported manifest version {format_version}; expected "
                f"{CACHE_FORMAT_VERSION}."
            )

        if value.get("dataset") != DATASET_ID:
            raise CacheContractError(
                f"Cache dataset must be {DATASET_ID!r}."
            )
        if value.get("feature_model") != FEATURE_MODEL_ID:
            raise CacheContractError(
                f"Cache feature_model must be {FEATURE_MODEL_ID!r}."
            )
        if tuple(value.get("feature_shape", ())) != FEATURE_SHAPE:
            raise CacheContractError(
                f"Cache feature_shape must be {FEATURE_SHAPE}."
            )
        if value.get("feature_dtype") != FEATURE_DTYPE:
            raise CacheContractError(
                f"Cache feature_dtype must be {FEATURE_DTYPE!r}."
            )
        if value.get("tokenizer") != GPT2_MODEL_ID:
            raise CacheContractError(
                f"Cache tokenizer must be {GPT2_MODEL_ID!r}."
            )

        raw_splits = value.get("splits", {})
        if not isinstance(raw_splits, dict):
            raise CacheContractError("Manifest splits must be an object.")

        splits: dict[str, SplitRecord] = {}
        for raw_split, raw_record in raw_splits.items():
            split = normalize_split(raw_split)
            if split in splits:
                raise CacheContractError(f"Duplicate split {split!r}.")
            splits[split] = SplitRecord.from_dict(raw_record, split)

        return cls(splits=splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset": self.dataset,
            "feature_model": self.feature_model,
            "feature_shape": list(self.feature_shape),
            "feature_dtype": self.feature_dtype,
            "tokenizer": self.tokenizer,
            "splits": {
                split: record.to_dict()
                for split, record in sorted(self.splits.items())
            },
        }

    def with_split(
        self,
        split: str,
        record: SplitRecord,
    ) -> "CacheManifest":
        normalized = normalize_split(split)
        updated = dict(self.splits)
        updated[normalized] = record
        return CacheManifest(splits=updated)


def load_manifest(cache_dir: str | os.PathLike[str]) -> CacheManifest:
    path = manifest_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"BLIP-2 v2 manifest is missing: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise CacheContractError(f"Could not read cache manifest: {path}") from error

    return CacheManifest.from_dict(raw)


def save_manifest_atomic(
    manifest: CacheManifest,
    cache_dir: str | os.PathLike[str],
) -> Path:
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = manifest_path(directory)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_FILENAME}.",
        suffix=".temporary",
        dir=directory,
        text=True,
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination
