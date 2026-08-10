"""Validated shard storage for the BLIP-2 feature-cache v2.

Training reads cache data through :class:`CacheStore`; feature extraction writes
through :class:`CacheWriter`. Neither class loads COCO or a Hugging Face model,
which keeps dataset loading free of recursive side effects.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock, Timeout

from ab.nn.transform.blip2_cache_contract_v2 import (
    CACHE_FORMAT_VERSION,
    DATASET_ID,
    FEATURE_DTYPE,
    FEATURE_MODEL_ID,
    FEATURE_SHAPE,
    GPT2_MODEL_ID,
    CacheContractError,
    CacheManifest,
    ShardRecord,
    SplitRecord,
    canonical_shard_name,
    expected_split_size,
    load_manifest,
    manifest_path,
    normalize_split,
    resolve_cache_dir,
    save_manifest_atomic,
    sha256_file,
)


_LEGACY_SHARD_PATTERN = re.compile(
    r"^coco_(train|val)_(\d+|final)\.pt$"
)


class CacheMissingError(FileNotFoundError):
    """Raised when a required split has not been built."""


class CacheIntegrityError(RuntimeError):
    """Raised when a shard or manifest is incomplete, stale, or corrupt."""


def _environment_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean flag, received {raw!r}."
    )


def _train_limit() -> int:
    raw = os.environ.get("NN_TRAIN_LIMIT", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"NN_TRAIN_LIMIT must be non-negative, received {raw!r}."
        ) from error
    if value < 0:
        raise ValueError(
            f"NN_TRAIN_LIMIT must be non-negative, received {value}."
        )
    return value


def _legacy_sort_key(path: Path) -> tuple[int, int]:
    match = _LEGACY_SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        raise CacheIntegrityError(
            f"Invalid legacy cache shard name: {path.name}"
        )
    suffix = match.group(2)
    return (1, 0) if suffix == "final" else (0, int(suffix))


def _load_torch_payload(path: Path) -> Any:
    """Prefer memory mapping without breaking older supported PyTorch builds."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (TypeError, RuntimeError, ValueError):
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def _validate_labels(labels: Any, path: Path) -> list[Any]:
    if not isinstance(labels, (list, tuple)):
        raise CacheIntegrityError(
            f"Shard labels must be a list or tuple: {path}"
        )
    return list(labels)


@dataclass(frozen=True)
class LoadedShard:
    features: torch.Tensor
    labels: list[Any]


@dataclass(frozen=True)
class SplitIndex:
    split: str
    records: tuple[ShardRecord, ...]
    offsets: tuple[int, ...]
    full_length: int
    view_length: int
    legacy: bool = False

    def locate(self, index: int) -> tuple[int, int]:
        import bisect

        if index < 0:
            index += self.view_length
        if index < 0 or index >= self.view_length:
            raise IndexError(
                f"Cache index {index} is outside [0, {self.view_length})."
            )

        shard_index = bisect.bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[shard_index]
        return shard_index, local_index


class CacheStore:
    """Read and validate cache metadata and individual shards."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        verify_hashes: bool | None = None,
        allow_legacy: bool | None = None,
    ):
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.verify_hashes = (
            _environment_flag("BLIP2_VERIFY_CACHE_HASHES", False)
            if verify_hashes is None
            else bool(verify_hashes)
        )
        self.allow_legacy = (
            _environment_flag("BLIP2_ALLOW_LEGACY_CACHE", True)
            if allow_legacy is None
            else bool(allow_legacy)
        )
        self._indices: dict[str, SplitIndex] = {}

    def _missing_message(self, split: str) -> str:
        return (
            f"BLIP-2 cache split {split!r} is unavailable in "
            f"{self.cache_dir}. Build it explicitly with:\n"
            "python -m ab.nn.tools.build_blip2_cache_v2 "
            f"--split {split} --cache-dir {self.cache_dir}"
        )

    def index(self, split: str) -> SplitIndex:
        normalized = normalize_split(split)
        if normalized in self._indices:
            return self._indices[normalized]

        try:
            manifest = load_manifest(self.cache_dir)
        except FileNotFoundError:
            if not self.allow_legacy:
                raise CacheMissingError(
                    self._missing_message(normalized)
                ) from None
            index = self._legacy_index(normalized)
        except CacheContractError as error:
            raise CacheIntegrityError(str(error)) from error
        else:
            index = self._manifest_index(manifest, normalized)

        self._indices[normalized] = index
        return index

    def _manifest_index(
        self,
        manifest: CacheManifest,
        split: str,
    ) -> SplitIndex:
        record = manifest.splits.get(split)
        if record is None:
            if self.allow_legacy:
                return self._legacy_index(split)
            raise CacheMissingError(self._missing_message(split))

        for shard in record.shards:
            path = self.cache_dir / shard.filename
            if not path.is_file():
                raise CacheIntegrityError(f"Cache shard is missing: {path}")
            size = path.stat().st_size
            if shard.size_bytes is not None and size != shard.size_bytes:
                raise CacheIntegrityError(
                    f"Cache shard size mismatch for {path}: expected "
                    f"{shard.size_bytes}, received {size}."
                )
            if self.verify_hashes and shard.sha256 is not None:
                digest = sha256_file(path)
                if digest != shard.sha256:
                    raise CacheIntegrityError(
                        f"Cache shard SHA-256 mismatch: {path}"
                    )

        expected = expected_split_size(split)
        if expected > 0 and record.sample_count != expected:
            raise CacheIntegrityError(
                f"Incomplete {split!r} cache: expected {expected} samples, "
                f"manifest reports {record.sample_count}."
            )

        return self._make_index(
            split,
            record.shards,
            record.sample_count,
            legacy=False,
        )

    def _legacy_index(self, split: str) -> SplitIndex:
        if not self.cache_dir.is_dir():
            raise CacheMissingError(self._missing_message(split))

        paths = [
            path
            for path in self.cache_dir.iterdir()
            if path.is_file()
            and (match := _LEGACY_SHARD_PATTERN.fullmatch(path.name))
            and match.group(1) == split
        ]
        paths.sort(key=_legacy_sort_key)
        if not paths:
            raise CacheMissingError(self._missing_message(split))

        records: list[ShardRecord] = []
        for path in paths:
            payload = self._validate_payload(
                path,
                split,
                expected_count=None,
                require_v2_metadata=False,
            )
            records.append(
                ShardRecord(
                    filename=path.name,
                    sample_count=int(payload.features.shape[0]),
                    sha256=None,
                    size_bytes=path.stat().st_size,
                )
            )

        full_length = sum(record.sample_count for record in records)
        expected = expected_split_size(split)
        if expected > 0 and full_length != expected:
            raise CacheIntegrityError(
                f"Incomplete legacy {split!r} cache: expected {expected} "
                f"samples, found {full_length}."
            )

        return self._make_index(
            split,
            tuple(records),
            full_length,
            legacy=True,
        )

    @staticmethod
    def _make_index(
        split: str,
        records: tuple[ShardRecord, ...],
        full_length: int,
        *,
        legacy: bool,
    ) -> SplitIndex:
        offsets = [0]
        for record in records:
            offsets.append(offsets[-1] + record.sample_count)

        limit = _train_limit() if split == "train" else 0
        view_length = min(full_length, limit) if limit > 0 else full_length
        return SplitIndex(
            split=split,
            records=tuple(records),
            offsets=tuple(offsets),
            full_length=full_length,
            view_length=view_length,
            legacy=legacy,
        )

    def load_shard(
        self,
        split: str,
        record: ShardRecord,
        *,
        legacy: bool = False,
    ) -> LoadedShard:
        normalized = normalize_split(split)
        path = self.cache_dir / record.filename
        return self._validate_payload(
            path,
            normalized,
            expected_count=record.sample_count,
            require_v2_metadata=not legacy,
        )

    @staticmethod
    def _validate_payload(
        path: Path,
        split: str,
        *,
        expected_count: int | None,
        require_v2_metadata: bool,
    ) -> LoadedShard:
        try:
            payload = _load_torch_payload(path)
        except Exception as error:
            raise CacheIntegrityError(
                f"Cache shard could not be loaded: {path}"
            ) from error

        if not isinstance(payload, dict):
            raise CacheIntegrityError(
                f"Cache shard must contain an object: {path}"
            )

        missing = {"features", "labels", "split"}.difference(payload)
        if missing:
            raise CacheIntegrityError(
                f"Cache shard is missing {sorted(missing)}: {path}"
            )
        if normalize_split(payload["split"]) != split:
            raise CacheIntegrityError(
                f"Cache split mismatch in {path}: requested {split!r}, "
                f"stored {payload['split']!r}."
            )

        if require_v2_metadata:
            expected_metadata = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "dataset": DATASET_ID,
                "feature_model": FEATURE_MODEL_ID,
                "feature_shape": list(FEATURE_SHAPE),
                "feature_dtype": FEATURE_DTYPE,
                "tokenizer": GPT2_MODEL_ID,
            }
            for key, expected_value in expected_metadata.items():
                if payload.get(key) != expected_value:
                    raise CacheIntegrityError(
                        f"Cache metadata mismatch for {key!r} in {path}: "
                        f"expected {expected_value!r}, received "
                        f"{payload.get(key)!r}."
                    )

        features = payload["features"]
        if not torch.is_tensor(features):
            raise CacheIntegrityError(f"features must be a tensor: {path}")
        if features.device.type != "cpu":
            raise CacheIntegrityError(
                f"features must be stored on CPU, got {features.device}: {path}"
            )
        if features.dtype != torch.float16:
            raise CacheIntegrityError(
                f"features must be float16, got {features.dtype}: {path}"
            )
        if features.ndim != 3 or tuple(features.shape[1:]) != FEATURE_SHAPE:
            raise CacheIntegrityError(
                f"features must have shape (N,{FEATURE_SHAPE[0]},"
                f"{FEATURE_SHAPE[1]}), got {tuple(features.shape)}: {path}"
            )
        if int(features.shape[0]) <= 0:
            raise CacheIntegrityError(f"Empty cache shard: {path}")
        if not torch.isfinite(features).all():
            raise CacheIntegrityError(f"NaN or Inf in cache shard: {path}")

        labels = _validate_labels(payload["labels"], path)
        actual_count = int(features.shape[0])
        if actual_count != len(labels):
            raise CacheIntegrityError(
                f"Feature/label mismatch in {path}: {actual_count} vs "
                f"{len(labels)}."
            )
        if expected_count is not None and actual_count != expected_count:
            raise CacheIntegrityError(
                f"Shard count mismatch in {path}: expected {expected_count}, "
                f"received {actual_count}."
            )

        return LoadedShard(features=features, labels=labels)


class CacheWriter:
    """Create v2 shards and atomically activate a complete split."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        lock_timeout_seconds: int | None = None,
    ):
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_seconds = int(
            lock_timeout_seconds
            if lock_timeout_seconds is not None
            else os.environ.get("BLIP2_CACHE_LOCK_TIMEOUT", "14400")
        )
        if self.lock_timeout_seconds <= 0:
            raise ValueError("Cache lock timeout must be greater than zero.")

    def create_build_dir(self, split: str) -> Path:
        normalized = normalize_split(split)
        return Path(
            tempfile.mkdtemp(
                prefix=f".blip2_{normalized}_build_",
                dir=self.cache_dir,
            )
        )

    def write_shard(
        self,
        build_dir: str | os.PathLike[str],
        split: str,
        shard_index: int,
        features: torch.Tensor,
        labels: list[Any],
    ) -> ShardRecord:
        normalized = normalize_split(split)
        directory = Path(build_dir).resolve()
        if directory.parent != self.cache_dir:
            raise ValueError(
                f"Build directory must be directly inside {self.cache_dir}."
            )
        if not directory.is_dir():
            raise FileNotFoundError(f"Build directory is missing: {directory}")

        filename = canonical_shard_name(normalized, shard_index)
        destination = directory / filename
        temporary = directory / f".{filename}.temporary"

        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "dataset": DATASET_ID,
            "feature_model": FEATURE_MODEL_ID,
            "feature_shape": list(FEATURE_SHAPE),
            "feature_dtype": FEATURE_DTYPE,
            "tokenizer": GPT2_MODEL_ID,
            "split": normalized,
            "features": features.detach().cpu().to(torch.float16),
            "labels": list(labels),
        }

        try:
            torch.save(payload, temporary)
            sample_count = int(payload["features"].shape[0])
            self._validate_payload_for_build(
                temporary,
                normalized,
                sample_count,
            )
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        return ShardRecord(
            filename=filename,
            sample_count=sample_count,
            sha256=sha256_file(destination),
            size_bytes=destination.stat().st_size,
        )

    @staticmethod
    def _validate_payload_for_build(
        path: Path,
        split: str,
        sample_count: int,
    ) -> None:
        CacheStore._validate_payload(
            path,
            split,
            expected_count=sample_count,
            require_v2_metadata=True,
        )

    def activate_split(
        self,
        build_dir: str | os.PathLike[str],
        split: str,
        records: list[ShardRecord],
    ) -> Path:
        normalized = normalize_split(split)
        directory = Path(build_dir).resolve()
        if directory.parent != self.cache_dir or not directory.is_dir():
            raise ValueError("Invalid cache build directory.")
        if not records:
            raise ValueError("Cannot activate an empty cache split.")

        sample_count = sum(record.sample_count for record in records)
        expected = expected_split_size(normalized)
        if expected > 0 and sample_count != expected:
            raise CacheIntegrityError(
                f"Refusing to activate {normalized!r}: expected {expected} "
                f"samples, generated {sample_count}."
            )

        for record in records:
            path = directory / record.filename
            if not path.is_file():
                raise CacheIntegrityError(f"Built shard is missing: {path}")
            if path.stat().st_size != record.size_bytes:
                raise CacheIntegrityError(f"Built shard size changed: {path}")
            if sha256_file(path) != record.sha256:
                raise CacheIntegrityError(f"Built shard digest changed: {path}")
            self._validate_payload_for_build(
                path,
                normalized,
                record.sample_count,
            )

        lock_path = self.cache_dir.parent / ".blip2_cache_v2.lock"
        try:
            with FileLock(
                str(lock_path),
                timeout=self.lock_timeout_seconds,
            ):
                return self._activate_locked(
                    directory,
                    normalized,
                    records,
                )
        except Timeout as error:
            raise TimeoutError(
                f"Timed out waiting for cache activation lock: {lock_path}"
            ) from error

    def _activate_locked(
        self,
        build_dir: Path,
        split: str,
        records: list[ShardRecord],
    ) -> Path:
        destination_manifest = manifest_path(self.cache_dir)
        old_manifest = (
            destination_manifest.read_bytes()
            if destination_manifest.is_file()
            else None
        )

        try:
            manifest = load_manifest(self.cache_dir)
        except FileNotFoundError:
            manifest = CacheManifest.empty()

        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=f".blip2_{split}_backup_",
                dir=self.cache_dir,
            )
        )
        moved_new: list[Path] = []

        old_names = {
            record.filename
            for record in manifest.splits.get(
                split,
                SplitRecord(sample_count=0, shards=()),
            ).shards
        }
        old_names.update(
            path.name
            for path in self.cache_dir.iterdir()
            if path.is_file()
            and (match := _LEGACY_SHARD_PATTERN.fullmatch(path.name))
            and match.group(1) == split
        )

        try:
            for name in sorted(old_names):
                old_path = self.cache_dir / name
                if old_path.is_file():
                    os.replace(old_path, backup_dir / name)

            for record in records:
                destination = self.cache_dir / record.filename
                os.replace(build_dir / record.filename, destination)
                moved_new.append(destination)

            split_record = SplitRecord(
                sample_count=sum(record.sample_count for record in records),
                shards=tuple(records),
            )
            updated_manifest = manifest.with_split(split, split_record)
            manifest_file = save_manifest_atomic(
                updated_manifest,
                self.cache_dir,
            )

            validation_store = CacheStore(
                self.cache_dir,
                verify_hashes=True,
                allow_legacy=False,
            )
            validation_store.index(split)
            for record in records:
                validation_store.load_shard(split, record)

            return manifest_file
        except Exception:
            for path in moved_new:
                path.unlink(missing_ok=True)
            for backup in backup_dir.iterdir():
                os.replace(backup, self.cache_dir / backup.name)

            if old_manifest is None:
                destination_manifest.unlink(missing_ok=True)
            else:
                rollback = Path(f"{destination_manifest}.rollback")
                rollback.write_bytes(old_manifest)
                os.replace(rollback, destination_manifest)
            raise
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)


def remove_build_directory(path: str | os.PathLike[str]) -> None:
    """Remove only a hidden v2 build directory created by :class:`CacheWriter`."""
    directory = Path(path).resolve()
    if not directory.name.startswith(".blip2_") or "_build_" not in directory.name:
        raise ValueError(f"Refusing to remove non-build directory: {directory}")
    shutil.rmtree(directory, ignore_errors=True)
