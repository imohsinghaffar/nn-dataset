"""Explicit, transactional BLIP-2/Q-Former feature-cache builder.

Example:
    python -m ab.nn.tools.build_blip2_cache_v2 \
        --split train --cache-dir out/cache --batch-size 32

This command never calls ``ab.nn.util.Loader.load_dataset``. It reads an
already-prepared COCO 2017 directory directly, so cache generation cannot
re-enter the cached transform or start an implicit dataset download.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ab.nn.transform.blip2_cache_contract_v2 import (
    FEATURE_MODEL_ID,
    FEATURE_SHAPE,
    CacheContractError,
    expected_split_size,
    normalize_split,
    resolve_cache_dir,
)
from ab.nn.transform.blip2_cache_store_v2 import (
    CacheIntegrityError,
    CacheMissingError,
    CacheStore,
    CacheWriter,
    remove_build_directory,
)
from ab.nn.transform.blip2_processor_v2 import Blip2ImageTransform
from ab.nn.util.Const import data_dir
from ab.nn.util.hf.download_utils import ensure_hf_model


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class CocoCaptionExtractionDataset(Dataset):
    """Network-free COCO 2017 reader used only for feature extraction."""

    def __init__(self, root: str | os.PathLike[str], split: str):
        from pycocotools.coco import COCO

        self.root = Path(root).expanduser().resolve()
        self.split = normalize_split(split)
        self.image_dir = self.root / f"{self.split}2017"
        self.annotation_file = (
            self.root
            / "annotations"
            / f"captions_{self.split}2017.json"
        )

        missing = [
            path
            for path in (self.image_dir, self.annotation_file)
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "COCO 2017 is incomplete. Prepare the dataset before cache "
                f"generation; missing:\n{formatted}"
            )

        self.coco = COCO(str(self.annotation_file))
        self.ids = sorted(self.coco.imgs.keys())
        if not self.ids:
            raise RuntimeError(
                f"COCO annotation contains no {self.split!r} images: "
                f"{self.annotation_file}"
            )

        expected = expected_split_size(self.split)
        if expected > 0 and len(self.ids) != expected:
            raise RuntimeError(
                f"COCO {self.split!r} contains {len(self.ids)} images; "
                f"expected {expected}."
            )

        self.image_transform = Blip2ImageTransform()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"COCO image is missing; builder will not download it: "
                f"{image_path}"
            )

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB").copy()
        except Exception as error:
            raise RuntimeError(f"Could not decode COCO image: {image_path}") from error

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        captions = [
            annotation["caption"].strip()
            for annotation in annotations
            if isinstance(annotation.get("caption"), str)
            and annotation["caption"].strip()
        ]
        if not captions:
            raise RuntimeError(f"COCO image {image_id} has no valid captions.")

        return self.image_transform(image), captions


def extraction_collate(batch):
    if not batch:
        raise ValueError("Cannot collate an empty extraction batch.")
    images = torch.stack([sample[0] for sample in batch], dim=0)
    labels = [sample[1] for sample in batch]
    return images, labels


def extract_qformer_tensor(raw_output: Any) -> torch.Tensor:
    if torch.is_tensor(raw_output):
        features = raw_output
    elif hasattr(raw_output, "last_hidden_state"):
        features = raw_output.last_hidden_state
    elif isinstance(raw_output, (list, tuple)) and raw_output:
        features = raw_output[0]
    else:
        raise RuntimeError(
            "Unsupported BLIP-2 Q-Former output type: "
            f"{type(raw_output).__name__}."
        )

    if not torch.is_tensor(features):
        raise RuntimeError("BLIP-2 Q-Former output is not a tensor.")
    if features.ndim != 3 or tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise RuntimeError(
            "Expected Q-Former output shape "
            f"(B,{FEATURE_SHAPE[0]},{FEATURE_SHAPE[1]}), received "
            f"{tuple(features.shape)}."
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("BLIP-2 produced NaN or Inf Q-Former features.")
    return features


def _device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(normalized)


def _positive(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def build_cache(
    *,
    split: str,
    cache_dir: str | os.PathLike[str] | None,
    coco_dir: str | os.PathLike[str],
    batch_size: int,
    shard_samples: int,
    num_workers: int,
    device_name: str,
    force: bool,
) -> Path:
    normalized_split = normalize_split(split)
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    batch_size = _positive(batch_size, "batch_size")
    shard_samples = _positive(shard_samples, "shard_samples")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    if not force:
        try:
            existing = CacheStore(resolved_cache_dir).index(normalized_split)
        except (CacheMissingError, CacheIntegrityError, CacheContractError):
            pass
        else:
            print(
                f"[CACHE-BUILD] Valid {normalized_split!r} cache already "
                f"exists ({existing.full_length} samples). Use --force to "
                "replace it."
            )
            return resolved_cache_dir

    device = _device(device_name)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    dataset = CocoCaptionExtractionDataset(coco_dir, normalized_split)
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": extraction_collate,
    }
    if num_workers > 0:
        # Spawned workers never inherit an initialized CUDA context or the
        # large BLIP-2 model from the parent process.
        loader_options.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    loader = DataLoader(**loader_options)

    local_model_path = ensure_hf_model(FEATURE_MODEL_ID)

    from transformers import Blip2Model

    print(
        f"[CACHE-BUILD] Loading {FEATURE_MODEL_ID} from {local_model_path} "
        f"on {device}."
    )
    model = Blip2Model.from_pretrained(
        local_model_path,
        local_files_only=True,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()

    writer = CacheWriter(resolved_cache_dir)
    build_dir = writer.create_build_dir(normalized_split)
    records = []
    buffered_features: list[torch.Tensor] = []
    buffered_labels: list[Any] = []
    buffered_count = 0
    shard_index = 0
    total_samples = 0

    def flush_buffer() -> None:
        nonlocal buffered_features
        nonlocal buffered_labels
        nonlocal buffered_count
        nonlocal shard_index

        if not buffered_features:
            return
        features = torch.cat(buffered_features, dim=0)
        record = writer.write_shard(
            build_dir,
            normalized_split,
            shard_index,
            features,
            buffered_labels,
        )
        records.append(record)
        print(
            f"[CACHE-BUILD] Wrote {record.filename}: "
            f"{record.sample_count} samples."
        )
        shard_index += 1
        buffered_features = []
        buffered_labels = []
        buffered_count = 0

    try:
        with torch.inference_mode():
            for images, labels in tqdm(
                loader,
                desc=f"Extracting BLIP-2 {normalized_split}",
            ):
                images = images.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=True,
                )
                raw_output = model.get_qformer_features(pixel_values=images)
                features = extract_qformer_tensor(raw_output)
                cpu_features = features.detach().cpu().to(torch.float16)

                buffered_features.append(cpu_features)
                buffered_labels.extend(labels)
                batch_count = int(cpu_features.shape[0])
                buffered_count += batch_count
                total_samples += batch_count

                if buffered_count >= shard_samples:
                    flush_buffer()

        flush_buffer()

        expected = expected_split_size(normalized_split)
        if expected > 0 and total_samples != expected:
            raise RuntimeError(
                f"Extraction produced {total_samples} samples; expected "
                f"{expected}. Existing cache was not modified."
            )

        manifest = writer.activate_split(
            build_dir,
            normalized_split,
            records,
        )
        print(
            f"[CACHE-BUILD] Activated {normalized_split!r}: "
            f"{total_samples} samples across {len(records)} shards."
        )
        return manifest
    finally:
        remove_build_directory(build_dir)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transactional BLIP-2 cached features for COCO 2017."
    )
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--coco-dir",
        default=str(Path(data_dir) / "coco"),
        help="Prepared COCO 2017 root containing annotations/ and split images.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-samples", type=int, default=16_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_cache(
        split=args.split,
        cache_dir=args.cache_dir,
        coco_dir=args.coco_dir,
        batch_size=args.batch_size,
        shard_samples=args.shard_samples,
        num_workers=args.num_workers,
        device_name=args.device,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
