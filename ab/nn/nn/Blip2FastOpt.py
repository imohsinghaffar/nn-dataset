"""BLIP-2 cached-feature captioner with a frozen OPT-2.7B decoder.

This module intentionally owns only model behaviour. Feature-cache discovery,
generation, validation, and dataset loading belong to the matching cached
transform. Keeping that boundary strict prevents recursive dataset loading and
makes failures reproducible across local, CI, and fresh-machine environments.

NN-Dataset contract
-------------------
* main class: ``Net(nn.Module)``
* constructor: ``Net(in_shape, out_shape, prm, device)``
* methods: ``train_setup(prm)`` and ``learn(train_data)``
* module function: ``supported_hyperparameters()``

Input contract
--------------
* features: float tensor ``(batch, 32, 768)``
* captions: GPT-2 token IDs ``(batch, sequence)`` or
  ``(batch, references, sequence)``; padding is ``-100``

The full BLIP-2 vision model is never loaded for training or inference. It is
loaded only when the pretrained language-projection checkpoint has to be
extracted once. Random projection fallback is deliberately forbidden.
"""

from __future__ import annotations

import gc
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from filelock import FileLock, Timeout


# These must be set before Transformers or Hugging Face modules are imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    GPT2Tokenizer,
)

from ab.nn.util.hf.download_utils import ensure_hf_model  # noqa: E402


MODEL_NAME = "Blip2FastOpt_v2"
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
OPT_MODEL_ID = "facebook/opt-2.7b"
GPT2_MODEL_ID = "gpt2"

NUM_VISUAL_TOKENS = 32
QFORMER_HIDDEN_SIZE = 768
GPT2_VOCAB_SIZE = 50_257

DEFAULT_TRAIN_MAX_LENGTH = 50
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_NUM_BEAMS = 3
DEFAULT_PROMPT = "a photo of "

DEFAULT_PROJECTION_PATH = (
    Path.home()
    / ".cache"
    / "nn-dataset"
    / "blip2"
    / "pretrained_projection_opt27b.pth"
)


def supported_hyperparameters() -> set[str]:
    """Hyperparameters accepted by the generic NN-Dataset trainer."""
    return {"lr", "batch"}


def _log(message: str) -> None:
    print(f"[{MODEL_NAME}] {message}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _positive_int(prm: dict[str, Any], key: str, default: int) -> int:
    value = int(prm.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero, received {value}.")
    return value


def _positive_float(prm: dict[str, Any], key: str, default: float) -> float:
    value = float(prm.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero, received {value}.")
    return value


def _model_dtype(device: torch.device) -> torch.dtype:
    """Use half precision only where it is reliably supported."""
    return torch.float16 if device.type == "cuda" else torch.float32


def _resolve_projection_path(prm: dict[str, Any]) -> Path:
    configured = (
        prm.get("projection_path")
        or os.environ.get("BLIP2_PROJECTION_PATH")
        or DEFAULT_PROJECTION_PATH
    )
    return Path(configured).expanduser().resolve()


class CachedFeatureEncoder(nn.Module):
    """Validate cached Q-Former features and move them to the model device."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def train(self, mode: bool = True) -> "CachedFeatureEncoder":
        del mode
        super().train(False)
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(
                "Cached BLIP-2 features must be a torch.Tensor, received "
                f"{type(features).__name__}."
            )

        expected_tail = (NUM_VISUAL_TOKENS, QFORMER_HIDDEN_SIZE)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected_tail:
            raise RuntimeError(
                "Expected cached BLIP-2/Q-Former features with shape "
                f"(B, {NUM_VISUAL_TOKENS}, {QFORMER_HIDDEN_SIZE}), received "
                f"{tuple(features.shape)}. Use the matching cached transform."
            )

        if not features.is_floating_point():
            raise TypeError(
                "Cached BLIP-2 features must use a floating-point dtype, "
                f"received {features.dtype}."
            )

        if not torch.isfinite(features).all():
            raise ValueError("Cached BLIP-2 features contain NaN or Inf.")

        return features.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )


class ProjectionCheckpoint:
    """Load or atomically extract BLIP-2's OPT language projection."""

    def __init__(
        self,
        path: Path,
        output_size: int,
        lock_timeout_seconds: int,
    ):
        self.path = path
        self.output_size = int(output_size)
        self.lock_timeout_seconds = int(lock_timeout_seconds)

        if self.lock_timeout_seconds <= 0:
            raise ValueError("projection_lock_timeout must be greater than zero.")

    @property
    def expected_weight_shape(self) -> tuple[int, int]:
        return self.output_size, QFORMER_HIDDEN_SIZE

    @property
    def expected_bias_shape(self) -> tuple[int]:
        return (self.output_size,)

    def _validate(self, state: object) -> dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            raise TypeError("Projection checkpoint must contain a state dict.")

        missing = {"weight", "bias"}.difference(state)
        unexpected = set(state).difference({"weight", "bias"})

        if missing:
            raise KeyError(f"Projection checkpoint is missing {sorted(missing)}.")
        if unexpected:
            raise KeyError(
                f"Projection checkpoint has unexpected keys {sorted(unexpected)}."
            )

        weight = state["weight"]
        bias = state["bias"]

        if not torch.is_tensor(weight) or not torch.is_tensor(bias):
            raise TypeError("Projection weight and bias must be tensors.")

        if tuple(weight.shape) != self.expected_weight_shape:
            raise ValueError(
                "Projection weight shape mismatch: expected "
                f"{self.expected_weight_shape}, received {tuple(weight.shape)}."
            )

        if tuple(bias.shape) != self.expected_bias_shape:
            raise ValueError(
                "Projection bias shape mismatch: expected "
                f"{self.expected_bias_shape}, received {tuple(bias.shape)}."
            )

        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Projection checkpoint contains NaN or Inf.")

        return {
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
        }

    def _read(self, path: Path) -> dict[str, torch.Tensor]:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError(
                f"Projection checkpoint could not be loaded: {path}"
            ) from error
        return self._validate(state)

    def ensure(self) -> dict[str, torch.Tensor]:
        if self.path.is_file():
            return self._read(self.path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{self.path}.lock")

        try:
            with FileLock(
                str(lock_path),
                timeout=self.lock_timeout_seconds,
            ):
                if self.path.is_file():
                    return self._read(self.path)

                self._extract_atomically()
                return self._read(self.path)
        except Timeout as error:
            raise TimeoutError(
                "Timed out waiting for BLIP-2 projection lock: "
                f"{lock_path}"
            ) from error

    def _extract_atomically(self) -> None:
        local_blip2_path = ensure_hf_model(BLIP2_MODEL_ID)
        temporary_path = Path(f"{self.path}.temporary.{os.getpid()}")
        temporary_model = None

        try:
            if temporary_path.exists():
                temporary_path.unlink()

            _log(
                "Projection checkpoint missing; extracting the pretrained "
                "BLIP-2 language projection once on CPU."
            )

            temporary_model = Blip2ForConditionalGeneration.from_pretrained(
                local_blip2_path,
                local_files_only=True,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )

            projection = getattr(temporary_model, "language_projection", None)
            if projection is None:
                raise RuntimeError(
                    "Salesforce BLIP-2 has no language_projection module."
                )

            state = self._validate(
                {
                    key: value.detach().cpu()
                    for key, value in projection.state_dict().items()
                }
            )

            torch.save(state, temporary_path)
            self._read(temporary_path)
            os.replace(temporary_path, self.path)
            _log(f"Projection checkpoint activated: {self.path}")
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Pretrained BLIP-2 projection extraction failed. Random "
                "projection fallback is disabled to protect reproducibility."
            ) from error
        finally:
            if temporary_model is not None:
                del temporary_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class CaptionCodec:
    """Translate between cached GPT-2 IDs, text, and OPT token IDs."""

    def __init__(
        self,
        opt_tokenizer: Any,
        gpt2_tokenizer: Any,
        device: torch.device,
        prompt: str,
        train_max_length: int,
    ):
        self.opt_tokenizer = opt_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.device = device
        self.prompt_text = " ".join(prompt.replace("\n", " ").split())
        self.train_max_length = train_max_length

        if not self.prompt_text:
            raise ValueError("caption_prompt must not be empty.")

    @staticmethod
    def _clean_gpt2_sequence(sequence: Iterable[int]) -> list[int]:
        cleaned: list[int] = []

        for raw_token in sequence:
            token = int(raw_token)
            if token == -100:
                continue
            if token < 0 or token >= GPT2_VOCAB_SIZE:
                raise ValueError(f"Invalid GPT-2 token ID: {token}.")
            cleaned.append(token)

        return cleaned

    def gpt2_ids_to_text(self, token_ids: torch.Tensor) -> list[str]:
        if token_ids.ndim != 2:
            raise RuntimeError(
                "Expected selected captions with shape (B, T), received "
                f"{tuple(token_ids.shape)}."
            )

        clean_ids = token_ids.clone()
        clean_ids[clean_ids < 0] = self.gpt2_tokenizer.eos_token_id
        clean_ids[clean_ids >= GPT2_VOCAB_SIZE] = self.gpt2_tokenizer.eos_token_id

        decoded = self.gpt2_tokenizer.batch_decode(clean_ids, skip_special_tokens=True)
        texts = [" ".join(str(t).replace("\n", " ").split()).strip() or "image" for t in decoded]
        return texts

    def text_to_opt_batch(
        self,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bos = self.opt_tokenizer.bos_token or "</s>"
        eos = self.opt_tokenizer.eos_token or "</s>"
        prompt = f"{bos}{self.prompt_text}"
        full_texts = [f"{prompt}{text}{eos}" for text in texts]

        encoded = self.opt_tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.train_max_length,
            add_special_tokens=False,
        )

        input_ids = encoded.input_ids.to(self.device, non_blocking=True)
        attention_mask = encoded.attention_mask.to(
            self.device,
            non_blocking=True,
        )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        prompt_ids = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        labels[:, : int(prompt_ids.numel())] = -100

        if not (labels != -100).any(dim=1).all():
            raise RuntimeError(
                "At least one OPT training sample has no caption tokens after "
                "prompt masking and truncation."
            )

        return input_ids, labels, attention_mask

    def opt_ids_to_gpt2_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        decoded = self.opt_tokenizer.batch_decode(
            generated_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        cleaned = [self._clean_generated_text(text) for text in decoded]

        encoded = self.gpt2_tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.train_max_length,
            add_special_tokens=False,
        )
        return encoded.input_ids.to(self.device, non_blocking=True)

    def _clean_generated_text(self, text: str) -> str:
        cleaned = " ".join(str(text).replace("\n", " ").split())
        lowered = cleaned.lower()

        prompt = self.prompt_text.lower()
        if lowered.startswith(prompt):
            cleaned = cleaned[len(self.prompt_text) :].strip()

        words = cleaned.split()
        if len(words) > 18:
            cleaned = " ".join(words[:18])

        return cleaned or "image"


class FrozenOPTDecoder(nn.Module):
    """Frozen OPT language model plus the only trainable projection layer."""

    def __init__(self, device: torch.device, prm: dict[str, Any]):
        super().__init__()
        self.device = device
        self.prm = prm
        self.dtype = _model_dtype(device)

        opt_path = ensure_hf_model(OPT_MODEL_ID)
        gpt2_path = ensure_hf_model(GPT2_MODEL_ID)

        self.opt_tokenizer = AutoTokenizer.from_pretrained(
            opt_path,
            use_fast=False,
            local_files_only=True,
        )
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token
        if self.opt_tokenizer.pad_token_id is None:
            raise RuntimeError("OPT tokenizer has no pad_token_id.")

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            gpt2_path,
            local_files_only=True,
        )
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        if self.gpt2_tokenizer.pad_token_id is None:
            raise RuntimeError("GPT-2 tokenizer has no pad_token_id.")

        _log(f"Loading frozen decoder {OPT_MODEL_ID} as {self.dtype}.")
        self.opt = AutoModelForCausalLM.from_pretrained(
            opt_path,
            local_files_only=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        self.opt.requires_grad_(False)
        self.opt.eval()

        self.embedding_size = int(
            self.opt.get_input_embeddings().embedding_dim
        )
        self.visual_projection = nn.Linear(
            QFORMER_HIDDEN_SIZE,
            self.embedding_size,
        ).to(device=device, dtype=torch.float32)

        projection_path = _resolve_projection_path(prm)
        projection_lock_timeout = _positive_int(
            prm,
            "projection_lock_timeout",
            int(os.environ.get("BLIP2_PROJECTION_LOCK_TIMEOUT", "7200")),
        )
        checkpoint = ProjectionCheckpoint(
            path=projection_path,
            output_size=self.embedding_size,
            lock_timeout_seconds=projection_lock_timeout,
        )
        self.visual_projection.load_state_dict(checkpoint.ensure(), strict=True)
        _log(f"Validated projection loaded from {projection_path}.")

        self.codec = CaptionCodec(
            opt_tokenizer=self.opt_tokenizer,
            gpt2_tokenizer=self.gpt2_tokenizer,
            device=device,
            prompt=str(prm.get("caption_prompt", DEFAULT_PROMPT)),
            train_max_length=_positive_int(
                prm,
                "train_max_length",
                DEFAULT_TRAIN_MAX_LENGTH,
            ),
        )
        self.max_new_tokens = _positive_int(
            prm,
            "max_new_tokens",
            DEFAULT_MAX_NEW_TOKENS,
        )
        self.num_beams = _positive_int(
            prm,
            "num_beams",
            DEFAULT_NUM_BEAMS,
        )

    def train(self, mode: bool = True) -> "FrozenOPTDecoder":
        super().train(mode)
        self.opt.eval()
        return self

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.visual_projection(features.float())
        return projected.to(dtype=self.dtype)

    def training_loss(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(visual_features.shape[0])
        visual_embeddings = self._project(visual_features)
        text_embeddings = self.opt.get_input_embeddings()(input_ids).to(
            dtype=self.dtype
        )
        inputs_embeds = torch.cat(
            [visual_embeddings, text_embeddings],
            dim=1,
        )

        visual_length = int(visual_embeddings.shape[1])
        visual_labels = torch.full(
            (batch_size, visual_length),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        joined_labels = torch.cat([visual_labels, labels], dim=1)

        visual_mask = torch.ones(
            (batch_size, visual_length),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.cat(
            [visual_mask, text_attention_mask],
            dim=1,
        )

        output = self.opt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=joined_labels,
            use_cache=False,
            return_dict=True,
        )
        loss = output.loss
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise RuntimeError("OPT did not return a scalar captioning loss.")
        return loss

    def generate(self, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size = int(visual_features.shape[0])
        visual_embeddings = self._project(visual_features)

        bos = self.opt_tokenizer.bos_token or "</s>"
        prompt_ids = self.opt_tokenizer(
            f"{bos}{self.codec.prompt_text}",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        prompt_ids = prompt_ids.expand(batch_size, -1)
        prompt_embeddings = self.opt.get_input_embeddings()(prompt_ids).to(
            dtype=self.dtype
        )

        inputs_embeds = torch.cat(
            [visual_embeddings, prompt_embeddings],
            dim=1,
        )
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            return self.opt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=2,
                repetition_penalty=1.25,
                length_penalty=0.35,
                early_stopping=(self.num_beams > 1),
                eos_token_id=self.opt_tokenizer.eos_token_id,
                pad_token_id=self.opt_tokenizer.pad_token_id,
                bos_token_id=self.opt_tokenizer.bos_token_id,
            )


class Net(nn.Module):
    """NN-Dataset model wrapper."""

    def __init__(
        self,
        in_shape: tuple,
        out_shape: tuple,
        prm: dict,
        device: torch.device,
    ):
        super().__init__()
        self.in_shape = tuple(in_shape) if in_shape else ()
        self.device = torch.device(device)
        self.prm: dict[str, Any] = dict(prm) if isinstance(prm, dict) else {}
        self.optimizer: torch.optim.Optimizer | None = None

        _seed_everything(int(self.prm.get("seed", 42)))
        self._validate_shapes(out_shape)

        self.encoder = CachedFeatureEncoder(self.device)
        self.decoder = FrozenOPTDecoder(self.device, self.prm)
        self._print_parameter_statistics()

    def _validate_shapes(self, out_shape: tuple) -> None:
        if self.in_shape and tuple(self.in_shape[-2:]) != (
            NUM_VISUAL_TOKENS,
            QFORMER_HIDDEN_SIZE,
        ):
            raise ValueError(
                "Blip2FastOpt_v2 requires cached feature in_shape ending in "
                f"({NUM_VISUAL_TOKENS}, {QFORMER_HIDDEN_SIZE}); received "
                f"{self.in_shape}."
            )

        vocab_size = int(out_shape[0]) if out_shape else GPT2_VOCAB_SIZE
        if vocab_size != GPT2_VOCAB_SIZE:
            raise ValueError(
                f"Expected GPT-2 vocabulary size {GPT2_VOCAB_SIZE}, received "
                f"{vocab_size}. Use the matching cached transform."
            )

    def _print_parameter_statistics(self) -> None:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        percentage = 100.0 * trainable / max(total, 1)
        _log(
            f"parameters total={total:,}, trainable={trainable:,} "
            f"({percentage:.6f}%)."
        )

    @staticmethod
    def _select_caption(captions: torch.Tensor) -> torch.Tensor:
        """Select the first valid reference for each sample deterministically."""
        if captions.ndim == 2:
            if not (captions != -100).any(dim=1).all():
                raise ValueError("At least one caption has no valid tokens.")
            return captions

        if captions.ndim != 3:
            raise RuntimeError(
                "Expected captions with shape (B,T) or (B,R,T), received "
                f"{tuple(captions.shape)}."
            )

        valid_references = (captions != -100).any(dim=2)
        if not valid_references.any(dim=1).all():
            raise ValueError("At least one sample has no valid caption reference.")

        selected_indices = valid_references.to(torch.int64).argmax(dim=1)
        batch_indices = torch.arange(captions.shape[0], device=captions.device)
        return captions[batch_indices, selected_indices]

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_features = self.encoder(features)

        if captions is not None:
            selected = self._select_caption(captions.long().to(self.device))
            texts = self.decoder.codec.gpt2_ids_to_text(selected)
            input_ids, labels, attention_mask = (
                self.decoder.codec.text_to_opt_batch(texts)
            )
            return self.decoder.training_loss(
                visual_features,
                input_ids,
                labels,
                attention_mask,
            )

        generated_opt_ids = self.decoder.generate(visual_features)
        return self.decoder.codec.opt_ids_to_gpt2_ids(generated_opt_ids)

    def train_setup(self, prm: dict) -> None:
        if isinstance(prm, dict):
            self.prm.update(prm)

        if any(parameter.requires_grad for parameter in self.decoder.opt.parameters()):
            raise RuntimeError("Frozen OPT parameters unexpectedly became trainable.")

        trainable_parameters = [
            parameter
            for parameter in self.decoder.visual_projection.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable projection parameters were found.")

        learning_rate = _positive_float(self.prm, "lr", 1e-4)
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=float(self.prm.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        with torch.no_grad():
            contract_input = torch.zeros(
                2,
                NUM_VISUAL_TOKENS,
                QFORMER_HIDDEN_SIZE,
                device=self.device,
            )
            projected = self.decoder.visual_projection(contract_input)
            expected = (
                2,
                NUM_VISUAL_TOKENS,
                self.decoder.embedding_size,
            )
            if tuple(projected.shape) != expected:
                raise RuntimeError(
                    f"Projection contract failed: expected {expected}, "
                    f"received {tuple(projected.shape)}."
                )

        _log(f"optimizer=AdamW, projection-only, lr={learning_rate:.2e}.")

    def learn(self, train_data):
        if self.optimizer is None:
            raise RuntimeError("train_setup() must be called before learn().")

        self.encoder.eval()
        self.decoder.train()
        self.decoder.opt.eval()

        total_loss = 0.0
        completed_batches = 0

        try:
            for features, captions in train_data:
                if isinstance(features, list):
                    features = torch.stack(features)
                if isinstance(captions, list):
                    captions = torch.stack(captions)

                features = features.to(self.device, non_blocking=True)
                captions = captions.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.forward(features, captions)

                if not torch.is_tensor(loss) or loss.numel() != 1:
                    raise RuntimeError(
                        "Expected scalar training loss, received "
                        f"{type(loss).__name__} with shape "
                        f"{getattr(loss, 'shape', None)}."
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss: {loss.detach().item()}"
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.visual_projection.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

                total_loss += float(loss.detach().item())
                completed_batches += 1

        except Exception as error:
            from ab.nn.util.Exception import LearnTimeException

            if not isinstance(error, LearnTimeException):
                raise
            _log(
                "epoch time limit reached after "
                f"{completed_batches} completed batches."
            )

        average_loss = total_loss / max(completed_batches, 1)
        _log(
            f"epoch complete: batches={completed_batches}, "
            f"average_loss={average_loss:.4f}."
        )
        return 0.0, average_loss
