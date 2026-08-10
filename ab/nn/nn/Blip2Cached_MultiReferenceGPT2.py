"""Portable cached BLIP-2 features with frozen GPT-2 and multi-reference loss."""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from ab.nn.captioning.blip2.contract import FEATURE_SHAPE
from ab.nn.captioning.blip2.environment import validate_environment
from ab.nn.captioning.blip2.gpt2 import GPT2_VOCAB_SIZE, gpt2_runtime_paths

validate_environment()

from transformers import AutoModelForCausalLM, GPT2TokenizerFast


def supported_hyperparameters():
    return {"lr", "batch"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = torch.device(device)
        self.prm = dict(prm or {})
        seed = int(self.prm.get("seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._reference_generator = torch.Generator(device="cpu")
        self._reference_generator.manual_seed(seed)

        if in_shape and tuple(in_shape[-2:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected cached input ending in {FEATURE_SHAPE}, got {in_shape}.")
        if out_shape and int(out_shape[0]) != GPT2_VOCAB_SIZE:
            raise ValueError(f"This model requires GPT-2 token IDs ({GPT2_VOCAB_SIZE}).")
        if self.device.type != "cuda" and not bool(self.prm.get("allow_cpu", False)):
            raise RuntimeError("Blip2Cached_MultiReferenceGPT2 requires CUDA unless allow_cpu=true.")
        if self.device.type == "cuda":
            total_gib = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            if total_gib < 8:
                raise RuntimeError(f"At least 8 GiB VRAM is required; found {total_gib:.1f} GiB.")

        decoder_path, tokenizer_path = gpt2_runtime_paths(self.prm.get("cache_dir"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2 = AutoModelForCausalLM.from_pretrained(
            str(decoder_path), dtype=self.dtype, low_cpu_mem_usage=True, local_files_only=True
        ).to(self.device)
        if int(self.gpt2.config.vocab_size) != GPT2_VOCAB_SIZE:
            raise RuntimeError("Portable GPT-2 decoder has an incompatible vocabulary.")
        self.gpt2.requires_grad_(False)
        self.gpt2.eval()
        hidden = int(self.gpt2.get_input_embeddings().embedding_dim)
        self.projection = nn.Linear(FEATURE_SHAPE[1], hidden).to(self.device, dtype=torch.float32)
        self.optimizer = None
        self.max_text_length = int(self.prm.get("max_text_length", 50))
        self.max_new_tokens = int(self.prm.get("max_new_tokens", 24))
        self.num_beams = int(self.prm.get("num_beams", 3))
        self.prompt = str(self.prm.get("caption_prompt", "a photo of "))

    def train(self, mode=True):
        super().train(mode)
        self.gpt2.eval()
        return self

    def _features(self, value):
        if value.ndim != 3 or tuple(value.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected (B,{FEATURE_SHAPE[0]},{FEATURE_SHAPE[1]}) features.")
        if not torch.isfinite(value).all():
            raise ValueError("Cached features contain NaN or Inf.")
        return value.to(self.device, dtype=torch.float32, non_blocking=True)

    def _select_reference_labels(self, labels):
        if labels.ndim != 3:
            return labels
        valid = (labels >= 0).any(dim=2)
        if not valid.any(dim=1).all():
            raise ValueError("At least one sample has no caption reference.")
        if self.training:
            chosen = []
            for row in valid.detach().cpu():
                options = torch.nonzero(row, as_tuple=False).flatten()
                chosen.append(int(options[torch.randint(len(options), (1,), generator=self._reference_generator).item()]))
            indices = torch.tensor(chosen, device=labels.device)
        else:
            indices = valid.long().argmax(dim=1)
        return labels[torch.arange(labels.shape[0], device=labels.device), indices]

    @staticmethod
    def _caption_targets(ids, attention_mask, prompt_length):
        targets = ids.clone()
        targets[attention_mask == 0] = -100
        targets[:, : min(int(prompt_length), targets.shape[1])] = -100
        return targets

    def _reference_text(self, labels):
        selected = self._select_reference_labels(labels)
        clean = selected.detach().cpu().clone()
        clean[clean < 0] = self.gpt2_tokenizer.eos_token_id
        return self.gpt2_tokenizer.batch_decode(clean, skip_special_tokens=True)

    def _loss(self, features, labels):
        eos = self.gpt2_tokenizer.eos_token
        texts = [self.prompt + text.strip() + eos for text in self._reference_text(labels)]
        encoded = self.gpt2_tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_text_length,
            add_special_tokens=False, return_tensors="pt",
        )
        prompt = self.gpt2_tokenizer(
            self.prompt, add_special_tokens=False, return_tensors="pt"
        )
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        targets = self._caption_targets(ids, mask, prompt.attention_mask[0].sum().item())
        visual = self.projection(features).to(self.dtype)
        text = self.gpt2.get_input_embeddings()(ids).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        prefix_labels = torch.full(visual.shape[:2], -100, dtype=torch.long, device=self.device)
        prefix_mask = torch.ones(visual.shape[:2], dtype=mask.dtype, device=self.device)
        return self.gpt2(
            inputs_embeds=embeddings,
            attention_mask=torch.cat((prefix_mask, mask), dim=1),
            labels=torch.cat((prefix_labels, targets), dim=1),
            use_cache=False,
        ).loss

    def _generate(self, features):
        visual = self.projection(features).to(self.dtype)
        prompt = self.gpt2_tokenizer(
            self.prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device).expand(features.shape[0], -1)
        text = self.gpt2.get_input_embeddings()(prompt).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.gpt2.generate(
                inputs_embeds=embeddings,
                attention_mask=mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                eos_token_id=self.gpt2_tokenizer.eos_token_id,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
            )

    def forward(self, features, captions=None):
        features = self._features(features)
        return self._loss(features, captions) if captions is not None else self._generate(features)

    def train_setup(self, prm):
        self.prm.update(prm or {})
        learning_rate = float(self.prm.get("lr", 1e-5))
        if not 0 < learning_rate <= 1:
            raise ValueError("lr must be in (0, 1].")
        self.optimizer = torch.optim.AdamW(self.projection.parameters(), lr=learning_rate, weight_decay=0.01)

    def learn(self, train_data):
        if self.optimizer is None:
            raise RuntimeError("train_setup() must be called before learn().")
        total, batches = 0.0, 0
        for features, labels in train_data:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self(features, labels.to(self.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite BLIP-2/GPT-2 loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.projection.parameters(), 1.0)
            self.optimizer.step()
            total += float(loss.detach())
            batches += 1
        return 0.0, total / max(batches, 1)
