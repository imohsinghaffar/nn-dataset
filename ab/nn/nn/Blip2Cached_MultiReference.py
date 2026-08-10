"""Cached BLIP-2 variant with diverse references and caption-only loss.

This is intentionally a separate model identity.  It keeps the verified
portable cache, pretrained projection, frozen OPT decoder, optimizer, and
generation behavior from :mod:`Blip2Cached`.  During training only, it samples
one valid COCO reference independently for every image and excludes the fixed
caption prompt from the language-model targets.
"""

from __future__ import annotations

import torch

from ab.nn.nn.Blip2Cached import Net as BaselineNet
from ab.nn.nn.Blip2Cached import supported_hyperparameters


class Net(BaselineNet):
    def __init__(self, in_shape, out_shape, prm, device):
        options = dict(prm or {})
        # OPT is frozen, but gradients still pass through it to the projection.
        # Recomputing the frozen decoder is slower and was unnecessary on the
        # tested 24 GiB RTX 3090.  Users can still opt in explicitly.
        options.setdefault("gradient_checkpointing", False)
        super().__init__(in_shape, out_shape, options, device)
        self._reference_generator = torch.Generator(device="cpu")
        self._reference_generator.manual_seed(int(options.get("seed", 42)))

    def _select_reference_labels(self, labels):
        if labels.ndim != 3:
            return labels
        valid = (labels >= 0).any(dim=2)
        if not valid.any(dim=1).all():
            raise ValueError("At least one sample has no caption reference.")
        if self.training:
            selected = []
            for row in valid.detach().cpu():
                choices = torch.nonzero(row, as_tuple=False).flatten()
                position = torch.randint(
                    len(choices),
                    (1,),
                    generator=self._reference_generator,
                ).item()
                selected.append(int(choices[position]))
            indices = torch.tensor(selected, device=labels.device)
        else:
            indices = valid.long().argmax(dim=1)
        rows = torch.arange(labels.shape[0], device=labels.device)
        return labels[rows, indices]

    def _reference_text(self, labels):
        labels = self._select_reference_labels(labels)
        clean = labels.detach().cpu().clone()
        clean[clean < 0] = self.opt_tokenizer.pad_token_id
        return self.opt_tokenizer.batch_decode(clean, skip_special_tokens=True)

    @staticmethod
    def _caption_targets(ids, attention_mask, prompt_length):
        targets = ids.clone()
        targets[attention_mask == 0] = -100
        targets[:, : min(int(prompt_length), targets.shape[1])] = -100
        return targets

    def _loss(self, features, labels):
        texts = [self.prompt + text.strip() for text in self._reference_text(labels)]
        encoded = self.opt_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        prompt = self.opt_tokenizer(
            self.prompt,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        prompt_length = int(prompt.attention_mask[0].sum().item())
        targets = self._caption_targets(ids, mask, prompt_length)

        visual = self.projection(features).to(self.dtype)
        text = self.opt.get_input_embeddings()(ids).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        prefix_labels = torch.full(
            visual.shape[:2], -100, dtype=torch.long, device=self.device
        )
        joined_labels = torch.cat((prefix_labels, targets), dim=1)
        prefix_mask = torch.ones(
            visual.shape[:2], dtype=mask.dtype, device=self.device
        )
        return self.opt(
            inputs_embeds=embeddings,
            attention_mask=torch.cat((prefix_mask, mask), dim=1),
            labels=joined_labels,
            use_cache=False,
        ).loss


__all__ = ["Net", "supported_hyperparameters"]
