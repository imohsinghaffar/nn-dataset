"""BLIP-2 Image Processor V2 (Resize -> Normalize -> Pixel Values).
Responsible ONLY for raw image preprocessing for cache builder.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import Blip2Processor
from ab.nn.util.hf.download_utils import ensure_hf_model

MODEL_ID = "Salesforce/blip2-opt-2.7b"

class Blip2ImageTransform:
    """
    Applies Salesforce BLIP-2 image preprocessing (224x224, float32 pixel values).
    """
    def __init__(self):
        local_path = ensure_hf_model(MODEL_ID)
        self.processor = Blip2Processor.from_pretrained(
            local_path,
            local_files_only=True,
            use_fast=False,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values.squeeze(0)

def transform(prm=None):
    return Blip2ImageTransform()
