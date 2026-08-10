"""Compare direct and cached BLIP-2 captions on fixed COCO samples."""

from __future__ import annotations

import argparse
import gc
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from ab.nn.captioning.blip2.environment import validate_environment

validate_environment()

from transformers import AutoProcessor, Blip2ForConditionalGeneration

from ab.nn.captioning.blip2.cache import CachedCaptionDataset
from ab.nn.captioning.blip2.contract import (
    MODEL_ID,
    MODEL_REVISION,
    OPT_VOCAB_SIZE,
    atomic_json,
)
from ab.nn.nn.Blip2Cached import Net
from ab.nn.tools.build_blip2_cached import CocoImages


def _clean(text: str, prompt: str) -> str:
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    value = next((line for line in lines if line), "")
    if value.lower().startswith(prompt.strip().lower()):
        value = value[len(prompt.strip()):].strip()
    return value or "image"


def _direct_captions(images, device, prompt, max_new_tokens, num_beams):
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=False,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        image_embeds = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        ).last_hidden_state
        image_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=device,
        )
        query_tokens = model.query_tokens.expand(len(images), -1, -1)
        query_output = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_mask,
            return_dict=True,
        ).last_hidden_state.to(image_embeds.dtype)
        visual = model.language_projection(query_output)
        prompt_ids = processor.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_ids = prompt_ids.expand(len(images), -1)
        text = model.language_model.get_input_embeddings()(prompt_ids).to(
            visual.dtype
        )
        embeddings = torch.cat((visual, text), dim=1)
        attention_mask = torch.ones(
            embeddings.shape[:2],
            dtype=torch.long,
            device=device,
        )
        generated = model.language_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    captions = [
        _clean(text, prompt)
        for text in processor.batch_decode(generated, skip_special_tokens=True)
    ]
    del pixel_values, generated, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return captions, elapsed, peak


def _cached_captions(dataset, count, device, prompt, max_new_tokens, num_beams):
    features = torch.stack([dataset[index][0] for index in range(count)])
    model = Net(
        in_shape=(1, 32, 768),
        out_shape=(OPT_VOCAB_SIZE,),
        prm={
            "lr": 1e-5,
            "caption_prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        },
        device=device,
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model(features)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    captions = [
        _clean(text, "")
        for text in model.opt_tokenizer.batch_decode(
            generated.cpu(),
            skip_special_tokens=True,
        )
    ]
    del features, generated, model
    gc.collect()
    torch.cuda.empty_cache()
    return captions, elapsed, peak


def _compare(direct, cached):
    smoothing = SmoothingFunction().method1
    rows = []
    for index, (reference, candidate) in enumerate(zip(direct, cached)):
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        rows.append({
            "index": index,
            "direct": reference,
            "cached": candidate,
            "exact_match": reference.lower() == candidate.lower(),
            "sequence_similarity": SequenceMatcher(
                None,
                reference.lower(),
                candidate.lower(),
            ).ratio(),
            "bleu4_against_direct": sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            ),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--prompt", default="a photo of ")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/blip2-parity-report.json"),
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Parity validation requires CUDA.")

    raw = CocoImages(args.coco_root.expanduser().resolve(), "val")
    cached_dataset = CachedCaptionDataset(
        "val",
        str(args.cache_dir.expanduser().resolve()),
    )
    count = min(args.samples, len(raw), len(cached_dataset))
    images = [raw[index][0] for index in range(count)]
    device = torch.device("cuda")

    direct, direct_seconds, direct_peak = _direct_captions(
        images,
        device,
        args.prompt,
        args.max_new_tokens,
        args.num_beams,
    )
    del images
    gc.collect()
    cached, cached_seconds, cached_peak = _cached_captions(
        cached_dataset,
        count,
        device,
        args.prompt,
        args.max_new_tokens,
        args.num_beams,
    )
    rows = _compare(direct, cached)
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "samples": count,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "direct_seconds": direct_seconds,
        "cached_seconds": cached_seconds,
        "direct_peak_vram_bytes": direct_peak,
        "cached_peak_vram_bytes": cached_peak,
        "exact_match_rate": sum(row["exact_match"] for row in rows) / count,
        "mean_sequence_similarity": sum(
            row["sequence_similarity"] for row in rows
        ) / count,
        "mean_bleu4_against_direct": sum(
            row["bleu4_against_direct"] for row in rows
        ) / count,
        "comparisons": rows,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))
    print(f"Report: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
