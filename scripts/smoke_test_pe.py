#!/usr/bin/env python3
"""Smoke-test the official PE-Core-L14-336 pooled and patch interfaces.

This script intentionally imports the official ``perception_models`` package at
runtime. It does not vendor weights or dependencies into DriveOne.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def shape(value):
    return list(value.shape) if hasattr(value, "shape") else None


def synchronize(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="PE-Core-L14-336")
    parser.add_argument("--image", help="Local image; if omitted, use a zero image for an interface-only check")
    parser.add_argument("--text", default="a driving scene")
    parser.add_argument("--device", default="cuda" if importlib.util.find_spec("torch") else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="Load the Hugging Face checkpoint")
    parser.add_argument("--output", help="Write JSON results")
    parser.add_argument("--strict", action="store_true", help="Fail if official dependencies/checkpoint are unavailable")
    args = parser.parse_args()

    result = {
        "status": "not_ready",
        "config": args.config,
        "pretrained_requested": args.pretrained,
        "image": str(Path(args.image).resolve()) if args.image else None,
        "warnings": [],
    }
    try:
        import torch
        from PIL import Image
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        from core.vision_encoder.tokenizer import SimpleTokenizer
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"].append("Install the official perception_models dependencies before running the pretrained smoke test.")
        output = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
        return 2 if args.strict else 0

    device = torch.device(args.device)
    try:
        model = pe.VisionTransformer.from_config(args.config, pretrained=args.pretrained).to(device).eval()
        image_size = model.image_size
        if args.image:
            image = transforms.get_image_transform(image_size)(Image.open(args.image).convert("RGB")).unsqueeze(0)
        else:
            image = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
            result["warnings"].append("No image supplied; visual values are synthetic and only interface shapes are meaningful.")
        image = image.to(device)

        with torch.inference_mode():
            synchronize(torch, device)
            start = time.perf_counter()
            features_with_cls = model.forward_features(image, strip_cls_token=False)
            synchronize(torch, device)
            patch_latency_ms = (time.perf_counter() - start) * 1000
            features_without_cls = model.forward_features(image, strip_cls_token=True)
            synchronize(torch, device)
            start = time.perf_counter()
            pooled = model(image)
            synchronize(torch, device)
            pooled_latency_ms = (time.perf_counter() - start) * 1000

        tokenizer = SimpleTokenizer()
        text_tokens = tokenizer.encode(args.text)
        result.update({
            "status": "ok",
            "device": str(device),
            "image_size": image_size,
            "context_length": getattr(model, "context_length", None),
            "text_model_path": "not_run_by_this_vision_tower_smoke_test",
            "tokenizer_context_length": tokenizer.context_length,
            "text_token_count": len(text_tokens),
            "text_token_ids": text_tokens,
            "vision_dtype": str(next(model.parameters()).dtype),
            "features_with_cls_shape": shape(features_with_cls),
            "features_without_cls_shape": shape(features_without_cls),
            "pooled_output_shape": shape(pooled),
            "patch_forward_latency_ms": patch_latency_ms,
            "pooled_forward_latency_ms": pooled_latency_ms,
            "patch_token_count": shape(features_without_cls)[1] if shape(features_without_cls) and len(shape(features_without_cls)) > 1 else None,
        })
        if args.pretrained:
            result["checkpoint_note"] = "Checkpoint loaded through official VisionTransformer.from_config. Record package/repository/checkpoint revisions with the run manifest."
        else:
            result["warnings"].append("The model was not pretrained; rerun with --pretrained before using these values as evidence.")
        result["warnings"].append("This command loads VisionTransformer only; it does not test the official CLIP text encoder.")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"].append("The requested configuration did not complete the official vision smoke test.")
        result["status"] = "failed"

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    if args.strict and result["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
