#!/usr/bin/env python3
"""Extract frozen PE-Core pooled image and text features for a JSONL subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", default="PE-Core-L14-336")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    image_paths: set[str] = set()
    prompts: set[str] = set()
    for input_path in args.inputs:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image_path = str(row.get("image_path") or "")
                prompt = str(row.get("prompt") or "")
                if image_path:
                    image_paths.add(image_path)
                if prompt:
                    prompts.add(prompt)
    image_keys = sorted(image_paths)
    prompt_keys = sorted(prompts)
    device = torch.device(args.device)
    model = pe.CLIP.from_config(args.config, pretrained=args.pretrained).to(device).eval()
    transform = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    image_features = []
    image_failures = []
    image_start = time.perf_counter()
    for start in range(0, len(image_keys), args.batch_size):
        batch_paths = image_keys[start : start + args.batch_size]
        tensors = []
        valid_paths = []
        for path in batch_paths:
            try:
                tensors.append(transform(Image.open(path).convert("RGB")))
                valid_paths.append(path)
            except Exception as exc:
                image_failures.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            synchronize(torch, device)
            values = model.encode_image(batch).detach().float().cpu()
            synchronize(torch, device)
        image_features.extend(zip(valid_paths, values))

    text_tokens = tokenizer(prompt_keys, context_length=model.context_length)
    with torch.inference_mode():
        synchronize(torch, device)
        text_values = model.encode_text(text_tokens.to(device)).detach().float().cpu()
        synchronize(torch, device)

    image_map = {path: value for path, value in image_features}
    # Keep the key list aligned with the feature rows.  If an image fails to
    # decode, silently retaining the original key list would make downstream
    # ``zip(keys, features)`` lookups assign the wrong feature to later images.
    successful_image_keys = [path for path in image_keys if path in image_map]
    payload = {
        "config": args.config,
        "pretrained": args.pretrained,
        "image_keys": successful_image_keys,
        "image_features": torch.stack([image_map[path] for path in successful_image_keys]),
        "text_keys": prompt_keys,
        "text_features": text_values,
        "context_length": int(model.context_length),
        "image_size": int(model.image_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "status": "ok" if not image_failures else "image_failures",
        "config": args.config,
        "pretrained": args.pretrained,
        "device": str(device),
        "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.inputs],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "image_count": len(image_keys),
        "image_feature_count": len(image_map),
        "image_feature_shape": list(payload["image_features"].shape),
        "prompt_count": len(prompt_keys),
        "text_feature_shape": list(text_values.shape),
        "context_length": int(model.context_length),
        "image_size": int(model.image_size),
        "image_failures": image_failures,
        "image_elapsed_s": time.perf_counter() - image_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not image_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
