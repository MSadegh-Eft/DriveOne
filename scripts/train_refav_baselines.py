#!/usr/bin/env python3
"""Train the first frozen-feature RefAV ranking controls on a JSONL subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


POSITIVE = 0
NEGATIVES = {1, 2}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["log_id"]), str(row["prompt"]), int(row["timestamp_ns"]))].append(row)
    return dict(sorted(groups.items()))


def box_features(row: dict[str, Any], image_width: float, image_height: float) -> list[float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Source image dimensions must be positive")
    box = row.get("projected_box")
    if box is None:
        coords = [0.0] * 4
        visible = 0.0
    else:
        coords = [float(box[0]) / image_width, float(box[1]) / image_height, float(box[2]) / image_width, float(box[3]) / image_height]
        visible = 1.0
    translation = list(row.get("translation_m") or [0.0, 0.0, 0.0])
    size = list(row.get("size") or [0.0, 0.0, 0.0])
    distance = float(row.get("distance_m") or 0.0)
    return coords + [visible, distance / 50.0] + [float(value) / 10.0 for value in size[:3]] + [float(value) / 50.0 for value in translation[:3]]


def metadata_features(row: dict[str, Any]) -> list[float]:
    return [float(row.get("score") or 0.0), float(row.get("raw_tracker_label") or 0.0) / 10.0, float(row.get("distance_m") or 0.0) / 50.0]


class CandidateScorer(nn.Module):
    def __init__(self, candidate_dim: int, category_count: int, mode: str, prompt_count: int, image_dim: int = 1024, text_dim: int = 1024):
        super().__init__()
        self.mode = mode
        self.candidate = nn.Sequential(nn.Linear(candidate_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.category = nn.Embedding(category_count, 16)
        context_dim = 0
        if mode == "pooled_pe":
            self.image = nn.Linear(image_dim, 64)
            self.text = nn.Linear(text_dim, 64)
            context_dim = 128
        elif mode == "task_id":
            # Keep the visual input fixed when comparing a learned task ID
            # with natural-language conditioning.  Otherwise the comparison
            # would change both the image input and the question input.
            self.prompt = nn.Embedding(prompt_count, 64)
            self.image = nn.Linear(image_dim, 64)
            context_dim = 128
        elif mode in ("candidate_only", "metadata_only"):
            context_dim = 0
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.head = nn.Sequential(nn.Linear(64 + 16 + context_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, candidate: torch.Tensor, category: torch.Tensor, prompt_id: torch.Tensor | None = None, image: torch.Tensor | None = None, text: torch.Tensor | None = None) -> torch.Tensor:
        values = [self.candidate(candidate), self.category(category)]
        if self.mode == "task_id":
            assert prompt_id is not None
            assert image is not None
            values.extend([self.image(image), self.prompt(prompt_id)])
        elif self.mode == "pooled_pe":
            assert image is not None and text is not None
            values.extend([self.image(image), self.text(text)])
        return self.head(torch.cat(values, dim=-1)).squeeze(-1)


def average_precision(labels: list[int | None], scores: list[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("One score is required per candidate")
    positive_count = sum(label == POSITIVE for label in labels)
    negative_count = sum(label in NEGATIVES for label in labels)
    if not positive_count or not negative_count:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    found = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index] == POSITIVE:
            found += 1
            total += found / rank
    return total / positive_count


def ranking_metrics(groups: dict[tuple[str, str, int], list[dict[str, Any]]], score_map: dict[int, float]) -> dict[str, float | int | None]:
    aps = []
    recalls = []
    labeled_aps = []
    labeled_recalls = []
    labeled_scores = []
    labeled_labels = []
    positive_count = 0
    negative_count = 0
    unknown_count = 0
    for rows in groups.values():
        labels = [int(row["label"]) if row.get("label") is not None else None for row in rows]
        scores = [float(score_map[id(row)]) for row in rows]
        ap = average_precision(labels, scores)
        positive_count += sum(label == POSITIVE for label in labels)
        negative_count += sum(label in NEGATIVES for label in labels)
        unknown_count += sum(label is None for label in labels)
        if ap is not None:
            aps.append(ap)
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            recalls.append(float(labels[order[0]] == POSITIVE))
        labeled_indices = [index for index, label in enumerate(labels) if label in (POSITIVE, *NEGATIVES)]
        if labeled_indices:
            labeled_group_labels = [labels[index] for index in labeled_indices]
            labeled_group_scores = [scores[index] for index in labeled_indices]
            labeled_ap = average_precision(labeled_group_labels, labeled_group_scores)
            if labeled_ap is not None:
                labeled_aps.append(labeled_ap)
                labeled_order = sorted(range(len(labeled_group_scores)), key=lambda index: (-labeled_group_scores[index], index))
                labeled_recalls.append(float(labeled_group_labels[labeled_order[0]] == POSITIVE))
            for label, score in zip(labeled_group_labels, labeled_group_scores):
                labeled_scores.append(float(score))
                labeled_labels.append(float(label == POSITIVE))
    if not aps:
        return {
            "group_count": len(groups), "rankable_group_count": 0, "mAP": None, "Recall@1": None,
            "labeled_only_mAP": float(np.mean(labeled_aps)) if labeled_aps else None,
            "labeled_only_Recall@1": float(np.mean(labeled_recalls)) if labeled_recalls else None,
            "positive_count": positive_count, "negative_count": negative_count, "unknown_count": unknown_count,
        }
    output: dict[str, float | int | None] = {
        "group_count": len(groups),
        "rankable_group_count": len(aps),
        "mAP": float(np.mean(aps)),
        "Recall@1": float(np.mean(recalls)),
        "labeled_only_mAP": float(np.mean(labeled_aps)) if labeled_aps else None,
        "labeled_only_Recall@1": float(np.mean(labeled_recalls)) if labeled_recalls else None,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "unknown_count": unknown_count,
    }
    logits = np.asarray(labeled_scores, dtype=float)
    targets = np.asarray(labeled_labels, dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    output["NLL"] = float(np.mean(-(targets * np.log(probabilities + 1e-8) + (1 - targets) * np.log(1 - probabilities + 1e-8))))
    output["Brier"] = float(np.mean((probabilities - targets) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
    output["ECE"] = float(ece)
    output["calibration_labeled_candidate_count"] = len(targets)
    output["calibration_positive_fraction"] = float(targets.mean())
    output["reliability_bins"] = [
        {
            "lower": float(left), "upper": float(right), "count": int(mask.sum()),
            "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
            "positive_fraction": float(targets[mask].mean()) if mask.any() else None,
        }
        for left, right in zip(bins[:-1], bins[1:])
        for mask in [(probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)]
    ]
    return output


def validate_splits(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen_logs: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        seen_candidates = set()
        group_images = {}
        group_holdout = {}
        for row in rows:
            log = str(row["log_id"])
            if log in seen_logs and seen_logs[log] != split:
                raise ValueError(f"Log {log} occurs in both {seen_logs[log]} and {split}")
            seen_logs[log] = split
            group = (log, str(row["prompt"]), int(row["timestamp_ns"]))
            candidate = (*group, row["track_id"])
            if candidate in seen_candidates:
                raise ValueError(f"Duplicate candidate in {split}: {candidate}")
            seen_candidates.add(candidate)
            image = str(row["image_path"])
            if group in group_images and group_images[group] != image:
                raise ValueError(f"Candidates in {group} do not share one image")
            group_images[group] = image
            holdout = (str(row.get("prompt_split", "")), bool(row.get("joint_holdout_eligible", False)))
            if group in group_holdout and group_holdout[group] != holdout:
                raise ValueError(f"Prompt holdout metadata differs inside {group}")
            group_holdout[group] = holdout
            if row.get("label") not in (None, 0, 1, 2):
                raise ValueError("Labels must be referred=0, related=1, other=2, or null")


def source_image_size(rows: list[dict[str, Any]]) -> tuple[int, int]:
    sizes = set()
    for path in sorted({str(row["image_path"]) for row in rows}):
        with Image.open(path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(f"This smoke test requires a single source image size, found {sorted(sizes)}")
    return next(iter(sizes))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_metrics(groups, scores):
    per_log = {
        log: ranking_metrics({key: rows for key, rows in groups.items() if key[0] == log}, scores)
        for log in sorted({key[0] for key in groups})
    }
    bins = {"1-64": (1, 64), "65-128": (65, 128), "129-256": (129, 256), "257+": (257, math.inf)}
    def property_metrics(predicate):
        return ranking_metrics({key: rows for key, rows in groups.items() if predicate(rows)}, scores)

    return {
        "per_log": per_log,
        "candidate_count": {
            name: ranking_metrics({key: rows for key, rows in groups.items() if low <= len(rows) <= high}, scores)
            for name, (low, high) in bins.items()
        },
        "joint_holdout": property_metrics(lambda rows: bool(rows[0].get("joint_holdout_eligible", False))),
        "non_joint_holdout": property_metrics(lambda rows: not bool(rows[0].get("joint_holdout_eligible", False))),
        "prompt_split": {
            value: property_metrics(lambda rows, value=value: str(rows[0].get("prompt_split", "")) == value)
            for value in sorted({str(rows[0].get("prompt_split", "")) for rows in groups.values()})
        },
    }


def per_group_metrics(groups, scores):
    """Return compact per-group values for paired seed/bootstrap analysis."""
    output = []
    for key, rows in groups.items():
        labels = [int(row["label"]) if row.get("label") is not None else None for row in rows]
        values = [float(scores[id(row)]) for row in rows]
        order = sorted(range(len(values)), key=lambda index: (-values[index], index))
        ap = average_precision(labels, values)
        labeled_indices = [index for index, label in enumerate(labels) if label in (POSITIVE, *NEGATIVES)]
        labeled_labels = [labels[index] for index in labeled_indices]
        labeled_values = [values[index] for index in labeled_indices]
        labeled_order = sorted(range(len(labeled_values)), key=lambda index: (-labeled_values[index], index))
        output.append({
            "log_id": str(key[0]), "prompt": str(key[1]), "timestamp_ns": int(key[2]),
            "prompt_split": str(rows[0].get("prompt_split", "")),
            "joint_holdout_eligible": bool(rows[0].get("joint_holdout_eligible", False)),
            "candidate_count": len(rows),
            "positive_count": sum(label == POSITIVE for label in labels),
            "negative_count": sum(label in NEGATIVES for label in labels),
            "unknown_count": sum(label is None for label in labels),
            "average_precision": ap,
            "recall_at_1": float(labels[order[0]] == POSITIVE) if labels and sum(label == POSITIVE for label in labels) else None,
            "labeled_average_precision": average_precision(labeled_labels, labeled_values) if labeled_values else None,
            "labeled_recall_at_1": float(labeled_labels[labeled_order[0]] == POSITIVE) if labeled_values and sum(label == POSITIVE for label in labeled_labels) else None,
        })
    return output


def build_inputs(
    rows: list[dict[str, Any]],
    category_ids: dict[str, int],
    prompt_ids: dict[str, int],
    feature_payload: dict[str, Any],
    device: torch.device,
    image_width: float,
    image_height: float,
) -> dict[int, dict[str, torch.Tensor]]:
    if len(feature_payload["image_keys"]) != len(feature_payload["image_features"]) or len(feature_payload["text_keys"]) != len(feature_payload["text_features"]):
        raise ValueError("Feature key and tensor counts disagree")
    image_lookup = {key: F.normalize(value.float().to(device), dim=0) for key, value in zip(feature_payload["image_keys"], feature_payload["image_features"])}
    text_lookup = {key: F.normalize(value.float().to(device), dim=0) for key, value in zip(feature_payload["text_keys"], feature_payload["text_features"])}
    output = {}
    for row in rows:
        candidate = box_features(row, image_width=image_width, image_height=image_height)
        category = category_ids.get(str(row["raw_tracker_label"]), 0)
        image = image_lookup[str(row["image_path"])]
        text = text_lookup[str(row["prompt"])]
        output[id(row)] = {
            "candidate": torch.tensor(candidate, dtype=torch.float32, device=device),
            "metadata": torch.tensor(metadata_features(row), dtype=torch.float32, device=device),
            "category": torch.tensor(category, dtype=torch.long, device=device),
            "prompt_id": torch.tensor(prompt_ids.get(str(row["prompt"]), 0), dtype=torch.long, device=device),
            "image": image,
            "text": text,
        }
    return output


def score_groups(model: CandidateScorer, groups: dict[tuple[str, str, int], list[dict[str, Any]]], inputs: dict[int, dict[str, torch.Tensor]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    model.eval()
    with torch.inference_mode():
        for rows in groups.values():
            values = [inputs[id(row)] for row in rows]
            candidate = torch.stack([value["metadata"] if model.mode == "metadata_only" else value["candidate"] for value in values])
            category = torch.stack([value["category"] for value in values])
            prompt_id = torch.stack([value["prompt_id"] for value in values])
            image = torch.stack([value["image"] for value in values])
            text = torch.stack([value["text"] for value in values])
            output = model(candidate, category, prompt_id, image, text)
            for row, score in zip(rows, output.detach().cpu().tolist()):
                scores[id(row)] = float(score)
    return scores


def train_one(model: CandidateScorer, groups: dict[tuple[str, str, int], list[dict[str, Any]]], inputs: dict[int, dict[str, torch.Tensor]], epochs: int, seed: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = random.Random(seed)
    keys = list(groups)
    for _ in range(epochs):
        rng.shuffle(keys)
        model.train()
        for key in keys:
            rows = groups[key]
            labeled = [row for row in rows if row.get("label") is not None]
            positives = [row for row in labeled if int(row["label"]) == POSITIVE]
            negatives = [row for row in labeled if int(row["label"]) in NEGATIVES]
            if not positives or not negatives:
                continue
            values = [inputs[id(row)] for row in labeled]
            scores = model(
                torch.stack([value["metadata"] if model.mode == "metadata_only" else value["candidate"] for value in values]),
                torch.stack([value["category"] for value in values]),
                torch.stack([value["prompt_id"] for value in values]),
                torch.stack([value["image"] for value in values]),
                torch.stack([value["text"] for value in values]),
            )
            target = torch.tensor([1.0 if int(row["label"]) == POSITIVE else 0.0 for row in labeled], device=scores.device)
            # BCE keeps multiple positives valid without treating unknown rows as
            # negatives. Ranking is still evaluated over the complete pool.
            loss = F.binary_cross_entropy_with_logits(scores, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, help="Optional assertion about source width; actual files are inspected")
    parser.add_argument("--image-height", type=int, help="Optional assertion about source height; actual files are inspected")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    read_start = time.perf_counter()
    rows_by_split = {name: read_jsonl(path) for name, path in [("train", args.train), ("validation", args.validation), ("test", args.test)]}
    read_seconds = time.perf_counter() - read_start
    validate_splits(rows_by_split)
    groups_by_split = {name: group_rows(rows) for name, rows in rows_by_split.items()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    categories = sorted({str(row["raw_tracker_label"]) for row in rows_by_split["train"]})
    category_ids = {name: index + 1 for index, name in enumerate(categories)}
    image_start = time.perf_counter()
    width, height = source_image_size(all_rows)
    image_validation_seconds = time.perf_counter() - image_start
    if (args.image_width is not None and args.image_width != width) or (args.image_height is not None and args.image_height != height):
        parser.error(f"Image-size assertion disagrees with actual files: {width}x{height}")
    train_prompts = sorted({str(row["prompt"]) for row in rows_by_split["train"]})
    prompt_ids = {prompt: index + 1 for index, prompt in enumerate(train_prompts)}
    prompt_ids["<unk>"] = 0
    feature_start = time.perf_counter()
    feature_payload = torch.load(args.features, map_location="cpu", weights_only=True)
    feature_load_seconds = time.perf_counter() - feature_start
    if feature_payload.get("config") != "PE-Core-L14-336" or feature_payload.get("pretrained") is not True:
        raise ValueError("This control protocol requires pretrained PE-Core-L14-336 features")
    input_start = time.perf_counter()
    inputs = {
        name: build_inputs(rows, category_ids, prompt_ids, feature_payload, device, width, height)
        for name, rows in rows_by_split.items()
    }
    input_build_seconds = time.perf_counter() - input_start

    results: dict[str, Any] = {
        "protocol": {
            "seed": args.seed,
            "epochs": args.epochs,
            "loss": "BCE on labeled candidates only",
            "unknown_candidates": "retained at evaluation and omitted from training loss",
            "device": str(device),
            "version": "refav-baseline-gate-v1",
            "source_image_size": [width, height],
            "dtype": "float32",
            "torch_version": str(torch.__version__),
            "threads": args.threads,
            "timings_seconds": {
                "jsonl_read": read_seconds,
                "image_validation": image_validation_seconds,
                "feature_load": feature_load_seconds,
                "input_build": input_build_seconds,
            },
            "script_sha256": sha256_file(Path(__file__)),
            "features_sha256": sha256_file(args.features),
            "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in (("train", args.train), ("validation", args.validation), ("test", args.test))},
            "category_vocabulary": "raw tracker categories fitted on training rows only; unknown=0",
            "query_holdout": "joint_holdout_eligible groups are reported separately; the main split remains log-disjoint with exact prompt overlap across logs",
            "candidate_order": "deterministically shuffled within each group before every model sees the rows",
            "task_id_conditioning": "pooled PE image plus learned prompt ID",
            "pooled_pe_conditioning": "pooled PE image plus PE text",
            "matching_limit": "context and scorer widths match; total trainable parameter counts differ",
            "latency_limit": "cached-feature training/evaluation only; no end-to-end latency claim",
        },
        "models": {},
    }
    for mode in ("candidate_only", "metadata_only", "task_id", "pooled_pe"):
        seed_everything(args.seed)
        model = CandidateScorer(3 if mode == "metadata_only" else 12, len(categories) + 1, mode, len(prompt_ids)).to(device)
        start = time.perf_counter()
        train_one(model, groups_by_split["train"], inputs["train"], args.epochs, args.seed)
        elapsed = time.perf_counter() - start
        score_start = time.perf_counter()
        model_scores = {split: score_groups(model, groups_by_split[split], inputs[split]) for split in ("train", "validation", "test")}
        score_seconds = time.perf_counter() - score_start
        model_results = {split: ranking_metrics(groups_by_split[split], model_scores[split]) for split in ("train", "validation", "test")}
        results["models"][mode] = {
            "train_seconds": elapsed, "score_seconds": score_seconds, "metrics": model_results,
            "stratified_metrics": {split: stratified_metrics(groups_by_split[split], model_scores[split]) for split in ("validation", "test")},
            "per_group_metrics": {split: per_group_metrics(groups_by_split[split], model_scores[split]) for split in ("validation", "test")},
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
