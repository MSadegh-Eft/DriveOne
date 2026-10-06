#!/usr/bin/env python3
"""Train the first frozen-feature RefAV ranking controls on a JSONL subset."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def box_features(row: dict[str, Any], image_width: float = 1920.0, image_height: float = 1200.0) -> list[float]:
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
        if mode == "task_id":
            self.prompt = nn.Embedding(prompt_count, 32)
            context_dim = 32
        elif mode == "pooled_pe":
            self.image = nn.Linear(image_dim, 64)
            self.text = nn.Linear(text_dim, 64)
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
            values.append(self.prompt(prompt_id))
        elif self.mode == "pooled_pe":
            assert image is not None and text is not None
            values.extend([self.image(image), self.text(text)])
        return self.head(torch.cat(values, dim=-1)).squeeze(-1)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
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
    labeled_scores = []
    labeled_labels = []
    for rows in groups.values():
        labels = [int(row["label"]) if row.get("label") is not None else -1 for row in rows]
        scores = [float(score_map[id(row)]) for row in rows]
        ap = average_precision(labels, scores)
        if ap is not None:
            aps.append(ap)
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            recalls.append(float(labels[order[0]] == POSITIVE))
            for row, label in zip(rows, labels):
                if label in (POSITIVE, *NEGATIVES):
                    labeled_scores.append(float(score_map[id(row)]))
                    labeled_labels.append(float(label == POSITIVE))
    if not aps:
        return {"group_count": len(groups), "rankable_group_count": 0, "mAP": None, "Recall@1": None}
    output: dict[str, float | int | None] = {
        "group_count": len(groups),
        "rankable_group_count": len(aps),
        "mAP": float(np.mean(aps)),
        "Recall@1": float(np.mean(recalls)),
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
    return output


def build_inputs(rows: list[dict[str, Any]], category_ids: dict[str, int], prompt_ids: dict[str, int], feature_payload: dict[str, Any], device: torch.device) -> dict[int, dict[str, torch.Tensor]]:
    image_lookup = {key: value.float() for key, value in zip(feature_payload["image_keys"], feature_payload["image_features"])}
    text_lookup = {key: value.float() for key, value in zip(feature_payload["text_keys"], feature_payload["text_features"])}
    output = {}
    for row in rows:
        candidate = box_features(row)
        category = category_ids.get(str(row.get("name", "")), 0)
        image = image_lookup[str(row["image_path"])]
        text = text_lookup[str(row["prompt"])]
        output[id(row)] = {
            "candidate": torch.tensor(candidate, dtype=torch.float32, device=device),
            "metadata": torch.tensor(metadata_features(row), dtype=torch.float32, device=device),
            "category": torch.tensor(category, dtype=torch.long, device=device),
            "prompt_id": torch.tensor(prompt_ids.get(str(row["prompt"]), 0), dtype=torch.long, device=device),
            "image": F.normalize(image.to(device), dim=0),
            "text": F.normalize(text.to(device), dim=0),
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
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    rows_by_split = {name: read_jsonl(path) for name, path in [("train", args.train), ("validation", args.validation), ("test", args.test)]}
    groups_by_split = {name: group_rows(rows) for name, rows in rows_by_split.items()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    categories = sorted({str(row.get("name", "")) for row in all_rows})
    category_ids = {name: index for index, name in enumerate(categories)}
    train_prompts = sorted({str(row["prompt"]) for row in rows_by_split["train"]})
    prompt_ids = {prompt: index + 1 for index, prompt in enumerate(train_prompts)}
    prompt_ids["<unk>"] = 0
    feature_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    inputs = {name: build_inputs(rows, category_ids, prompt_ids, feature_payload, device) for name, rows in rows_by_split.items()}

    results: dict[str, Any] = {"protocol": {"seed": args.seed, "epochs": args.epochs, "loss": "BCE on labeled candidates only", "unknown_candidates": "retained at evaluation and omitted from training loss", "device": str(device)}, "models": {}}
    for mode in ("candidate_only", "metadata_only", "task_id", "pooled_pe"):
        model = CandidateScorer(3 if mode == "metadata_only" else 12, len(categories), mode, len(prompt_ids)).to(device)
        start = time.perf_counter()
        train_one(model, groups_by_split["train"], inputs["train"], args.epochs, args.seed)
        elapsed = time.perf_counter() - start
        model_results = {split: ranking_metrics(groups_by_split[split], score_groups(model, groups_by_split[split], inputs[split])) for split in ("train", "validation", "test")}
        results["models"][mode] = {"train_seconds": elapsed, "metrics": model_results, "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
