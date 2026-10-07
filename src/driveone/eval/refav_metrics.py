"""Leakage-aware deterministic controls for RefAV referred-track ranking."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np


POSITIVE = 0
NEGATIVES = {1, 2}


def _label(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _box_area(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _tracker_score(row: Mapping[str, Any]) -> float:
    """Read the historical ``score`` alias or the repaired field."""
    try:
        return float(row.get("score", row.get("tracker_score")))
    except (TypeError, ValueError):
        return float("-inf")


def _stable_random_order(size: int, seed: int, group_index: int) -> np.ndarray:
    return np.random.default_rng(seed + 1009 * group_index).permutation(size)


def _average_precision(labels: list[int | None], order: Iterable[int]) -> float | None:
    positives = sum(label == POSITIVE for label in labels)
    if positives == 0:
        return None
    found = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == POSITIVE:
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def _group_metrics(labels: list[int | None], order: Iterable[int]) -> dict[str, float | int | None]:
    order = list(order)
    positive_count = sum(label == POSITIVE for label in labels)
    negative_count = sum(label in NEGATIVES for label in labels)
    ap = _average_precision(labels, order)
    top1 = labels[order[0]] if order else None
    labeled_order = [index for index in order if labels[index] is not None]
    labeled_labels = [labels[index] for index in labeled_order]
    labeled_ap = _average_precision(labeled_labels, range(len(labeled_labels)))
    labeled_top1 = labeled_labels[0] if labeled_labels else None
    return {
        "candidate_count": len(labels),
        "labeled_candidate_count": len(labeled_labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "average_precision": ap,
        "recall_at_1": int(top1 == POSITIVE) if positive_count and order else None,
        "labeled_average_precision": labeled_ap,
        "labeled_recall_at_1": int(labeled_top1 == POSITIVE) if positive_count and labeled_order else None,
    }


def _rank_order(name: str, rows: list[Mapping[str, Any]], seed: int, group_index: int, category_frequency: Mapping[str, int]) -> list[int]:
    tie = _stable_random_order(len(rows), seed, group_index)
    tie_rank = {int(index): rank for rank, index in enumerate(tie)}

    def key(index: int) -> tuple[float, int]:
        row = rows[index]
        if name == "random":
            return (0.0, tie_rank[index])
        if name == "oracle":
            return (0.0 if _label(row.get("label")) == POSITIVE else 1.0, tie_rank[index])
        if name == "tracker_score":
            value = _tracker_score(row)
            return (-value, tie_rank[index])
        if name == "candidate_distance":
            try:
                value = float(row.get("distance_m"))
            except (TypeError, ValueError):
                value = float("inf")
            return (value, tie_rank[index])
        if name == "projected_box_area":
            return (-_box_area(row.get("projected_box")), tie_rank[index])
        if name == "category_frequency":
            category = str(row.get("name", ""))
            return (-float(category_frequency.get(category, 0)), tie_rank[index])
        raise ValueError(f"Unknown control: {name}")

    return sorted(range(len(rows)), key=key)


def _summarize(group_results: list[dict[str, float | int | None]]) -> dict[str, Any]:
    rankable = [result for result in group_results if result["positive_count"] and result["negative_count"]]
    aps = [float(result["average_precision"]) for result in rankable if result["average_precision"] is not None]
    recalls = [float(result["recall_at_1"]) for result in rankable if result["recall_at_1"] is not None]
    labeled_aps = [float(result["labeled_average_precision"]) for result in rankable if result["labeled_average_precision"] is not None]
    labeled_recalls = [float(result["labeled_recall_at_1"]) for result in rankable if result["labeled_recall_at_1"] is not None]
    return {
        "group_count": len(group_results),
        "rankable_group_count": len(rankable),
        "mean_average_precision": float(np.mean(aps)) if aps else None,
        "recall_at_1": float(np.mean(recalls)) if recalls else None,
        "labeled_only_mean_average_precision": float(np.mean(labeled_aps)) if labeled_aps else None,
        "labeled_only_recall_at_1": float(np.mean(labeled_recalls)) if labeled_recalls else None,
        "mean_candidate_count": float(np.mean([result["candidate_count"] for result in group_results])) if group_results else None,
        "mean_positive_count": float(np.mean([result["positive_count"] for result in group_results])) if group_results else None,
        "mean_negative_count": float(np.mean([result["negative_count"] for result in group_results])) if group_results else None,
    }


def _projected_box_area(row: Mapping[str, Any]) -> float:
    return _box_area(row.get("projected_box"))


def _hard_negative_rows(
    rows: list[Mapping[str, Any]],
    score_delta: float,
    log_area_delta: float | None = None,
) -> list[Mapping[str, Any]]:
    """Keep positives and labeled negatives matched to positive metadata.

    The optional area constraint uses ``log1p(projected pixel area)``.  This
    makes the tolerance approximately multiplicative for visible boxes while
    keeping zero-area/out-of-view boxes well-defined.  This is an evaluation
    subset only; these fields are never supplied as model inputs.
    """
    positive_scores = []
    positive_log_areas = []
    for row in rows:
        if _label(row.get("label")) != POSITIVE:
            continue
        score = _tracker_score(row)
        if not np.isfinite(score):
            continue
        positive_scores.append(score)
        positive_log_areas.append(float(np.log1p(_projected_box_area(row))))
    if not positive_scores:
        return []
    selected = []
    for row in rows:
        label = _label(row.get("label"))
        if label == POSITIVE:
            selected.append(row)
            continue
        if label not in NEGATIVES:
            continue
        score = _tracker_score(row)
        if not np.isfinite(score):
            continue
        score_match = min(abs(score - positive_score) for positive_score in positive_scores) <= score_delta
        if not score_match:
            continue
        if log_area_delta is not None:
            log_area = float(np.log1p(_projected_box_area(row)))
            area_match = min(abs(log_area - positive_area) for positive_area in positive_log_areas) <= log_area_delta
            if not area_match:
                continue
        selected.append(row)
    return selected


def run_control_suite(
    records: Iterable[Mapping[str, Any]],
    seeds: Iterable[int] = (0, 1),
    hard_negative_delta: float = 0.05,
    size_matched_log_area_delta: float = 0.2,
) -> dict[str, Any]:
    """Evaluate deterministic controls while retaining unknown candidates.

    Primary ranking metrics use only groups with at least one labeled positive
    and one labeled negative. Unknown candidates remain in the ranked pool and
    can therefore lower a method's score; they are never relabeled as negatives.
    """
    groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    all_rows = list(records)
    for row in all_rows:
        key = (row.get("log_id"), row.get("prompt"), row.get("timestamp_ns"))
        groups.setdefault(key, []).append(row)

    category_frequency = Counter(str(row.get("name", "")) for row in all_rows)
    controls = ("random", "tracker_score", "candidate_distance", "projected_box_area", "category_frequency", "oracle")
    results: dict[str, Any] = {}
    for seed in seeds:
        for control in controls:
            per_group = []
            for group_index, rows in enumerate(groups.values()):
                labels = [_label(row.get("label")) for row in rows]
                order = _rank_order(control, rows, int(seed), group_index, category_frequency)
                per_group.append(_group_metrics(labels, order))
            results[f"{control}:seed_{seed}"] = _summarize(per_group)

    hard_groups = {
        key: _hard_negative_rows(rows, hard_negative_delta)
        for key, rows in groups.items()
    }
    hard_groups = {
        key: rows
        for key, rows in hard_groups.items()
        if any(_label(row.get("label")) == POSITIVE for row in rows)
        and any(_label(row.get("label")) in NEGATIVES for row in rows)
    }
    size_matched_groups = {
        key: _hard_negative_rows(rows, hard_negative_delta, log_area_delta=size_matched_log_area_delta)
        for key, rows in groups.items()
    }
    size_matched_groups = {
        key: rows
        for key, rows in size_matched_groups.items()
        if any(_label(row.get("label")) == POSITIVE for row in rows)
        and any(_label(row.get("label")) in NEGATIVES for row in rows)
    }

    def evaluate_subset(subset: Mapping[Any, list[Mapping[str, Any]]]) -> dict[str, Any]:
        subset_results: dict[str, Any] = {}
        for seed in seeds:
            for control in controls:
                per_group = []
                for group_index, rows in enumerate(subset.values()):
                    labels = [_label(row.get("label")) for row in rows]
                    order = _rank_order(control, rows, int(seed), group_index, category_frequency)
                    per_group.append(_group_metrics(labels, order))
                subset_results[f"{control}:seed_{seed}"] = _summarize(per_group)
        return subset_results

    return {
        "protocol": {
            "task": "RefAV referred-track ranking",
            "unknown_candidates": "retained in the candidate pool and excluded from label counts",
            "primary_groups": "at least one REFERRED_OBJECT and one RELATED_OBJECT/OTHER_OBJECT",
            "controls": list(controls),
            "seeds": [int(seed) for seed in seeds],
        },
        "dataset": {
            "record_count": len(all_rows),
            "group_count": len(groups),
            "log_count": len({row.get("log_id") for row in all_rows}),
            "labeled_record_count": sum(_label(row.get("label")) is not None for row in all_rows),
            "unmatched_record_count": sum(_label(row.get("label")) is None for row in all_rows),
            "rankable_group_count": sum(
                any(_label(row.get("label")) == POSITIVE for row in rows)
                and any(_label(row.get("label")) in NEGATIVES for row in rows)
                for rows in groups.values()
            ),
        },
        "results": results,
        "hard_negative_results": {
            "score_tolerance": hard_negative_delta,
            "group_count": len(hard_groups),
            "results": evaluate_subset(hard_groups),
        },
        "score_and_size_matched_results": {
            "score_tolerance": hard_negative_delta,
            "log1p_projected_area_tolerance": size_matched_log_area_delta,
            "group_count": len(size_matched_groups),
            "results": evaluate_subset(size_matched_groups),
        },
    }
