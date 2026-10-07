"""Protocol-diagnosis helpers for the RefAV referred-track ranking pilot.

This module inspects the fixed candidate rows and existing baseline/control
artifacts.  It does not train a model and it does not create a new candidate
pool.  The purpose is to measure whether labels are correlated with visible
geometry, tracker metadata, or candidate availability strongly enough to make
the current ranking task a shortcut.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from driveone.eval.refav_metrics import POSITIVE, NEGATIVES, run_control_suite


COUNT_BINS: dict[str, tuple[int, float]] = {
    "1-64": (1, 64),
    "65-128": (65, 128),
    "129-256": (129, 256),
    "257+": (257, math.inf),
}
LABEL_NAMES = {POSITIVE: "REFERRED_OBJECT", 1: "RELATED_OBJECT", 2: "OTHER_OBJECT", None: "UNKNOWN"}
CONTROL_NAMES = (
    "random",
    "tracker_score",
    "candidate_distance",
    "projected_box_area",
    "category_frequency",
    "oracle",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of one external input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    """Group rows by the fixed RefAV ranking key."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        key = (str(row["log_id"]), str(row["prompt"]), int(row["timestamp_ns"]))
        groups[key].append(row)
    return dict(sorted(groups.items()))


def label_value(value: Any) -> int | None:
    """Normalize JSON numeric/string labels while preserving unknown rows."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number in (POSITIVE, *NEGATIVES) else None


def projected_area(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    size = row.get("size") or []
    translation = row.get("translation_m") or []
    return {
        "projected_area_px": projected_area(row.get("projected_box")),
        "tracker_score": _number(row.get("score")),
        "distance_m": _number(row.get("distance_m")),
        "match_distance_m": _number(row.get("match_distance_m")),
        "size_length_m": _number(size[0]) if len(size) > 0 else None,
        "size_width_m": _number(size[1]) if len(size) > 1 else None,
        "size_height_m": _number(size[2]) if len(size) > 2 else None,
        "translation_x_m": _number(translation[0]) if len(translation) > 0 else None,
        "translation_y_m": _number(translation[1]) if len(translation) > 1 else None,
        "translation_z_m": _number(translation[2]) if len(translation) > 2 else None,
    }


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0, "missing": 0, "mean": None, "std": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "missing": 0,
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _distribution_with_missing(values: Iterable[float | None]) -> dict[str, Any]:
    values = list(values)
    output = _distribution(values)
    output["missing"] = int(sum(value is None for value in values))
    return output


def _compact_metric(metric: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep ranking/calibration fields and discard reliability-bin detail."""
    if not metric:
        return {}
    keys = (
        "group_count",
        "rankable_group_count",
        "mAP",
        "Recall@1",
        "mean_average_precision",
        "recall_at_1",
        "labeled_only_mean_average_precision",
        "labeled_only_recall_at_1",
        "mean_candidate_count",
        "mean_positive_count",
        "mean_negative_count",
        "NLL",
        "Brier",
        "ECE",
        "calibration_labeled_candidate_count",
        "calibration_positive_fraction",
    )
    return {key: metric[key] for key in keys if key in metric}


def _compact_control_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    def compact_results(results: Mapping[str, Any]) -> dict[str, Any]:
        return {
            control: _compact_metric(results.get(f"{control}:seed_0"))
            for control in CONTROL_NAMES
            if f"{control}:seed_0" in results
        }

    return {
        "dataset": dict(suite.get("dataset", {})),
        "full_pool": compact_results(suite.get("results", {})),
        "score_matched": {
            "group_count": suite.get("hard_negative_results", {}).get("group_count"),
            "results": compact_results(suite.get("hard_negative_results", {}).get("results", {})),
        },
        "score_and_size_matched": {
            "group_count": suite.get("score_and_size_matched_results", {}).get("group_count"),
            "results": compact_results(suite.get("score_and_size_matched_results", {}).get("results", {})),
        },
    }


def _label_distribution(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = label_value(row.get("label"))
        by_label[LABEL_NAMES[label]].append(row)

    features = (
        "projected_area_px",
        "tracker_score",
        "distance_m",
        "match_distance_m",
        "size_length_m",
        "size_width_m",
        "size_height_m",
        "translation_x_m",
        "translation_y_m",
        "translation_z_m",
    )
    output: dict[str, Any] = {}
    for label_name in ("REFERRED_OBJECT", "RELATED_OBJECT", "OTHER_OBJECT", "UNKNOWN"):
        label_rows = by_label.get(label_name, [])
        feature_rows = [_feature_values(row) for row in label_rows]
        output[label_name] = {
            "row_count": len(label_rows),
            "features": {
                feature: _distribution_with_missing(item[feature] for item in feature_rows)
                for feature in features
            },
            "projection_status": dict(sorted(Counter(str(row.get("projection_status", "MISSING")) for row in label_rows).items())),
            "match_status": dict(sorted(Counter(str(row.get("match_status", "MISSING")) for row in label_rows).items())),
            "categories": dict(Counter(str(row.get("name", "UNKNOWN")) for row in label_rows).most_common()),
        }
    referred = output["REFERRED_OBJECT"]
    negatives = output["RELATED_OBJECT"]["row_count"] + output["OTHER_OBJECT"]["row_count"]
    referred_projected = referred["projection_status"].get("PROJECTED", 0)
    negative_projected = sum(
        output[name]["projection_status"].get("PROJECTED", 0)
        for name in ("RELATED_OBJECT", "OTHER_OBJECT")
    )
    output["comparison"] = {
        "referred_visible_rate": referred_projected / referred["row_count"] if referred["row_count"] else None,
        "negative_visible_rate": negative_projected / negatives if negatives else None,
        "visibility_gap_referred_minus_negative": (
            referred_projected / referred["row_count"] - negative_projected / negatives
            if referred["row_count"] and negatives
            else None
        ),
    }
    return output


def _group_summary(groups: Mapping[tuple[str, str, int], list[Mapping[str, Any]]]) -> dict[str, Any]:
    counts = [len(rows) for rows in groups.values()]
    positive_groups = sum(any(label_value(row.get("label")) == POSITIVE for row in rows) for rows in groups.values())
    rankable_groups = sum(
        any(label_value(row.get("label")) == POSITIVE for row in rows)
        and any(label_value(row.get("label")) in NEGATIVES for row in rows)
        for rows in groups.values()
    )
    labels = [label_value(row.get("label")) for rows in groups.values() for row in rows]
    return {
        "group_count": len(groups),
        "rankable_group_count": rankable_groups,
        "positive_group_count": positive_groups,
        "oracle_candidate_coverage": positive_groups / len(groups) if groups else None,
        "rankable_group_rate": rankable_groups / len(groups) if groups else None,
        "candidate_count": _distribution(counts),
        "row_count": len(labels),
        "label_counts": {LABEL_NAMES[label]: int(labels.count(label)) for label in (POSITIVE, 1, 2, None)},
        "unknown_row_rate": labels.count(None) / len(labels) if labels else None,
    }


def _subset_rows(groups: Mapping[tuple[str, str, int], list[Mapping[str, Any]]], predicate: Any) -> list[Mapping[str, Any]]:
    return [row for key, rows in groups.items() if predicate(key, rows) for row in rows]


def _control_strata(groups: Mapping[tuple[str, str, int], list[Mapping[str, Any]]]) -> dict[str, Any]:
    by_log: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for key, rows in groups.items():
        by_log[key[0]].extend(rows)
    by_count = {
        name: _subset_rows(groups, lambda _key, rows, low=low, high=high: low <= len(rows) <= high)
        for name, (low, high) in COUNT_BINS.items()
    }

    def suite_for(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"dataset": {"record_count": 0, "group_count": 0}, "full_pool": {}}
        return _compact_control_suite(run_control_suite(rows, seeds=[0]))

    return {
        "per_log": {log_id: suite_for(rows) for log_id, rows in sorted(by_log.items())},
        "candidate_count": {name: suite_for(rows) for name, rows in by_count.items()},
    }


def _learned_strata(baseline: Mapping[str, Any], split: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model, model_data in baseline.get("models", {}).items():
        strata = model_data.get("stratified_metrics", {}).get(split, {})
        output[model] = {
            "per_log": {
                key: _compact_metric(value)
                for key, value in strata.get("per_log", {}).items()
            },
            "candidate_count": {
                key: _compact_metric(value)
                for key, value in strata.get("candidate_count", {}).items()
            },
        }
    return output


def _shortcut_flags(
    feature_summary: Mapping[str, Any],
    controls: Mapping[str, Any],
    baseline: Mapping[str, Any],
    split: str,
    *,
    visibility_gap_threshold: float = 0.10,
    area_ratio_threshold: float = 1.5,
    control_margin: float = 0.03,
) -> dict[str, Any]:
    referred = feature_summary["REFERRED_OBJECT"]
    negatives = feature_summary["RELATED_OBJECT"]["row_count"] + feature_summary["OTHER_OBJECT"]["row_count"]
    negative_features = []
    for name in ("RELATED_OBJECT", "OTHER_OBJECT"):
        negative_features.extend([feature_summary[name]["features"]])

    def median(label_name: str, feature: str) -> float | None:
        return feature_summary[label_name]["features"][feature].get("median")

    pos_area = median("REFERRED_OBJECT", "projected_area_px")
    neg_area_values = [median(name, "projected_area_px") for name in ("RELATED_OBJECT", "OTHER_OBJECT")]
    neg_area_values = [value for value in neg_area_values if value is not None and value > 0]
    neg_area = float(np.mean(neg_area_values)) if neg_area_values else None
    area_ratio = pos_area / neg_area if pos_area is not None and neg_area else None
    visibility_gap = feature_summary["comparison"].get("visibility_gap_referred_minus_negative")

    model_metrics = baseline.get("models", {}).get("pooled_pe", {}).get("metrics", {}).get(split, {})
    pooled_map = model_metrics.get("mAP")
    full_controls = controls.get("full_pool", {})
    control_maps = {
        name: values.get("mean_average_precision")
        for name, values in full_controls.items()
        if name != "oracle" and values.get("mean_average_precision") is not None
    }
    dominant_controls = {
        name: value
        for name, value in control_maps.items()
        if pooled_map is not None and value - pooled_map >= control_margin
    }
    size_controls = controls.get("score_and_size_matched", {}).get("results", {})
    size_maps = {
        name: values.get("mean_average_precision")
        for name, values in size_controls.items()
        if name != "oracle" and values.get("mean_average_precision") is not None
    }
    strongest_size_control = max(size_maps.values(), default=None)

    flags = {
        "visibility_gap": {
            "value": visibility_gap,
            "threshold": visibility_gap_threshold,
            "flagged": visibility_gap is not None and visibility_gap >= visibility_gap_threshold,
        },
        "projected_area_median_ratio": {
            "value": area_ratio,
            "threshold": area_ratio_threshold,
            "flagged": area_ratio is not None and area_ratio >= area_ratio_threshold,
        },
        "full_pool_control_dominance": {
            "pooled_pe_mAP": pooled_map,
            "control_mAP": control_maps,
            "margin": control_margin,
            "dominant_controls": dominant_controls,
            "flagged": bool(dominant_controls),
        },
        "score_and_size_matched_control": {
            "control_mAP": size_maps,
            "strongest_mAP": strongest_size_control,
            "pooled_pe_mAP": pooled_map,
            "flagged": strongest_size_control is not None and pooled_map is not None and strongest_size_control - pooled_map >= control_margin,
        },
        "positive_rows": referred["row_count"],
        "negative_rows": negatives,
    }
    reasons = []
    if flags["visibility_gap"]["flagged"]:
        reasons.append("referred rows are visibly more likely to have a projected box than labeled negatives")
    if flags["projected_area_median_ratio"]["flagged"]:
        reasons.append("referred rows have a substantially larger median projected area")
    if flags["full_pool_control_dominance"]["flagged"]:
        reasons.append("deterministic controls beat pooled PE by the declared mAP margin on the full pool")
    if flags["score_and_size_matched_control"]["flagged"]:
        reasons.append("a deterministic control remains stronger after score-and-size matching")
    flags["reasons"] = reasons
    return flags


def diagnose_split(
    rows: list[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    controls: Mapping[str, Any],
    split: str,
    **thresholds: float,
) -> dict[str, Any]:
    groups = group_rows(rows)
    features = _label_distribution(rows)
    compact_controls = _compact_control_suite(controls)
    flags = _shortcut_flags(features, compact_controls, baseline, split, **thresholds)
    return {
        "dataset": _group_summary(groups),
        "label_and_feature_distributions": features,
        "controls": compact_controls,
        "control_strata": _control_strata(groups),
        "learned_model_strata": _learned_strata(baseline, split),
        "shortcut_flags": flags,
    }


def decide_report(splits: Mapping[str, Mapping[str, Any]], *, minimum_rankable_groups: int = 50) -> dict[str, Any]:
    """Apply the predeclared diagnosis decision tree."""
    reasons: list[str] = []
    stop = False
    repair = False
    for split, report in splits.items():
        dataset = report["dataset"]
        if dataset["rankable_group_count"] < minimum_rankable_groups:
            stop = True
            reasons.append(f"{split} has fewer than {minimum_rankable_groups} rankable groups")
        flags = report["shortcut_flags"]
        if flags.get("reasons"):
            repair = True
            reasons.extend(f"{split}: {reason}" for reason in flags["reasons"])
    if stop:
        code = "REFAV_BRANCH_STOPPED"
        next_action = "Stop the RefAV branch and choose a task with a valid candidate pool."
    elif repair:
        code = "PROTOCOL_REPAIR_REQUIRED"
        next_action = "Do not add patch tokens; design and test a label-independent candidate-pool repair."
    else:
        code = "PROTOCOL_VALIDATION_CONTINUES"
        next_action = "Run the second-seed pooled baseline and confidence-interval check before adding patch tokens."
    return {
        "reasons": reasons,
        "minimum_rankable_groups": minimum_rankable_groups,
        "next_action": next_action,
        # Keep the decision code last so both humans and simple consumers can
        # treat it as the terminal outcome of the diagnosis report.
        "code": code,
    }


__all__ = [
    "COUNT_BINS",
    "CONTROL_NAMES",
    "diagnose_split",
    "decide_report",
    "group_rows",
    "label_value",
    "projected_area",
    "sha256_file",
]
