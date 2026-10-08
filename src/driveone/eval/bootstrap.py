"""Paired group bootstrap utilities for RefAV baseline comparisons."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def paired_bootstrap(
    left: Iterable[float],
    right: Iterable[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the paired mean difference ``left - right``.

    Inputs are already matched by group key.  Missing metrics must be removed
    before calling this function because a group without a positive or a
    negative has no defined AP/Recall value.
    """
    left_values = np.asarray(list(left), dtype=float)
    right_values = np.asarray(list(right), dtype=float)
    if left_values.shape != right_values.shape:
        raise ValueError("Paired bootstrap inputs must have the same length")
    if left_values.ndim != 1 or len(left_values) == 0:
        raise ValueError("Paired bootstrap requires at least one value")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap sample or confidence settings")
    differences = left_values - right_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "group_count": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "confidence": float(confidence),
        "confidence_interval": [float(low), float(high)],
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def paired_metric_bootstrap(
    left_groups: Iterable[Mapping[str, Any]],
    right_groups: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Join per-group records and bootstrap one metric difference."""
    key_fields = ("log_id", "prompt", "timestamp_ns")

    def key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        return (str(row["log_id"]), str(row["prompt"]), int(row["timestamp_ns"]))

    left_map = {key(row): row for row in left_groups}
    right_map = {key(row): row for row in right_groups}
    shared = sorted(set(left_map) & set(right_map))
    pairs = [
        (left_map[item].get(metric), right_map[item].get(metric))
        for item in shared
        if left_map[item].get(metric) is not None and right_map[item].get(metric) is not None
    ]
    if not pairs:
        raise ValueError(f"No paired groups have metric {metric!r}")
    result = paired_bootstrap(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs],
        samples=samples, seed=seed,
    )
    result["metric"] = metric
    result["shared_group_count"] = len(shared)
    return result
