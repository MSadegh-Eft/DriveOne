"""Candidate-source audit helpers; no learned model or target-based filtering.

An event can be absent at a timestamp. Event frequency and candidate recall
therefore have different denominators. Missing image files are infrastructure
failures, not evidence that an object is invisible. Projection here measures
geometric image intersection, not occlusion or human-visible evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .refav_repair import taxonomy_group
from .refav_tracker import RING_CAMERAS

AUDIT_VERSION = "refav-candidate-sources-v1"
THRESHOLDS_M = (1.0, 2.0, 4.0)


def pool_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    """Canonical pool hash includes geometry, never labels or prompts."""
    fields = ("log_id", "timestamp_ns", "track_id", "name", "translation_m", "size", "rotation", "score")
    values = [{k: row.get(k) for k in fields} for row in rows]
    values.sort(key=lambda r: (r["log_id"], int(r["timestamp_ns"]), str(r["track_id"])))
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_pool(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [(r["log_id"], int(r["timestamp_ns"]), str(r["track_id"])) for r in rows]
    counts = Counter(keys)
    return {"row_count": len(rows), "duplicate_keys": [list(k) for k, n in sorted(counts.items()) if n > 1], "sha256": pool_hash(rows)}


def validate_splits(splits: Mapping[str, Sequence[str]]) -> None:
    owners: dict[str, str] = {}
    for split, logs in splits.items():
        for log in logs:
            if log in owners:
                raise ValueError(f"Log {log} repeats in {owners[log]} and {split}")
            owners[log] = split


def count_bin(count: int) -> str:
    if count < 0:
        raise ValueError("Candidate count must be non-negative")
    if count == 0:
        return "0"
    if count <= 10:
        return "1-10"
    if count <= 50:
        return "11-50"
    if count <= 100:
        return "51-100"
    return "101+"


def build_pool(
    *, log_id: str, timestamp_ns: int, ids: Sequence[Any], names: Sequence[Any],
    centers_city: np.ndarray, sizes: np.ndarray, rotations_city: np.ndarray,
    scores: Sequence[float], city_se3_ego: Any, roi: Sequence[bool],
    selected_ids: set[str] | None = None, include_ego: bool = False,
) -> list[dict[str, Any]]:
    """Fixed finite/positive-size/50m/ROI filters, with no label inputs.

    Raw tracker class and confidence are audit fields only. Confidence is not
    used to choose candidates here. ``selected_ids`` is the official whole-log
    filter replay and must never be called causal. Ego injection is separately
    recorded and is not a camera-visible object proposal.
    """
    count = len(ids)
    if any(len(x) != count for x in (names, centers_city, sizes, rotations_city, scores, roi)):
        raise ValueError("Candidate array lengths differ")
    ego_positions = city_se3_ego.inverse().transform_from(np.asarray(centers_city, float))
    output = []
    for i in range(count):
        position = np.asarray(centers_city[i], float)
        size = np.asarray(sizes[i], float)
        rotation = np.asarray(rotations_city[i], float)
        if selected_ids is not None and str(ids[i]) not in selected_ids:
            continue
        if not (np.isfinite(position).all() and np.isfinite(size).all() and np.isfinite(rotation).all()):
            continue
        if not np.all(size > 0) or not np.isfinite(scores[i]):
            continue
        if np.linalg.norm(ego_positions[i, :2]) >= 50.0 or not roi[i]:
            continue
        output.append({
            "log_id": log_id, "timestamp_ns": int(timestamp_ns), "track_id": str(ids[i]),
            "name": str(names[i]), "translation_m": position.tolist(), "size": size.tolist(),
            "rotation": rotation.tolist(), "score": float(scores[i]),
            "distance_m": float(np.linalg.norm(ego_positions[i, :2])), "synthetic_ego": False,
        })
    if include_ego:
        output.append({"log_id": log_id, "timestamp_ns": int(timestamp_ns), "track_id": "ego",
                       "name": "EGO_VEHICLE", "translation_m": city_se3_ego.translation.tolist(),
                       "size": [4.877, 2.0, 1.473], "rotation": city_se3_ego.rotation.tolist(),
                       "score": 1.0, "distance_m": 0.0, "synthetic_ego": True})
    output.sort(key=lambda row: row["track_id"])
    return output


def associate(
    candidates: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]],
    *, threshold_m: float = 2.0, compatible: bool = True,
) -> dict[int, dict[str, Any]]:
    """Maximum-cardinality threshold-gated one-to-one XY association.

    Dummy columns permit unmatched rows. A large fixed penalty first maximizes
    the number of valid edges, then minimizes total distance. Matching after
    an ungated Hungarian assignment can otherwise reject a valid association.
    Ambiguity is recorded if either side has multiple valid geometric edges;
    it does not alter pool membership.
    """
    from scipy.optimize import linear_sum_assignment

    if not candidates or not gt:
        return {}
    left = np.asarray([r["translation_m"][:2] for r in candidates])
    right = np.asarray([r["translation_m"][:2] for r in gt])
    distances = np.linalg.norm(left[:, None] - right[None], axis=2)
    group = taxonomy_group if compatible else str
    allowed_class = np.asarray([[group(a["name"]) == group(b["name"]) for b in gt] for a in candidates])
    valid = (distances <= threshold_m) & allowed_class
    penalty = (max(len(candidates), len(gt)) + 1) * (threshold_m + 1)
    costs = np.full((len(candidates), len(gt) + len(candidates)), penalty)
    costs[:, :len(gt)] = np.where(valid, distances, penalty * 3)
    rows, columns = linear_sum_assignment(costs)
    return {int(i): {"gt_index": int(j), "distance_m": float(distances[i, j]),
                     "ambiguous": bool(valid[i].sum() > 1 or valid[:, j].sum() > 1)}
            for i, j in zip(rows, columns) if j < len(gt) and valid[i, j]}


def box_corners(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 8, 3))
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    local = np.asarray([r["size"] for r in rows])[:, None] * signs[None] / 2
    rotations = np.asarray([r["rotation"] for r in rows])
    return np.einsum("nij,nkj->nki", rotations, local) + np.asarray([r["translation_m"] for r in rows])[:, None]


def project_pool(
    rows: Sequence[Mapping[str, Any]], camera_models: Mapping[str, Any],
    camera_files: Mapping[str, Any], poses: Mapping[int, Any], *, tolerance_ns: int = 100_000_000,
) -> list[list[dict[str, Any]]]:
    """Fixed-order seven-camera projections at each image's actual ego pose.

    Boxes are stationary in city coordinates during the <=100ms image offset;
    only ego motion is compensated. No object-motion interpolation is assumed.
    Cuboid edges crossing the camera near plane are clipped before projection.
    This is a geometric projection test; it does not measure occlusion.
    """
    from .refav_tracker import _camera_index

    output: list[list[dict[str, Any]]] = [[] for _ in rows]
    if not rows:
        return output
    corners_city = box_corners(rows)
    timestamp = int(rows[0]["timestamp_ns"])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
    for name in RING_CAMERAS:
        entry = _camera_index(camera_files, timestamp, name, tolerance_ns)
        camera = camera_models.get(name)
        if isinstance(camera, tuple):
            camera = camera[0]
        if entry is None or camera is None or entry[1] not in poses:
            for out in output:
                out.append({"camera_name": name, "status": "MISSING_CAMERA_ASSET", "box": None})
            continue
        path, image_ts = entry
        corners_ego = poses[image_ts].inverse().transform_from(corners_city.reshape(-1, 3))
        points = camera.ego_SE3_cam.inverse().transform_from(corners_ego).reshape(-1, 8, 3)
        intr = camera.intrinsics
        for i, cuboid in enumerate(points):
            near = 0.1
            clipped = [p for p in cuboid if p[2] >= near]
            for a, b in edges:
                p, q = cuboid[a], cuboid[b]
                if (p[2] < near) != (q[2] < near):
                    clipped.append(p + (q - p) * ((near - p[2]) / (q[2] - p[2])))
            bounds = None
            if clipped:
                pts = np.asarray(clipped)
                uv = (intr.K @ pts.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                low = np.maximum(uv.min(axis=0), [0, 0])
                high = np.minimum(uv.max(axis=0), [intr.width_px, intr.height_px])
                if np.prod(np.maximum(0, high - low)) > 1.0:
                    bounds = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
            # The synthetic ego box is never visible external-object evidence.
            if rows[i].get("synthetic_ego"):
                bounds = None
            output[i].append({"camera_name": name, "status": "PROJECTED" if bounds else "OUT_OF_VIEW",
                              "box": bounds, "image_path": str(path), "image_timestamp_ns": int(image_ts),
                              "timestamp_delta_ns": abs(int(image_ts) - timestamp)})
    return output


def camera_summary(projections: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {"row_count": len(projections),
            "any_projected": sum(any(p["status"] == "PROJECTED" for p in row) for row in projections),
            "all_seven_assets": sum(all(p["status"] != "MISSING_CAMERA_ASSET" for p in row) for row in projections),
            "camera_order": list(RING_CAMERAS)}


def coverage(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # RefAV contains scenario prompts whose referred object is the ego car.
    # That is a separate scenario-level task for this project: an external
    # referred-track candidate must have a camera-renderable object box.
    positive = [g for g in groups if g["gt_positive_count"] > 0]
    eligible = [g for g in groups if g.get("eligible_external_positive_count", g["eligible_positive_count"]) > 0]
    external_positive = [g for g in groups if g.get("gt_external_positive_count", g["gt_positive_count"]) > 0]
    external_eligible = [g for g in groups if g.get("eligible_external_positive_count", g["eligible_positive_count"]) > 0]
    matched_external = lambda g: g.get("matched_external_positive_count", g["matched_positive_count"])
    projected_external = lambda g: g.get("projected_external_positive_count", g["projected_positive_count"])
    allseven_external = lambda g: g.get("projected_external_positive_count_all_seven_assets", g.get("positive_count_all_seven_assets", 0))
    matched_total = sum(matched_external(g) for g in groups)
    return {
        "group_count": len(groups), "gt_positive_group_count": len(positive),
        "event_frequency": len(positive) / len(groups) if groups else None,
        "gt_external_positive_group_count": len(external_positive),
        "external_event_frequency": len(external_positive) / len(groups) if groups else None,
        "eligible_positive_group_count": len(eligible),
        "positive_availability_all_groups": sum(matched_external(g) > 0 for g in groups) / len(groups) if groups else None,
        "positive_availability_given_eligible_gt_positive": sum(matched_external(g) > 0 for g in eligible) / len(eligible) if eligible else None,
        "eligible_positive_object_count": sum(g.get("eligible_external_positive_count", g["eligible_positive_count"]) for g in groups),
        "matched_positive_object_count": matched_total,
        "positive_projection_rate_downloaded_cameras": sum(projected_external(g) for g in groups) / matched_total if matched_total else None,
        "unknown_candidate_count": sum(g["unknown_count"] for g in groups),
        "candidate_row_count": sum(g["candidate_count"] for g in groups),
        "unknown_fraction": sum(g["unknown_count"] for g in groups) / sum(g["candidate_count"] for g in groups) if sum(g["candidate_count"] for g in groups) else None,
        "zero_candidate_groups": sum(g["candidate_count"] == 0 for g in groups),
        "ego_positive_group_count": sum(g.get("ego_positive_count", 0) > 0 for g in groups),
        "ego_positive_object_count": sum(g.get("ego_positive_count", 0) for g in groups),
        "matched_positive_count_all_seven_assets": sum(allseven_external(g) for g in groups),
    }


def choose_decision(*, infrastructure_complete: bool, deployable_pass: bool, oracle_pass: bool | None,
                    shortcut_dominated: bool = False) -> tuple[str, str]:
    """Incomplete downloads never establish that the dataset itself failed."""
    if deployable_pass and not shortcut_dominated and infrastructure_complete:
        return "OFFICIAL_PROTOCOL_REPAIRED", "Candidate, coverage, camera and control gates pass; model gate remains separate."
    if not infrastructure_complete:
        return "ALTERNATE_CANDIDATE_SOURCE_REQUIRED", "Audit is incomplete: missing camera assets prevent the oracle/visual gate; do not stop RefAV on this evidence."
    if shortcut_dominated or oracle_pass is False:
        return "REFAV_BRANCH_STOPPED", "Complete-input oracle/shortcut gate fails."
    if oracle_pass:
        return "REFAV_ORACLE_ONLY", "Oracle pool passes but no evaluated independent causal source passes."
    return "ALTERNATE_CANDIDATE_SOURCE_REQUIRED", "A valid independent source or evaluable target interface is still required."
