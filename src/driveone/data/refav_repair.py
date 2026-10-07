"""Label-independent candidate-pool and target-association helpers for RefAV.

This module is deliberately separate from the first Valeo4Cast adapter.  The
important ordering is:

1. build one candidate pool from tracker rows and infrastructure metadata;
2. hash that pool;
3. associate candidates with ground truth;
4. copy prompt-specific RefAV labels after the association.

The prompt and its relevance label must never affect candidate membership.  The
functions are small and mostly operate on ordinary Python mappings so the
contract can be tested without downloading Argoverse or loading a model.
The optional end-to-end adapter reuses the AV2 projection utilities from
``refav_tracker`` when the ``refav`` environment is installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .refav_tracker import (
    LABEL_TO_ID,
    RING_CAMERAS,
    TrackerPreparationError,
    _camera_index,
    _ego_yaw,
    _require_runtime_dependencies,
    _resolved_image_path,
    _box_corners,
    build_camera_file_index,
    build_camera_models,
    build_roi_map,
    global_to_ego,
    load_tracker_pickle,
    project_candidate,
    roi_mask_for_frame,
)


REPAIR_VERSION = "refav-le3de2e-repair-v1"
MATCH_STATUSES = (
    "MATCHED_ANNOTATED",
    "MATCHED_UNANNOTATED_GT",
    "UNMATCHED_TRACK",
    "AMBIGUOUS_MATCH",
)

# AV2 uses fine-grained tracker and annotation names.  Matching at a stable
# semantic level prevents exact-string mismatches such as BOX_TRUCK versus
# LARGE_VEHICLE from turning valid associations into unknown rows.  The map is
# fixed in code and is independent of prompts and mining labels.
TAXONOMY_GROUPS: dict[str, str] = {
    "REGULAR_VEHICLE": "VEHICLE",
    "LARGE_VEHICLE": "VEHICLE",
    "BOX_TRUCK": "VEHICLE",
    "TRUCK": "VEHICLE",
    "TRUCK_CAB": "VEHICLE",
    "BUS": "VEHICLE",
    "SCHOOL_BUS": "VEHICLE",
    "ARTICULATED_BUS": "VEHICLE",
    "EMERGENCY_VEHICLE": "VEHICLE",
    "VEHICLE": "VEHICLE",
    "BICYCLIST": "BICYCLE",
    "BICYCLE": "BICYCLE",
    "MOTORCYCLIST": "MOTORCYCLE",
    "MOTORCYCLE": "MOTORCYCLE",
    "PEDESTRIAN": "PEDESTRIAN",
    "ANIMAL": "ANIMAL",
    "WHEELED_DEVICE": "WHEELED_DEVICE",
    "WHEELCHAIR": "WHEELED_DEVICE",
    "STROLLER": "WHEELED_DEVICE",
}


def taxonomy_group(name: Any) -> str:
    """Return the fixed matching group for an AV2 class name."""
    normalized = str(name).strip().upper()
    return TAXONOMY_GROUPS.get(normalized, normalized)


def class_compatible(left: Any, right: Any) -> bool:
    """Check compatibility using only the pinned taxonomy map."""
    return taxonomy_group(left) == taxonomy_group(right)


def _finite_vector(value: Any, size: int) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape == (size,) and bool(np.isfinite(array).all())


def _stable_json(value: Any) -> str:
    def jsonable(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return [jsonable(value) for value in item.tolist()]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): jsonable(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(value) for value in item]
        return item
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_pool_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash only candidate identity and construction fields.

    Labels and prompt-specific fields are intentionally excluded.  This makes
    it possible to prove that target association did not change the pool.
    """
    identity_fields = (
        "log_id",
        "timestamp_ns",
        "tracker_timestamp_ns",
        "track_id",
        "candidate_ordinal",
        "name",
        "translation_m",
        "size",
        "yaw",
        "tracker_score",
        "roi_eligible",
    )
    canonical = []
    for row in rows:
        canonical.append({field: row.get(field) for field in identity_fields})
    canonical.sort(key=lambda row: (
        str(row.get("log_id")),
        int(row.get("timestamp_ns", 0)),
        int(row.get("track_id", -1)),
        int(row.get("candidate_ordinal", -1)),
    ))
    return hashlib.sha256((_stable_json(canonical) + "\n").encode("utf-8")).hexdigest()


def validate_candidate_pool(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate identity uniqueness and return reproducibility diagnostics."""
    keys = [
        (str(row.get("log_id")), int(row.get("timestamp_ns")), int(row.get("track_id")))
        for row in rows
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    group_keys = [(key[0], key[1]) for key in keys]
    return {
        "row_count": len(rows),
        "group_count": len(set(group_keys)),
        "duplicate_candidate_keys": [list(key) for key in duplicates],
        "candidate_pool_hash": candidate_pool_hash(rows),
        "has_prompt_dependent_fields": any(
            row.get("prompt") is not None
            or row.get("mining_category") is not None
            or row.get("label") is not None
            for row in rows
        ),
    }


def build_candidate_pool(
    *,
    log_id: str,
    frames: Iterable[Mapping[str, Any]],
    decision_timestamps: Iterable[int],
    ego_positions_by_timestamp: Mapping[int, Sequence[Sequence[float]]] | None = None,
    roi_masks_by_timestamp: Mapping[int, Sequence[bool]] | None = None,
    max_distance_m: float = 50.0,
    supported_groups: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build candidates without reading prompts or relevance labels.

    ``decision_timestamps`` may come from the infrastructure-aligned RefAV
    timestamp list, but it must be independent of the object labels.  The
    function keeps tracker confidence as an audit field; downstream models
    must exclude it from candidate features.
    """
    timestamps = {int(value) for value in decision_timestamps}
    output: list[dict[str, Any]] = []
    required = ("timestamp_ns", "track_id", "score", "label", "translation_m", "size", "name", "yaw")
    for frame in sorted(frames, key=lambda item: int(item["timestamp_ns"])):
        timestamp = int(frame["timestamp_ns"])
        if timestamp not in timestamps:
            continue
        vector_fields = ("track_id", "score", "label", "translation_m", "size", "name", "yaw")
        arrays = {field: np.asarray(frame[field]) for field in vector_fields if field in frame}
        missing = [field for field in required if field not in frame]
        if missing:
            raise TrackerPreparationError(f"Frame {timestamp} is missing {missing}")
        count = len(arrays["track_id"])
        if any(len(array) != count for array in arrays.values()):
            raise TrackerPreparationError(f"Array length mismatch at {log_id}/{timestamp}")
        ego_positions = None
        if ego_positions_by_timestamp is not None:
            ego_positions = np.asarray(ego_positions_by_timestamp[timestamp], dtype=float)
            if ego_positions.shape != (count, 3):
                raise TrackerPreparationError(f"Ego-position shape mismatch at {log_id}/{timestamp}")
        roi_mask = None
        if roi_masks_by_timestamp is not None:
            roi_mask = np.asarray(roi_masks_by_timestamp[timestamp], dtype=bool)
            if roi_mask.shape != (count,):
                raise TrackerPreparationError(f"ROI-mask shape mismatch at {log_id}/{timestamp}")
        for index in range(count):
            name = str(arrays["name"][index])
            if supported_groups is not None and taxonomy_group(name) not in supported_groups:
                continue
            city_position = np.asarray(arrays["translation_m"][index], dtype=float)
            size = np.asarray(arrays["size"][index], dtype=float)
            if not _finite_vector(city_position, 3) or not _finite_vector(size, 3):
                continue
            position = ego_positions[index] if ego_positions is not None else city_position
            distance_m = float(np.linalg.norm(np.asarray(position, dtype=float)[:2]))
            if not math.isfinite(distance_m) or distance_m >= max_distance_m:
                continue
            roi_eligible = True if roi_mask is None else bool(roi_mask[index])
            if not roi_eligible:
                continue
            output.append({
                "log_id": str(log_id),
                "timestamp_ns": timestamp,
                "track_id": int(arrays["track_id"][index]),
                "candidate_ordinal": int(index),
                "name": name,
                "taxonomy_group": taxonomy_group(name),
                "tracker_label": int(arrays["label"][index]),
                "tracker_score": float(arrays["score"][index]),
                # Compatibility alias used by the existing control suite.
                # The repair contract still treats this as audit metadata,
                # never as a model feature.
                "score": float(arrays["score"][index]),
                "translation_m": city_position.tolist(),
                "ego_translation_m": np.asarray(position, dtype=float).tolist(),
                "size": size.tolist(),
                "yaw": float(arrays["yaw"][index]),
                "distance_m": distance_m,
                "roi_eligible": True,
                # Compatibility alias used by the existing verifier.  Rows
                # that reach this function have already passed the fixed ROI
                # filter, so this is not a target label.
                "is_drivable": True,
                "label": None,
                "label_name": None,
                "match_status": "UNMATCHED_TRACK",
            })
    output.sort(key=lambda row: (row["log_id"], row["timestamp_ns"], row["track_id"], row["candidate_ordinal"]))
    return output


def _row_value(row: Any, *names: str) -> Any:
    for name in names:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        if hasattr(row, name):
            return getattr(row, name)
    return None


def match_candidates_compatible(
    candidate_positions_xy: np.ndarray,
    candidate_names: Sequence[Any],
    annotation_rows: Sequence[Any],
    *,
    distance_threshold_m: float = 2.0,
    linear_sum_assignment: Any | None = None,
) -> dict[int, dict[str, Any]]:
    """One-to-one matching with the fixed compatibility taxonomy."""
    if linear_sum_assignment is None:
        _, _, _, _, _, linear_sum_assignment = _require_runtime_dependencies()
    positions = np.asarray(candidate_positions_xy, dtype=float)
    matches: dict[int, dict[str, Any]] = {}
    groups = sorted({taxonomy_group(name) for name in candidate_names})
    for group in groups:
        candidate_indices = [index for index, name in enumerate(candidate_names) if taxonomy_group(name) == group]
        gt_rows = [row for row in annotation_rows if taxonomy_group(_row_value(row, "category", "name")) == group]
        if not candidate_indices or not gt_rows:
            continue
        gt_positions = np.asarray([
            [float(_row_value(row, "tx_m", "x_m", "x")), float(_row_value(row, "ty_m", "y_m", "y"))]
            for row in gt_rows
        ], dtype=float)
        distances = np.linalg.norm(positions[candidate_indices, None, :2] - gt_positions[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for row_index, column_index in zip(rows, columns):
            distance = float(distances[row_index, column_index])
            if distance <= distance_threshold_m:
                candidate_index = int(candidate_indices[row_index])
                matches[candidate_index] = {
                    "gt_track_uuid": str(_row_value(gt_rows[column_index], "track_uuid", "track_id")),
                    "distance_m": distance,
                    "annotation": gt_rows[column_index],
                    "taxonomy_group": group,
                }
    return matches


def attach_prompt_labels(
    candidate_rows: Sequence[Mapping[str, Any]],
    prompt_annotations: Mapping[tuple[str, int], Sequence[Any]],
    all_annotations_by_timestamp: Mapping[tuple[str, int], Sequence[Any]],
    *,
    distance_threshold_m: float = 2.0,
) -> list[dict[str, Any]]:
    """Copy prompt labels after candidate construction and matching."""
    _, _, _, _, _, linear_sum_assignment = _require_runtime_dependencies()
    output: list[dict[str, Any]] = []
    by_group: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_group[(str(row["log_id"]), int(row["timestamp_ns"]))].append(row)
    for (log_id, timestamp), rows in sorted(by_group.items()):
        all_gt = list(all_annotations_by_timestamp.get((log_id, timestamp), ()))
        positions = np.asarray([row["ego_translation_m"] for row in rows], dtype=float)
        matches = match_candidates_compatible(
            positions[:, :2], [row["name"] for row in rows], all_gt,
            distance_threshold_m=distance_threshold_m,
            linear_sum_assignment=linear_sum_assignment,
        )
        prompts = sorted({key[0] for key in prompt_annotations if key[1] == timestamp and key[0] == log_id})
        for prompt in prompts:
            annotated_by_uuid = {
                str(_row_value(annotation, "track_uuid", "track_id")): annotation
                for annotation in prompt_annotations.get((prompt, timestamp), ())
            }
            for index, row in enumerate(rows):
                item = dict(row)
                match = matches.get(index)
                item["prompt"] = prompt
                item["matched_track_uuid"] = match["gt_track_uuid"] if match else None
                item["match_distance_m"] = match["distance_m"] if match else None
                annotation = annotated_by_uuid.get(match["gt_track_uuid"]) if match else None
                if annotation is not None:
                    label_name = str(_row_value(annotation, "mining_category", "label", "refav_label"))
                    item["label_name"] = label_name
                    item["label"] = LABEL_TO_ID.get(label_name)
                    item["match_status"] = "MATCHED_ANNOTATED"
                elif match is not None:
                    item["match_status"] = "MATCHED_UNANNOTATED_GT"
                else:
                    item["match_status"] = "UNMATCHED_TRACK"
                output.append(item)
    return output


def project_candidate_all_cameras(
    center_ego: np.ndarray,
    size_lwh: np.ndarray,
    yaw_ego: float,
    timestamp_ns: int,
    camera_models: Mapping[str, tuple[Any, Any]],
    camera_files: Mapping[str, tuple[list[int], list[Path]]],
    *,
    camera_tolerance_ns: int = 100_000_000,
) -> list[dict[str, Any]]:
    """Project one candidate into all seven cameras in fixed order."""
    projections: list[dict[str, Any]] = []
    for camera_name in RING_CAMERAS:
        projection = project_candidate(
            center_ego,
            size_lwh,
            yaw_ego,
            timestamp_ns,
            Path("."),
            camera_models,
            camera_files,
            camera_name=camera_name,
            camera_tolerance_ns=camera_tolerance_ns,
        )
        projections.append({
            "camera_name": camera_name,
            "image_path": projection.image_path,
            "image_timestamp_ns": projection.image_timestamp_ns,
            "timestamp_delta_ns": projection.timestamp_delta_ns,
            "projected_box": list(projection.projected_box) if projection.projected_box else None,
            "status": projection.status,
        })
    return projections


def projection_summary(projections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize fixed-order camera projections without choosing a best view."""
    return {
        "camera_count": len(projections),
        "projected_camera_count": sum(item.get("status") == "PROJECTED" for item in projections),
        "available_camera_count": sum(item.get("status") != "MISSING_SHARED_CAMERA_FRAME" for item in projections),
        "any_projected": any(item.get("status") == "PROJECTED" for item in projections),
        "cameras": list(projections),
    }


def provenance_for_paths(paths: Iterable[Path], *, source_url: str, source_revision: str, license_name: str) -> dict[str, Any]:
    """Record reproducibility metadata for a repaired manifest."""
    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()
    return {
        "repair_version": REPAIR_VERSION,
        "source_url": source_url,
        "source_revision": source_revision,
        "license": license_name,
        "files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest(path)}
            for path in paths
        ],
    }
