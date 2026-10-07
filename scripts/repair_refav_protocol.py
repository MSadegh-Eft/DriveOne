#!/usr/bin/env python3
"""Build the official, label-independent RefAV tracker candidate pilot.

The script consumes the official RefAV scenario-mining annotations, the
official Le3DE2E tracker pickle, and a small set of local AV2 sensor logs.  It
constructs candidates before looking at prompt-specific relevance labels.  It
then performs fixed taxonomy-compatible matching and attaches labels as a
separate step.

This is a protocol audit tool.  It does not train a model or alter the older
Valeo4Cast artifacts.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_repair import (  # noqa: E402
    REPAIR_VERSION,
    LABEL_TO_ID,
    build_candidate_pool,
    candidate_pool_hash,
    match_candidates_compatible,
    project_candidate_all_cameras,
    projection_summary,
    provenance_for_paths,
    validate_candidate_pool,
)
from driveone.data.refav_tracker import (  # noqa: E402
    _ego_yaw,
    _camera_index,
    _load_pose_table,
    _require_runtime_dependencies,
    build_camera_file_index,
    build_camera_models,
    build_roi_map,
    global_to_ego,
    load_tracker_pickle,
    roi_mask_for_frame,
)


TRACKER_URL = "https://huggingface.co/datasets/CainanD/AV2_Tracker_Predictions"
TRACKER_FILE_URL = TRACKER_URL + "/resolve/main/Le3DE2E_tracking_predictions_val.pkl"
ANNOTATION_URL = "https://huggingface.co/datasets/CainanD/RefAV"
ANNOTATION_FILE_URL = ANNOTATION_URL + "/resolve/main/scenario_mining_val_annotations.feather"
TRACKER_REVISION = "d983c7955b1c6a126542bea0a4dfb3a2c7327e6a"
TRACKER_EXPECTED_SHA256 = "fd702ade8b640d325e5096e90f63cf1bf43f94a87a6aabfb52e71aa7b8470875"
ANNOTATION_EXPECTED_SHA256 = "e461e51057fdf347a11bdd60609e7de0b0bd8d0eb199314b3fd73c50106d48c9"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _dedupe_gt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one geometry row per UUID for matching, deterministically."""
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: str(item.get("track_uuid", ""))):
        by_uuid.setdefault(str(row["track_uuid"]), row)
    return list(by_uuid.values())


def _prepare_log(*, log_id: str, tracker_frames: list[dict[str, Any]], annotations: Any, sensor_root: Path,
                 distance_threshold_m: float, camera_tolerance_ns: int, tracker_tolerance_ns: int,
                 progress: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Construct one prompt-independent pool and its prompt-labelled view."""
    pd, _, _, quat_to_mat, SE3, linear_sum_assignment = _require_runtime_dependencies()
    pose_table = _load_pose_table(sensor_root)
    camera_models = build_camera_models(sensor_root)
    camera_files = build_camera_file_index(sensor_root)
    roi_map = build_roi_map(sensor_root)
    log_annotations = annotations[annotations["log_id"] == log_id].copy()
    timestamp_values = sorted({int(value) for value in log_annotations["timestamp_ns"].tolist()})
    tracker_frames = sorted(tracker_frames, key=lambda frame: int(frame["timestamp_ns"]))
    tracker_timestamps = [int(frame["timestamp_ns"]) for frame in tracker_frames]
    aligned_specs: list[tuple[int, int, dict[str, Any]]] = []
    for decision_timestamp in timestamp_values:
        if decision_timestamp not in pose_table.index:
            continue
        insertion = bisect.bisect_left(tracker_timestamps, decision_timestamp)
        nearby = []
        if insertion < len(tracker_frames):
            nearby.append((abs(tracker_timestamps[insertion] - decision_timestamp), insertion))
        if insertion > 0:
            nearby.append((abs(tracker_timestamps[insertion - 1] - decision_timestamp), insertion - 1))
        if not nearby:
            continue
        delta, frame_index = min(nearby, key=lambda item: (item[0], item[1]))
        if delta > tracker_tolerance_ns:
            continue
        if not any(
            _camera_index(camera_files, decision_timestamp, camera_name, camera_tolerance_ns) is not None
            for camera_name in ("ring_front_center", "ring_front_left", "ring_front_right",
                                "ring_rear_left", "ring_rear_right", "ring_side_left", "ring_side_right")
        ):
            continue
        aligned_specs.append((decision_timestamp, tracker_timestamps[frame_index], tracker_frames[frame_index]))
    base_pool: list[dict[str, Any]] = []
    projection_counts = defaultdict(int)
    for index, (timestamp, tracker_timestamp, frame) in enumerate(aligned_specs, start=1):
        pose_row = pose_table.loc[timestamp]
        city_positions = np.asarray(frame["translation_m"], dtype=float)
        ego_positions = global_to_ego(city_positions, pose_row, SE3, quat_to_mat)
        roi_mask = roi_mask_for_frame(frame, roi_map)
        rows = build_candidate_pool(
            log_id=log_id,
            frames=[frame],
            decision_timestamps=[tracker_timestamp],
            ego_positions_by_timestamp={tracker_timestamp: ego_positions},
            roi_masks_by_timestamp={tracker_timestamp: roi_mask},
            max_distance_m=50.0,
        )
        ego_yaw = _ego_yaw(pose_row, quat_to_mat)
        for row in rows:
            candidate_index = int(row["candidate_ordinal"])
            row["timestamp_ns"] = timestamp
            projections = project_candidate_all_cameras(
                ego_positions[candidate_index],
                np.asarray(frame["size"])[candidate_index],
                float(frame["yaw"][candidate_index]) - ego_yaw,
                timestamp,
                camera_models,
                camera_files,
                camera_tolerance_ns=camera_tolerance_ns,
            )
            projection = projection_summary(projections)
            row["yaw"] = float(frame["yaw"][candidate_index]) - ego_yaw
            row["tracker_timestamp_ns"] = tracker_timestamp
            row["camera_projections"] = projection["cameras"]
            row["projected_camera_count"] = projection["projected_camera_count"]
            row["available_camera_count"] = projection["available_camera_count"]
            row["any_projected"] = projection["any_projected"]
            row["projection_status"] = "PROJECTED" if projection["any_projected"] else (
                "OUT_OF_VIEW" if projection["available_camera_count"] else "MISSING_CAMERA_FRAMES"
            )
            projection_counts[row["projection_status"]] += 1
            # Compatibility fields are fixed to front-center only for old
            # readers.  The authoritative input is camera_projections.
            front = next(item for item in projections if item["camera_name"] == "ring_front_center")
            row["camera_name"] = "ring_front_center"
            row["image_path"] = front["image_path"]
            row["camera_timestamp_ns"] = front["image_timestamp_ns"]
            row["camera_delta_ns"] = front["timestamp_delta_ns"]
            row["projected_box"] = front["projected_box"]
            base_pool.append(row)
        if progress and (index == 1 or index % 50 == 0):
            progress(f"{log_id}: {index}/{len(aligned_specs)} timestamps; {len(base_pool)} candidates")

    pool_validation = validate_candidate_pool(base_pool)
    if pool_validation["duplicate_candidate_keys"]:
        raise RuntimeError(f"Candidate pool has duplicate keys for {log_id}")

    prompt_annotations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_annotations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in log_annotations.to_dict(orient="records"):
        key = (str(record["prompt"]), int(record["timestamp_ns"]))
        prompt_annotations[key].append(record)
        all_annotations[(log_id, int(record["timestamp_ns"]))].append(record)
    all_annotations = {key: _dedupe_gt(value) for key, value in all_annotations.items()}
    # attach_prompt_labels expects the per-prompt dictionary keyed by
    # (prompt,timestamp); make it log-local and then add the log to each row.
    labelled = []
    sensitivity = {
        str(threshold): {"group_count": 0, "groups_with_referred": 0, "matched_candidate_count": 0}
        for threshold in (1.0, 2.0, 4.0)
    }
    by_timestamp = defaultdict(list)
    for row in base_pool:
        by_timestamp[(log_id, int(row["timestamp_ns"]))].append(row)
    for (prompt, timestamp), prompt_rows in sorted(prompt_annotations.items()):
        candidates = by_timestamp.get((log_id, timestamp), [])
        if not candidates:
            continue
        gt_rows = all_annotations.get((log_id, timestamp), [])
        positions = np.asarray([row["ego_translation_m"] for row in candidates], dtype=float)
        for threshold in (1.0, 2.0, 4.0):
            threshold_matches = match_candidates_compatible(
                positions[:, :2], [row["name"] for row in candidates], gt_rows,
                distance_threshold_m=threshold,
                linear_sum_assignment=linear_sum_assignment,
            )
            stats = sensitivity[str(threshold)]
            stats["group_count"] += 1
            stats["matched_candidate_count"] += len(threshold_matches)
            matched_ids = {value["gt_track_uuid"] for value in threshold_matches.values()}
            if any(str(row["track_uuid"]) in matched_ids and row.get("mining_category") == "REFERRED_OBJECT" for row in prompt_rows):
                stats["groups_with_referred"] += 1
        matches = match_candidates_compatible(
            positions[:, :2], [row["name"] for row in candidates], gt_rows,
            distance_threshold_m=distance_threshold_m,
            linear_sum_assignment=linear_sum_assignment,
        )
        prompt_by_uuid = {str(row["track_uuid"]): row for row in prompt_rows}
        for candidate_index, candidate in enumerate(candidates):
            item = dict(candidate)
            item["prompt"] = prompt
            match = matches.get(candidate_index)
            item["matched_track_uuid"] = match["gt_track_uuid"] if match else None
            item["match_distance_m"] = match["distance_m"] if match else None
            annotation = prompt_by_uuid.get(match["gt_track_uuid"]) if match else None
            if annotation is None:
                item["label"] = None
                item["label_name"] = None
                item["match_status"] = "MATCHED_UNANNOTATED_GT" if match else "UNMATCHED_TRACK"
            else:
                label_name = str(annotation.get("mining_category"))
                item["label_name"] = label_name
                item["label"] = LABEL_TO_ID.get(label_name)
                item["match_status"] = "MATCHED_ANNOTATED"
            labelled.append(item)
    summary = {
        "log_id": log_id,
        "annotation_prompt_count": int(log_annotations["prompt"].nunique()),
        "annotation_timestamp_count": len(timestamp_values),
        "aligned_timestamp_count": len(aligned_specs),
        "tracker_timestamp_tolerance_ms": tracker_tolerance_ns / 1_000_000,
        "base_candidate_count": len(base_pool),
        "candidate_pool_hash": candidate_pool_hash(base_pool),
        "pool_validation": pool_validation,
        "projection_status_counts": dict(sorted(projection_counts.items())),
        "labelled_row_count": len(labelled),
        "labelled_positive_count": sum(row.get("label") == 0 for row in labelled),
        "labelled_negative_count": sum(row.get("label") in {1, 2} for row in labelled),
        "unknown_row_count": sum(row.get("label") is None for row in labelled),
        "matching_sensitivity": sensitivity,
        "positive_projection_rate": (
            sum(row.get("label") == 0 and row.get("any_projected") for row in labelled)
            / max(1, sum(row.get("label") == 0 for row in labelled))
        ),
    }
    return base_pool, labelled, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True, help="AV2 val directory containing log subdirectories")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Labelled Feather output")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    parser.add_argument("--camera-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--tracker-tolerance-ms", type=float, default=250.0)
    parser.add_argument("--source-revision", default=TRACKER_REVISION)
    args = parser.parse_args()

    pd, *_ = _require_runtime_dependencies()
    if not args.tracker.is_file():
        raise SystemExit(f"Tracker file does not exist: {args.tracker}")
    if not args.annotations.is_file():
        raise SystemExit(f"Annotation file does not exist: {args.annotations}")
    required = {"log_id", "prompt", "timestamp_ns", "track_uuid", "mining_category", "category", "tx_m", "ty_m"}
    # Read only the columns needed for candidate matching and label transfer;
    # the official validation table has 17M rows and contains extra fields
    # (quaternions, dimensions, point counts) that are not used here.
    annotations = pd.read_feather(args.annotations, columns=sorted(required))
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise SystemExit(f"Official scenario-mining annotations miss required columns: {missing}")
    selected_logs = list(dict.fromkeys(args.logs))
    available_logs = set(str(value) for value in annotations["log_id"].unique())
    missing_logs = sorted(set(selected_logs).difference(available_logs))
    if missing_logs:
        raise SystemExit(f"Requested logs are absent from official annotations: {missing_logs}")
    tracker_logs = load_tracker_pickle(args.tracker, selected_logs)
    missing_tracker_logs = sorted(set(selected_logs).difference(tracker_logs))
    if missing_tracker_logs:
        raise SystemExit(f"Requested logs are absent from official tracker: {missing_tracker_logs}")

    all_base: list[dict[str, Any]] = []
    all_labelled: list[dict[str, Any]] = []
    per_log: dict[str, Any] = {}
    tolerance_ns = int(round(args.camera_tolerance_ms * 1_000_000))
    for log_id in selected_logs:
        sensor_root = args.sensor_root / log_id
        if not sensor_root.is_dir():
            raise SystemExit(f"Sensor log directory does not exist: {sensor_root}")
        base, labelled, summary = _prepare_log(
            log_id=log_id,
            tracker_frames=tracker_logs[log_id],
            annotations=annotations,
            sensor_root=sensor_root,
            distance_threshold_m=args.distance_threshold_m,
            camera_tolerance_ns=tolerance_ns,
            tracker_tolerance_ns=int(round(args.tracker_tolerance_ms * 1_000_000)),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        all_base.extend(base)
        all_labelled.extend(labelled)
        per_log[log_id] = summary

    pool_validation = validate_candidate_pool(all_base)
    if pool_validation["duplicate_candidate_keys"]:
        raise SystemExit("The combined candidate pool contains duplicate keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(all_labelled).to_feather(args.output)
    sensitivity_global = {}
    for threshold in (1.0, 2.0, 4.0):
        key = str(threshold)
        group_count = sum(item["matching_sensitivity"][key]["group_count"] for item in per_log.values())
        referred_count = sum(item["matching_sensitivity"][key]["groups_with_referred"] for item in per_log.values())
        matched_count = sum(item["matching_sensitivity"][key]["matched_candidate_count"] for item in per_log.values())
        sensitivity_global[key] = {
            "group_count": group_count,
            "groups_with_referred": referred_count,
            "positive_availability_rate": referred_count / max(1, group_count),
            "matched_candidate_count": matched_count,
        }
    artifact_hash = provenance_for_paths(
        [args.tracker, args.annotations],
        source_url=f"{TRACKER_FILE_URL}; {ANNOTATION_FILE_URL}",
        source_revision=args.source_revision,
        license_name="RefAV repository/data terms; Hugging Face tracker card MIT; Argoverse 2 terms apply",
    )
    summary = {
        "repair_version": REPAIR_VERSION,
        "candidate_pool_contract": "candidate construction precedes target association; prompt and mining labels are excluded",
        "logs": selected_logs,
        "per_log": per_log,
        "base_pool": pool_validation,
        "labelled_row_count": len(all_labelled),
        "positive_row_count": sum(row.get("label") == 0 for row in all_labelled),
        "negative_row_count": sum(row.get("label") in {1, 2} for row in all_labelled),
        "unknown_row_count": sum(row.get("label") is None for row in all_labelled),
        "positive_availability": {
            "group_count": len({(row["log_id"], row["prompt"], row["timestamp_ns"]) for row in all_labelled}),
            "groups_with_referred": len({
                (row["log_id"], row["prompt"], row["timestamp_ns"])
                for row in all_labelled if row.get("label") == 0
            }),
        },
        "positive_projection_rate": (
            sum(row.get("label") == 0 and row.get("any_projected") for row in all_labelled)
            / max(1, sum(row.get("label") == 0 for row in all_labelled))
        ),
        "matching_sensitivity": sensitivity_global,
        "camera_policy": {
            "cameras": ["ring_front_center", "ring_front_left", "ring_front_right", "ring_rear_left", "ring_rear_right", "ring_side_left", "ring_side_right"],
            "nearest_frame_tolerance_ms": args.camera_tolerance_ms,
            "tracker_nearest_frame_tolerance_ms": args.tracker_tolerance_ms,
            "out_of_view_rows_retained": True,
            "best_camera_selection": False,
        },
        "matching_policy": {
            "taxonomy": "fixed semantic compatibility map",
            "assignment": "one-to-one Hungarian",
            "threshold_m": args.distance_threshold_m,
            "sensitivity_thresholds_m": [1.0, 2.0, 4.0],
            "unknown_statuses": ["UNMATCHED_TRACK", "MATCHED_UNANNOTATED_GT", "AMBIGUOUS_MATCH"],
        },
        "provenance": artifact_hash,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "repair_version": REPAIR_VERSION,
        "output_path": str(args.output.resolve()),
        "output_sha256": artifact_hash_for_file(args.output),
        "candidate_pool_hash": pool_validation["candidate_pool_hash"],
        "source": artifact_hash,
        "logs": selected_logs,
        "summary_path": str(args.summary.resolve()),
        "label_independent_candidate_construction": True,
        "candidate_count_uses_prompt_labels": False,
        "candidate_ids_use_match_status": False,
        "matching_sensitivity_complete": True,
        "positive_availability": summary["positive_availability"]["groups_with_referred"] / max(1, summary["positive_availability"]["group_count"]),
        "positive_projection_rate": summary["positive_projection_rate"],
        "matching_sensitivity": summary["matching_sensitivity"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_value(manifest), indent=2, sort_keys=True))
    return 0


def artifact_hash_for_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
