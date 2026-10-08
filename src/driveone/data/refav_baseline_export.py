"""Export the corrected v6 RefAV pool in the frozen-baseline row format.

The candidate-source audit deliberately stores prompt-independent candidate
rows.  The first learned baseline needs prompt-expanded rows, a fixed camera
image, and ego-frame geometry.  This module performs that conversion without
changing candidate membership or the source pool hash.

The source pool is always checked before prompt labels or derived model fields
are attached.  Unknown labels remain ``None`` and are never converted into
``OTHER_OBJECT``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .refav_candidate_sources import pool_hash


DEFAULT_CAMERA = "ring_front_center"
LABEL_TO_ID = {"REFERRED_OBJECT": 0, "RELATED_OBJECT": 1, "OTHER_OBJECT": 2}


class BaselineExportError(RuntimeError):
    """Raised when the v6 source cannot be converted without ambiguity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_group_seed(seed: int, key: tuple[str, str, int]) -> int:
    """Create a stable per-group shuffle seed independent of Python hash randomization."""
    payload = f"{seed}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def read_groups(path: Path, source: str) -> list[dict[str, Any]]:
    groups = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("source") == source:
                groups.append(value)
    groups.sort(key=lambda item: (str(item["log_id"]), int(item["timestamp_ns"]), str(item["prompt"])))
    return groups


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    """Read one audit Feather file while preserving its source row order."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BaselineExportError("The refav environment must provide pandas and pyarrow") from exc
    frame = pd.read_feather(path)
    required = {
        "log_id", "timestamp_ns", "track_id", "name", "translation_m", "size",
        "rotation", "score", "distance_m", "synthetic_ego", "camera_projections",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BaselineExportError(f"{path} is missing source columns: {missing}")
    return [{key: _json(value) for key, value in row.items()} for row in frame.to_dict("records")]


def load_source_rows(audit_dir: Path, logs: Iterable[str], source: str) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for log_id in sorted(str(value) for value in logs):
        path = audit_dir / f"{source}_{log_id}.feather"
        if not path.is_file():
            raise BaselineExportError(f"Missing source Feather: {path}")
        rows = read_source_rows(path)
        for row in rows:
            if str(row["log_id"]) != log_id:
                raise BaselineExportError(f"Wrong log in {path}: {row['log_id']}")
        output[log_id] = rows
    return output


def _camera_entry(row: Mapping[str, Any], camera_name: str = DEFAULT_CAMERA) -> dict[str, Any]:
    value = row.get("camera_projections")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise BaselineExportError("camera_projections must be a list or JSON list")
    entries = [item for item in value if item.get("camera_name") == camera_name]
    if len(entries) != 1:
        raise BaselineExportError(f"Expected one {camera_name} projection, found {len(entries)}")
    entry = entries[0]
    if not entry.get("image_path"):
        raise BaselineExportError(f"Missing {camera_name} image path")
    if abs(int(entry.get("timestamp_delta_ns", 0))) > 100_000_000:
        raise BaselineExportError(f"{camera_name} frame is outside the 100 ms policy")
    return entry


def _pose_table(sensor_root: Path) -> Any:
    try:
        from .refav_tracker import _load_pose_table
        return _load_pose_table(sensor_root)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BaselineExportError(f"Could not load AV2 poses under {sensor_root}") from exc


def _ego_translation(city_translation: Any, pose_row: Any) -> list[float]:
    try:
        from av2.geometry.se3 import SE3
        from av2.geometry.geometry import quat_to_mat
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BaselineExportError("The refav environment must provide av2") from exc
    quaternion = np.array([[pose_row["qw"], pose_row["qx"], pose_row["qy"], pose_row["qz"]]], dtype=float)
    pose = SE3(
        rotation=quat_to_mat(quaternion)[0],
        translation=np.array([pose_row["tx_m"], pose_row["ty_m"], pose_row["tz_m"]], dtype=float),
    )
    return pose.inverse().transform_from(np.asarray(city_translation, dtype=float)[None])[0].tolist()


def load_tracker_index(path: Path, logs: Iterable[str]) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load numeric tracker labels from the pinned official prediction file."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    wanted = {str(log) for log in logs}
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw_log, frames in payload.items():
        log_id = str(raw_log)
        if log_id not in wanted:
            continue
        for frame in frames:
            timestamp = int(frame["timestamp_ns"])
            fields = ("track_id", "label", "name", "score")
            arrays = {field: np.asarray(frame[field]) for field in fields}
            count = len(arrays["track_id"])
            if any(len(values) != count for values in arrays.values()):
                raise BaselineExportError(f"Tracker field lengths disagree at {log_id}/{timestamp}")
            for index in range(count):
                key = (log_id, timestamp, str(arrays["track_id"][index]))
                item = {
                    "raw_tracker_label": int(arrays["label"][index]),
                    "tracker_name": str(arrays["name"][index]),
                    "tracker_score": float(arrays["score"][index]),
                }
                if key in output and output[key] != item:
                    raise BaselineExportError(f"Duplicate tracker key with different values: {key}")
                output[key] = item
    return output


def _validate_source_group(rows: list[dict[str, Any]], group: Mapping[str, Any]) -> None:
    expected_count = int(group["candidate_count"])
    if len(rows) != expected_count or len(group["labels"]) != expected_count:
        raise BaselineExportError(
            f"Candidate/label length mismatch for {group['log_id']}/{group['timestamp_ns']}/{group['prompt']}: "
            f"rows={len(rows)}, labels={len(group['labels'])}, expected={expected_count}"
        )
    keys = [(str(row["log_id"]), int(row["timestamp_ns"]), str(row["track_id"])) for row in rows]
    if keys != sorted(keys, key=lambda item: item[2]):
        raise BaselineExportError("Source rows are not in the expected track-id order")
    digest = pool_hash(rows)
    if digest != str(group["pool_hash"]):
        raise BaselineExportError(
            f"Pool hash mismatch for {group['log_id']}/{group['timestamp_ns']}: {digest} != {group['pool_hash']}"
        )
    if len(set(keys)) != len(keys):
        raise BaselineExportError("Duplicate candidate key in source group")


def build_baseline_rows(
    *,
    groups: list[dict[str, Any]],
    source_rows: Mapping[str, list[dict[str, Any]]],
    sensor_root: Path,
    tracker_index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    shuffle_seed: int = 0,
    camera_name: str = DEFAULT_CAMERA,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build prompt-expanded baseline rows and source-integrity metadata."""
    pose_cache: dict[str, Any] = {}
    by_time: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for log_id, rows in source_rows.items():
        for row in rows:
            by_time[(log_id, int(row["timestamp_ns"]))].append(row)

    exported: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_manifest = []
    seen_groups: set[tuple[str, str, int]] = set()
    for group in groups:
        log_id = str(group["log_id"])
        timestamp = int(group["timestamp_ns"])
        prompt = str(group["prompt"])
        key = (log_id, prompt, timestamp)
        if key in seen_groups:
            raise BaselineExportError(f"Duplicate source group: {key}")
        seen_groups.add(key)
        rows = list(by_time.get((log_id, timestamp), ()))
        _validate_source_group(rows, group)
        if log_id not in pose_cache:
            pose_cache[log_id] = _pose_table(sensor_root / log_id)
        poses = pose_cache[log_id]
        if timestamp not in poses.index:
            raise BaselineExportError(f"Missing pose for {log_id}/{timestamp}")
        pose = poses.loc[timestamp]
        converted = []
        for source_row, label in zip(rows, group["labels"]):
            track_key = (log_id, timestamp, str(source_row["track_id"]))
            tracker = tracker_index.get(track_key)
            if tracker is None:
                raise BaselineExportError(f"Missing official tracker row: {track_key}")
            if tracker["tracker_name"] != str(source_row["name"]):
                raise BaselineExportError(f"Tracker/source category mismatch: {track_key}")
            if abs(float(tracker["tracker_score"]) - float(source_row["score"])) > 1e-5:
                raise BaselineExportError(f"Tracker/source score mismatch: {track_key}")
            camera = _camera_entry(source_row, camera_name)
            image_path = Path(str(camera["image_path"]))
            if not image_path.is_file():
                raise BaselineExportError(f"Missing camera image: {image_path}")
            value = dict(source_row)
            value.update({
                "prompt": prompt,
                "split": str(group["split"]),
                "prompt_split": str(group["prompt_split"]),
                "joint_holdout_eligible": bool(group["joint_holdout_eligible"]),
                "label": None if label is None else int(label),
                "source_label": None if label is None else int(label),
                "match_status": "UNKNOWN" if label is None else "MATCHED_ANNOTATED",
                "match_distance_m": None,
                "city_translation_m": list(source_row["translation_m"]),
                "translation_m": _ego_translation(source_row["translation_m"], pose),
                "raw_tracker_label": int(tracker["raw_tracker_label"]),
                "projected_box": camera.get("box"),
                "camera_name": camera_name,
                "camera_timestamp_ns": int(camera["image_timestamp_ns"]),
                "camera_delta_ns": int(camera["timestamp_delta_ns"]),
                "image_path": str(image_path),
            })
            converted.append(value)
        rng = np.random.default_rng(stable_group_seed(shuffle_seed, key))
        order = rng.permutation(len(converted))
        shuffled = [converted[int(index)] for index in order]
        exported[str(group["split"])].extend(shuffled)
        group_manifest.append({
            "log_id": log_id,
            "prompt": prompt,
            "timestamp_ns": timestamp,
            "split": str(group["split"]),
            "prompt_split": str(group["prompt_split"]),
            "joint_holdout_eligible": bool(group["joint_holdout_eligible"]),
            "candidate_count": len(rows),
            "pool_hash": str(group["pool_hash"]),
        })
    return dict(exported), {
        "source": "le3de2e_causal",
        "camera_name": camera_name,
        "shuffle_seed": int(shuffle_seed),
        "group_count": len(group_manifest),
        "group_manifest": group_manifest,
        "source_pool_hashes_unchanged": True,
    }


def write_baseline_exports(rows_by_split: Mapping[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    """Write Feather and JSONL split files plus a compact manifest."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BaselineExportError("Writing exports requires pandas and pyarrow") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"splits": {}}
    for split, rows in rows_by_split.items():
        feather = output_dir / f"refav_causal_{split}.feather"
        jsonl = output_dir / f"refav_causal_{split}.jsonl"
        pd.DataFrame.from_records(rows).to_feather(feather)
        with jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json(row), sort_keys=True, separators=(",", ":")) + "\n")
        groups = {(str(r["log_id"]), str(r["prompt"]), int(r["timestamp_ns"])) for r in rows}
        manifest["splits"][split] = {
            "group_count": len(groups),
            "row_count": len(rows),
            "feather": str(feather.resolve()),
            "feather_sha256": sha256_file(feather),
            "jsonl": str(jsonl.resolve()),
            "jsonl_sha256": sha256_file(jsonl),
        }
    (output_dir / "baseline_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["manifest"] = str((output_dir / "baseline_export_manifest.json").resolve())
    return manifest
