"""Dependency-light RefAV data contract and audit helpers.

The official RefAV submission format is a multi-object, multi-timestamp track
format.  These helpers intentionally do not turn it into a single-label
classification dataset.  They produce diagnostics for a custom fixed-frame
track-ranking study and preserve the official fields needed for later HOTA and
balanced-accuracy evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REQUIRED_FIELDS = (
    "log_id",
    "prompt",
    "timestamp_ns",
    "track_id",
    "score",
    "label",
    "translation_m",
    "size",
    "yaw",
)
LABEL_NAMES = {0: "REFERRED_OBJECT", 1: "RELATED_OBJECT", 2: "OTHER_OBJECT"}
LABEL_ALIASES = {
    "REFERRED_OBJECT": 0,
    "REFERRED": 0,
    "RELATED_OBJECT": 1,
    "RELATED": 1,
    "OTHER_OBJECT": 2,
    "OTHER": 2,
}
ALIASES = {
    "seq_id": "log_id",
    "timestamp": "timestamp_ns",
    "time_ns": "timestamp_ns",
    "track": "track_id",
    "confidence": "score",
    "object_label": "label",
    "translation": "translation_m",
    "position": "translation_m",
    "extent": "size",
    "heading": "yaw",
    "track_uuid": "track_id",
    "mining_category": "label",
}
CAMERA_FIELDS = (
    "camera_frame",
    "camera",
    "camera_name",
    "frame_id",
    "image_path",
    "image",
    "image_uri",
    "camera_timestamp_ns",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_label(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.upper() in LABEL_ALIASES:
            return LABEL_ALIASES[stripped.upper()]
        if stripped.isdigit():
            return int(stripped)
    return value


def _is_missing(value: Any) -> bool:
    """Scalar missingness check that is safe for numpy/list-valued fields."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one flat track record without discarding unknown fields."""
    output: dict[str, Any] = dict(record)
    for old, new in ALIASES.items():
        if new not in output and old in output:
            output[new] = output[old]
    if "label" in output:
        output["label"] = _coerce_label(output["label"])
    if "timestamp_ns" in output:
        try:
            output["timestamp_ns"] = int(output["timestamp_ns"])
        except (TypeError, ValueError):
            pass
    return output


def _flatten(value: Any, context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Flatten common JSON exports while retaining mapping-key context."""
    context = dict(context or {})
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_flatten(item, context))
        return rows
    if not isinstance(value, Mapping):
        return []

    normalized = normalize_record(value)
    has_track = any(key in normalized for key in ("track_id", "label", "translation_m"))
    if has_track:
        row = dict(context)
        row.update(normalized)
        return [row]

    rows = []
    for key, child in value.items():
        child_context = dict(context)
        # JSON cannot preserve tuple keys; accept the common string form.
        if isinstance(key, str) and "|" in key:
            parts = key.split("|", 1)
            child_context.setdefault("log_id", parts[0])
            child_context.setdefault("prompt", parts[1])
        elif isinstance(key, str) and "(" in key and "," in key:
            child_context.setdefault("source_key", key)
        rows.extend(_flatten(child, child_context))
    return rows


def _as_list(value: Any) -> list[Any]:
    """Convert numpy-like arrays and ordinary sequences to a Python list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _flatten_refav_pickle(payload: Any) -> list[dict[str, Any]]:
    """Flatten the official RefAV spatio-temporal pickle format.

    The official format is ``{(log_id, prompt): [frame, ...]}``, where each
    frame stores N-length arrays for track_id, score, label, translation_m,
    size, and yaw.  Keeping this conversion here makes the normalization
    explicit and lets the verifier reject malformed array lengths.
    """
    if not isinstance(payload, Mapping):
        return _flatten(payload)
    rows: list[dict[str, Any]] = []
    for key, frames in payload.items():
        log_id = prompt = None
        if isinstance(key, tuple) and len(key) == 2:
            log_id, prompt = key
        elif isinstance(key, list) and len(key) == 2:
            log_id, prompt = key
        elif isinstance(key, str) and "|" in key:
            log_id, prompt = key.split("|", 1)
        context = {"log_id": log_id, "prompt": prompt}
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError(f"RefAV frame {frame_index} for {key!r} is not a mapping")
            timestamp = frame.get("timestamp_ns", frame.get("timestamp"))
            track_ids = _as_list(frame.get("track_id", []))
            n = len(track_ids)
            vector_fields = ("score", "label", "name", "translation_m", "size", "yaw")
            arrays: dict[str, list[Any]] = {}
            for field in vector_fields:
                if field not in frame:
                    continue
                arrays[field] = _as_list(frame[field])
                if len(arrays[field]) != n:
                    raise ValueError(
                        f"RefAV frame {frame_index} for {key!r}: {field} length "
                        f"{len(arrays[field])} does not match track_id length {n}"
                    )
            for index, track_id in enumerate(track_ids):
                row = dict(context)
                row.update({
                    "timestamp_ns": timestamp,
                    "track_id": track_id,
                    "frame_index": frame_index,
                })
                for field, values in arrays.items():
                    row[field] = values[index]
                # Preserve optional temporal truth without confusing it with
                # the per-object relevance label.
                if "is_positive" in frame:
                    row["is_positive"] = frame["is_positive"]
                if "raw_category" in frame:
                    row["raw_category"] = frame["raw_category"]
                rows.append(normalize_record(row))
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, CSV, or pickle RefAV-like records.

    Pickle is supported only for trusted local files because RefAV examples and
    submissions are often serialized Python objects.  The verifier never
    downloads or executes remote pickle files.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [normalize_record(json.loads(line)) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return _flatten(json.load(handle))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [normalize_record(row) for row in csv.DictReader(handle)]
    if suffix == ".feather":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Feather input requires pandas and pyarrow") from exc
        frame = pd.read_feather(path)
        rows = []
        for row in frame.to_dict(orient="records"):
            # RefAV's official annotation Feather stores position/size as
            # scalar columns and relevance in mining_category. Preserve the
            # original columns while making the audit aliases explicit.
            if "translation_m" not in row and all(k in row for k in ("tx_m", "ty_m", "tz_m")):
                row["translation_m"] = [row["tx_m"], row["ty_m"], row["tz_m"]]
            if "size" not in row and all(k in row for k in ("length_m", "width_m", "height_m")):
                row["size"] = [row["length_m"], row["width_m"], row["height_m"]]
            rows.append(normalize_record(row))
        return rows
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return _flatten_refav_pickle(pickle.load(handle))
    raise ValueError(f"Unsupported record file: {path}")


def template_key(prompt: Any) -> str:
    """Conservative prompt-template key for holdout diagnostics."""
    text = str(prompt or "").lower().strip()
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<id>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _group_key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("log_id"), row.get("prompt"), row.get("timestamp_ns"))


def inspect_records(records: Iterable[Mapping[str, Any]], source_path: str | None = None) -> dict[str, Any]:
    rows = [normalize_record(row) for row in records]
    missing = Counter()
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    logs: set[Any] = set()
    prompts: set[Any] = set()
    templates: set[str] = set()
    labels = Counter()
    camera_rows = 0
    leakage_fields = Counter()
    temporal_presence = Counter()
    match_statuses = Counter()
    projection_statuses = Counter()
    official_filter = Counter()
    duplicate_keys: list[tuple[Any, Any, Any]] = []
    seen_candidate_keys: set[tuple[Any, Any, Any]] = set()
    nonfinite_fields = Counter()
    invalid_labels = Counter()
    timestamp_values: dict[tuple[Any, Any], list[int]] = defaultdict(list)

    for row in rows:
        unknown_candidate = row.get("match_status") in {"UNMATCHED_TRACK", "MATCHED_UNANNOTATED_GT"}
        for field in REQUIRED_FIELDS:
            if field == "label" and unknown_candidate:
                continue
            if field not in row or _is_missing(row[field]):
                missing[field] += 1
        if row.get("log_id") is not None:
            logs.add(row["log_id"])
        if row.get("prompt") is not None:
            prompts.add(row["prompt"])
            templates.add(template_key(row["prompt"]))
        if row.get("label") in LABEL_NAMES:
            labels[LABEL_NAMES[row["label"]]] += 1
        elif not _is_missing(row.get("label")):
            invalid_labels[str(row.get("label"))] += 1
        if any(field in row and not _is_missing(row[field]) for field in CAMERA_FIELDS):
            camera_rows += 1
        if "is_positive" in row:
            value = row["is_positive"]
            temporal_presence["true" if value is True else "false" if value is False else "ambiguous"] += 1
        if row.get("match_status") not in (None, ""):
            match_statuses[str(row["match_status"])] += 1
        if row.get("projection_status") not in (None, ""):
            projection_statuses[str(row["projection_status"])] += 1
        if isinstance(row.get("distance_m"), (int, float)) and isinstance(row.get("is_drivable"), bool):
            if math.isfinite(float(row["distance_m"])):
                official_filter["eligible" if row["distance_m"] <= 50.0 and row["is_drivable"] else "filtered"] += 1
            else:
                official_filter["missing_or_invalid_geometry"] += 1
        else:
            official_filter["missing_or_invalid_geometry"] += 1
        for field in ("label", "name", "score", "timestamp_ns"):
            if field in row:
                leakage_fields[field] += 1
        # A track is legitimately repeated for different prompts in the
        # official annotation table. Duplicates are invalid only within the
        # declared (log, prompt, timestamp) candidate group.
        candidate_key = (row.get("log_id"), row.get("prompt"), row.get("timestamp_ns"), row.get("track_id"))
        if all(value is not None for value in candidate_key):
            if candidate_key in seen_candidate_keys:
                duplicate_keys.append(candidate_key)
            seen_candidate_keys.add(candidate_key)
        for field in ("score", "timestamp_ns", "yaw"):
            value = row.get(field)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                nonfinite_fields[field] += 1
        if row.get("log_id") is not None and row.get("prompt") is not None and isinstance(row.get("timestamp_ns"), int):
            timestamp_values[(row["log_id"], row["prompt"])].append(row["timestamp_ns"])
        groups[_group_key(row)].append(row)

    timestamp_order_violations = []
    for key, values in timestamp_values.items():
        unique_values = list(dict.fromkeys(values))
        if unique_values != sorted(unique_values):
            timestamp_order_violations.append({"log_id": key[0], "prompt": key[1]})

    group_stats = []
    groups_with_positive = 0
    groups_with_negative = 0
    groups_with_both = 0
    for key, group in groups.items():
        positives = sum(row.get("label") == 0 for row in group)
        negatives = sum(row.get("label") in (1, 2) for row in group)
        groups_with_positive += positives > 0
        groups_with_negative += negatives > 0
        groups_with_both += positives > 0 and negatives > 0
        group_stats.append({
            "log_id": key[0],
            "prompt": key[1],
            "timestamp_ns": key[2],
            "candidate_count": len(group),
            "referred_count": positives,
            "negative_count": negatives,
        })

    return {
        "source_path": source_path,
        "record_count": len(rows),
        "group_count": len(groups),
        "unique_log_count": len(logs),
        "unique_prompt_count": len(prompts),
        "unique_template_count": len(templates),
        "missing_required_fields": dict(missing),
        "label_counts": dict(labels),
        "temporal_presence_counts": dict(temporal_presence),
        "match_status_counts": dict(match_statuses),
        "projection_status_counts": dict(projection_statuses),
        "projected_record_count": projection_statuses.get("PROJECTED", 0),
        "projection_fraction": projection_statuses.get("PROJECTED", 0) / len(rows) if rows else 0.0,
        "official_filter_counts": dict(official_filter),
        "official_filter_reproducible": official_filter["missing_or_invalid_geometry"] == 0 and bool(official_filter),
        "camera_associated_record_count": camera_rows,
        "camera_association_fraction": camera_rows / len(rows) if rows else 0.0,
        "candidate_groups_with_referred": groups_with_positive,
        "candidate_groups_with_negative": groups_with_negative,
        "candidate_groups_with_positive_and_negative": groups_with_both,
        "candidate_group_examples": group_stats[:20],
        "potential_input_leakage_fields": dict(leakage_fields),
        "duplicate_candidate_key_count": len(duplicate_keys),
        "duplicate_candidate_key_examples": [list(key) for key in duplicate_keys[:20]],
        "nonfinite_numeric_fields": dict(nonfinite_fields),
        "invalid_label_values": dict(invalid_labels),
        "timestamp_order_violations": timestamp_order_violations,
        "normalized_prompt_holdout_feasible": len(templates) >= 2,
        "template_holdout_requires_external_family_ids": True,
        "log_disjoint_feasible": len(logs) >= 2,
        "fixed_frame_ranking_feasible": bool(groups_with_both),
        "official_label_names": LABEL_NAMES,
        "notes": [
            "RefAV labels are scenario relevance labels, not yielding or action labels.",
            "A single Recall@1 target is not native when a group has multiple referred objects.",
            "Use official HOTA and balanced-accuracy metrics for benchmark comparison where possible.",
            "Exclude label, name, score, and future timestamps from learned candidate features; retain them only for audit and controls.",
        ],
    }


def validate_candidate_feature_schema(fields: Iterable[str]) -> list[str]:
    """Return fields that are forbidden as learned candidate features."""
    forbidden = {
        "label",
        "relevance_label",
        "name",
        "mining_category",
        "is_positive",
        "future_timestamp",
        "timestamp_ns",
        "track_id",
    }
    return sorted(set(fields).intersection(forbidden))


def validate_log_disjoint(split_to_logs: Mapping[str, Iterable[Any]]) -> list[str]:
    """Return overlap errors for an official log-level split manifest."""
    sets = {name: set(values) for name, values in split_to_logs.items()}
    errors = []
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left].intersection(sets[right])
            if overlap:
                errors.append(f"{left}/{right} overlap: {sorted(map(str, overlap))}")
    return errors


def build_manifest(
    *,
    files: Iterable[Path],
    config: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    file_entries = []
    for path in sorted({Path(path) for path in files}):
        if path.is_file():
            file_entries.append({
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return {
        "manifest_version": 1,
        "purpose": "DriveOne RefAV referred-track ranking feasibility gate",
        "config": dict(config),
        "files": file_entries,
        "inspection": dict(inspection),
    }
