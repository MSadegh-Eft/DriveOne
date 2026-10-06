"""Prepare a small RefAV referred-track ranking manifest from tracker output.

The public Valeo4Cast pickle stores predictions in the city frame, while the
RefAV annotations and camera projection APIs use the egovehicle frame.  This
module performs that coordinate conversion, matches tracker candidates to
ground-truth cuboids using the AV2 2 m centre-distance rule, and joins the
matched object IDs to prompt-specific RefAV relevance labels.

The output is deliberately a *derived diagnostic* rather than an official
RefAV submission.  Unmatched tracker candidates are retained with a null
label and ``UNMATCHED_TRACK`` status; they are never silently converted to
``OTHER_OBJECT``.
"""

from __future__ import annotations

import bisect
import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


LABEL_TO_ID = {"REFERRED_OBJECT": 0, "RELATED_OBJECT": 1, "OTHER_OBJECT": 2}
RING_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)
DEFAULT_CAMERA = "ring_front_center"


class TrackerPreparationError(RuntimeError):
    """Raised when a tracker/sensor/annotation contract cannot be built."""


@dataclass(frozen=True)
class Match:
    candidate_index: int
    gt_track_uuid: str
    distance_m: float


@dataclass(frozen=True)
class CameraProjection:
    camera_name: str | None
    image_path: str | None
    image_timestamp_ns: int | None
    timestamp_delta_ns: int | None
    projected_box: tuple[float, float, float, float] | None
    status: str


def _require_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import pandas as pd
        from av2.geometry.camera.pinhole_camera import Intrinsics, PinholeCamera
        from av2.geometry.geometry import quat_to_mat
        from av2.geometry.se3 import SE3
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrackerPreparationError(
            "Tracker preparation requires pandas, av2, and scipy in the refav environment"
        ) from exc
    return pd, Intrinsics, PinholeCamera, quat_to_mat, SE3, linear_sum_assignment


def load_tracker_pickle(path: Path, log_ids: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load and validate the public ``{log_id: [frame, ...]}`` pickle."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TrackerPreparationError(f"Tracker payload is not a mapping: {path}")
    wanted = set(log_ids) if log_ids is not None else None
    output: dict[str, list[dict[str, Any]]] = {}
    required = ("track_id", "timestamp_ns", "score", "label", "translation_m", "size", "name", "yaw")
    for log_id, frames in payload.items():
        log_id = str(log_id)
        if wanted is not None and log_id not in wanted:
            continue
        if not isinstance(frames, (list, tuple)):
            raise TrackerPreparationError(f"Tracker frames for {log_id} are not a sequence")
        checked: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise TrackerPreparationError(f"Tracker frame {frame_index} for {log_id} is not a mapping")
            missing = [field for field in required if field not in frame]
            if missing:
                raise TrackerPreparationError(f"Tracker frame {frame_index} for {log_id} misses {missing}")
            n = len(np.asarray(frame["track_id"]))
            for field in ("score", "label", "translation_m", "size", "name", "yaw"):
                if len(np.asarray(frame[field])) != n:
                    raise TrackerPreparationError(
                        f"Tracker frame {frame_index} for {log_id}: {field} length does not match track_id"
                    )
            checked.append(dict(frame))
        # The published Valeo4Cast files contain two out-of-order frames in
        # the pilot logs.  Sorting by their explicit timestamps is
        # deterministic and preserves every frame; duplicate timestamps are
        # still rejected because they would make a decision group ambiguous.
        checked.sort(key=lambda frame: int(frame["timestamp_ns"]))
        timestamps = [int(frame["timestamp_ns"]) for frame in checked]
        if len(set(timestamps)) != len(timestamps):
            raise TrackerPreparationError(f"Duplicate tracker timestamps for {log_id}")
        output[log_id] = checked
    return output


def _pose_se3(row: Mapping[str, Any], SE3: Any, quat_to_mat: Any) -> Any:
    quaternion = np.array([[row["qw"], row["qx"], row["qy"], row["qz"]]], dtype=np.float64)
    return SE3(
        rotation=quat_to_mat(quaternion)[0],
        translation=np.array([row["tx_m"], row["ty_m"], row["tz_m"]], dtype=np.float64),
    )


def global_to_ego(points_city: np.ndarray, pose_row: Mapping[str, Any], SE3: Any, quat_to_mat: Any) -> np.ndarray:
    """Transform city-frame points to the egovehicle frame at one timestamp."""
    return _pose_se3(pose_row, SE3, quat_to_mat).inverse().transform_from(np.asarray(points_city, dtype=np.float64))


def _ego_yaw(pose_row: Mapping[str, Any], quat_to_mat: Any) -> float:
    q = np.array([[pose_row["qw"], pose_row["qx"], pose_row["qy"], pose_row["qz"]]], dtype=np.float64)
    rotation = quat_to_mat(q)[0]
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def match_candidates(
    candidate_xy_ego: np.ndarray,
    candidate_names: np.ndarray,
    annotations: Any,
    *,
    distance_threshold_m: float = 2.0,
    linear_sum_assignment: Any | None = None,
) -> dict[int, Match]:
    """Match candidates to same-class annotation cuboids one-to-one."""
    if linear_sum_assignment is None:
        _, _, _, _, _, linear_sum_assignment = _require_runtime_dependencies()
    matches: dict[int, Match] = {}
    for category in np.unique(candidate_names):
        candidate_indices = np.flatnonzero(candidate_names == category)
        gt = annotations[annotations["category"] == category]
        if len(gt) == 0 or len(candidate_indices) == 0:
            continue
        gt_xy = gt[["tx_m", "ty_m"]].to_numpy(dtype=float)
        distances = np.linalg.norm(candidate_xy_ego[candidate_indices, None, :] - gt_xy[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            distance = float(distances[row, column])
            if distance <= distance_threshold_m:
                matches[int(candidate_indices[row])] = Match(
                    candidate_index=int(candidate_indices[row]),
                    gt_track_uuid=str(gt.iloc[column]["track_uuid"]),
                    distance_m=distance,
                )
    return matches


def _box_corners(center_ego: np.ndarray, size_lwh: np.ndarray, yaw_ego: float) -> np.ndarray:
    length, width, height = [float(x) for x in size_lwh]
    local = np.array(
        [[sx * length / 2, sy * width / 2, sz * height / 2]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    rotation = np.array([[math.cos(yaw_ego), -math.sin(yaw_ego), 0.0],
                         [math.sin(yaw_ego), math.cos(yaw_ego), 0.0],
                         [0.0, 0.0, 1.0]])
    return local @ rotation.T + np.asarray(center_ego, dtype=float)


def build_camera_file_index(sensor_root: Path) -> dict[str, tuple[list[int], list[Path]]]:
    """Index camera filenames once; repeated network-directory scans are costly."""
    index: dict[str, tuple[list[int], list[Path]]] = {}
    for camera_name in RING_CAMERAS:
        paths = sorted((sensor_root / "sensors" / "cameras" / camera_name).glob("*.jpg"))
        timestamps = [int(path.stem) for path in paths]
        if paths:
            index[camera_name] = (timestamps, paths)
    return index


def _camera_index(camera_files: Mapping[str, tuple[list[int], list[Path]]], timestamp_ns: int, camera_name: str, tolerance_ns: int) -> tuple[Path, int] | None:
    entry = camera_files.get(camera_name)
    if entry is None:
        return None
    timestamps, paths = entry
    if not paths:
        return None
    index = bisect.bisect_left(timestamps, int(timestamp_ns))
    candidates = []
    if index < len(paths):
        candidates.append((abs(timestamps[index] - timestamp_ns), paths[index], timestamps[index]))
    if index > 0:
        candidates.append((abs(timestamps[index - 1] - timestamp_ns), paths[index - 1], timestamps[index - 1]))
    delta, path, selected_timestamp = min(candidates, key=lambda item: item[0])
    return (path, selected_timestamp) if delta <= tolerance_ns else None


def build_camera_models(sensor_root: Path) -> dict[str, tuple[Any, Any]]:
    """Build AV2 pinhole models for ring cameras present in a log."""
    pd, Intrinsics, PinholeCamera, quat_to_mat, SE3, _ = _require_runtime_dependencies()
    intrinsics_path = sensor_root / "calibration" / "intrinsics.feather"
    extrinsics_path = sensor_root / "calibration" / "egovehicle_SE3_sensor.feather"
    if not intrinsics_path.is_file() or not extrinsics_path.is_file():
        raise TrackerPreparationError(f"Missing camera calibration under {sensor_root}")
    intrinsics = pd.read_feather(intrinsics_path).set_index("sensor_name")
    extrinsics = pd.read_feather(extrinsics_path).set_index("sensor_name")
    models: dict[str, tuple[Any, Any]] = {}
    for camera_name in RING_CAMERAS:
        if camera_name not in intrinsics.index or camera_name not in extrinsics.index:
            continue
        intr = intrinsics.loc[camera_name]
        ext = extrinsics.loc[camera_name]
        camera = PinholeCamera(
            _pose_se3(ext, SE3, quat_to_mat),
            Intrinsics(float(intr.fx_px), float(intr.fy_px), float(intr.cx_px), float(intr.cy_px), int(intr.width_px), int(intr.height_px)),
            camera_name,
        )
        models[camera_name] = (camera, intr)
    if not models:
        raise TrackerPreparationError(f"No ring-camera calibration found under {sensor_root}")
    return models


def build_roi_map(sensor_root: Path) -> Any:
    """Load the official AV2 ROI map used by scenario-mining evaluation."""
    try:
        from av2.map.map_api import ArgoverseStaticMap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrackerPreparationError("ROI filtering requires the av2 package") from exc
    candidates = sorted((sensor_root / "map").glob("log_map_archive_*.json"))
    if len(candidates) != 1:
        raise TrackerPreparationError(f"Expected one AV2 log map archive under {sensor_root / 'map'}, found {candidates}")
    # ``from_json`` loads vector geometry only.  The official ROI evaluator
    # queries a raster ROI layer, so build it from the map directory.
    return ArgoverseStaticMap.from_map_dir(sensor_root / "map", build_raster=True)


def roi_mask_for_frame(frame: Mapping[str, Any], roi_map: Any) -> np.ndarray:
    """Return the official ROI mask for city-frame tracker cuboids."""
    try:
        from av2.map.map_api import RasterLayerType
        from av2.structures.cuboid import Cuboid, CuboidList
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrackerPreparationError("ROI filtering requires the av2 map and cuboid APIs") from exc
    from scipy.spatial.transform import Rotation

    yaw = np.asarray(frame["yaw"], dtype=float)
    quaternions_xyzw = Rotation.from_euler("z", yaw).as_quat()
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
    boxes = np.concatenate(
        [
            np.asarray(frame["translation_m"], dtype=float),
            np.asarray(frame["size"], dtype=float),
            quaternions_wxyz,
        ],
        axis=1,
    )
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    cuboid_list = CuboidList([Cuboid.from_numpy(params) for params in boxes])
    vertices = cuboid_list.vertices_m.reshape(-1, 8, 3)
    points_mask = roi_map.get_raster_layer_points_boolean(vertices.reshape(-1, 3)[:, :2], RasterLayerType.ROI)
    return np.asarray(points_mask.reshape(-1, 8).any(axis=1), dtype=bool)


@lru_cache(maxsize=4096)
def _resolved_image_path(image_path: Path) -> str:
    """Resolve an unchanged image path once rather than once per candidate."""
    return str(image_path.resolve())


def project_candidate(
    center_ego: np.ndarray,
    size_lwh: np.ndarray,
    yaw_ego: float,
    timestamp_ns: int,
    sensor_root: Path,
    camera_models: Mapping[str, tuple[Any, Any]],
    camera_files: Mapping[str, tuple[list[int], list[Path]]],
    *,
    camera_name: str = DEFAULT_CAMERA,
    camera_tolerance_ns: int = 100_000_000,
) -> CameraProjection:
    """Project one candidate into one deterministic, shared camera frame."""
    corners = _box_corners(center_ego, size_lwh, yaw_ego)
    model_entry = camera_models.get(camera_name)
    image_entry = _camera_index(camera_files, timestamp_ns, camera_name, camera_tolerance_ns)
    if model_entry is None or image_entry is None:
        return CameraProjection(None, None, None, None, None, "MISSING_SHARED_CAMERA_FRAME")

    image_path, image_timestamp = image_entry
    camera, intrinsics = model_entry
    uv, points_cam, _ = camera.project_ego_to_img(corners)
    valid_depth = np.isfinite(uv).all(axis=1) & (points_cam[:, 2] > 0.1)
    common = {
        "camera_name": camera_name,
        "image_path": _resolved_image_path(image_path),
        "image_timestamp_ns": int(image_timestamp),
        "timestamp_delta_ns": abs(int(image_timestamp) - int(timestamp_ns)),
    }
    if not valid_depth.any():
        return CameraProjection(**common, projected_box=None, status="OUT_OF_VIEW")

    x0, y0 = np.min(uv[valid_depth], axis=0)
    x1, y1 = np.max(uv[valid_depth], axis=0)
    x0 = max(0.0, min(float(intrinsics.width_px), float(x0)))
    x1 = max(0.0, min(float(intrinsics.width_px), float(x1)))
    y0 = max(0.0, min(float(intrinsics.height_px), float(y0)))
    y1 = max(0.0, min(float(intrinsics.height_px), float(y1)))
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= 1.0:
        return CameraProjection(**common, projected_box=None, status="OUT_OF_VIEW")
    return CameraProjection(
        **common,
        projected_box=(x0, y0, x1, y1),
        status="PROJECTED",
    )


def _load_pose_table(sensor_root: Path) -> Any:
    pd, *_ = _require_runtime_dependencies()
    path = sensor_root / "city_SE3_egovehicle.feather"
    if not path.is_file():
        raise TrackerPreparationError(f"Missing ego pose table: {path}")
    return pd.read_feather(path).set_index("timestamp_ns")


def select_decision_timestamps(
    tracker_frames: list[dict[str, Any]],
    annotations: Any,
    pose_table: Any,
    *,
    camera_files: Mapping[str, tuple[list[int], list[Path]]],
    camera_tolerance_ns: int = 100_000_000,
) -> dict[str, list[int]]:
    """Select all aligned prompt timestamps without inspecting relevance labels.

    Prompt timestamps define the RefAV observation window.  Tracker, pose, and
    shared-camera availability are infrastructure checks only; labels are not
    used to choose or discard timestamps.
    """
    tracker_by_timestamp = {int(frame["timestamp_ns"]): frame for frame in tracker_frames}
    selected: dict[str, list[int]] = {}
    for prompt in annotations["prompt"].drop_duplicates().tolist():
        prompt_annotations = annotations[annotations["prompt"] == prompt]
        timestamps = []
        for timestamp in sorted(set(int(value) for value in prompt_annotations["timestamp_ns"].tolist())):
            if timestamp not in tracker_by_timestamp or timestamp not in pose_table.index:
                continue
            if _camera_index(camera_files, timestamp, DEFAULT_CAMERA, camera_tolerance_ns) is None:
                continue
            timestamps.append(timestamp)
        if timestamps:
            selected[str(prompt)] = timestamps
    return selected


def prepare_records(
    *,
    tracker_path: Path,
    annotation_path: Path,
    sensor_roots: Mapping[str, Path],
    log_ids: Iterable[str],
    distance_threshold_m: float = 2.0,
    camera_tolerance_ns: int = 100_000_000,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create one candidate group for every aligned prompt/timestamp pair."""
    pd, _, _, quat_to_mat, SE3, linear_sum_assignment = _require_runtime_dependencies()
    annotations = pd.read_feather(annotation_path)
    wanted_logs = list(log_ids)
    annotations = annotations[annotations["log_id"].isin(wanted_logs)].copy()
    trackers = load_tracker_pickle(tracker_path, wanted_logs)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "selection_rule": "all annotation prompt timestamps intersected with tracker, pose, and shared-camera timestamps; no relevance-label checks",
        "camera_policy": DEFAULT_CAMERA,
        "logs": {},
        "selected_prompt_count": 0,
        "selected_group_count": 0,
        "unusable_prompts": [],
    }
    for log_id in wanted_logs:
        if progress:
            progress(f"Loading log {log_id}")
        if log_id not in trackers or log_id not in sensor_roots:
            raise TrackerPreparationError(f"Missing tracker or sensor root for {log_id}")
        sensor_root = Path(sensor_roots[log_id])
        pose_table = _load_pose_table(sensor_root)
        camera_models = build_camera_models(sensor_root)
        camera_files = build_camera_file_index(sensor_root)
        roi_map = build_roi_map(sensor_root)
        log_annotations = annotations[annotations["log_id"] == log_id]
        annotation_groups = {
            (str(prompt), int(timestamp)): group
            for (prompt, timestamp), group in log_annotations.groupby(["prompt", "timestamp_ns"], sort=False)
        }
        selected = select_decision_timestamps(
            trackers[log_id],
            log_annotations,
            pose_table,
            camera_files=camera_files,
            camera_tolerance_ns=camera_tolerance_ns,
        )
        summary["logs"][log_id] = {
            "prompt_count": int(log_annotations["prompt"].nunique()),
            "selected_prompts": selected,
        }
        summary["selected_prompt_count"] += len(selected)
        summary["selected_group_count"] += sum(len(timestamps) for timestamps in selected.values())
        summary["unusable_prompts"].extend(
            {"log_id": log_id, "prompt": str(prompt)}
            for prompt in log_annotations["prompt"].drop_duplicates().tolist()
            if prompt not in selected
        )
        frames_by_timestamp = {int(frame["timestamp_ns"]): frame for frame in trackers[log_id]}
        # Geometry and projection depend on the frame, not on the prompt.
        # Cache them within this log without changing row order or labels.
        geometry_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        projection_cache: dict[tuple[int, int], CameraProjection] = {}
        completed_groups = 0
        for prompt, timestamps in selected.items():
          for timestamp in timestamps:
            frame = frames_by_timestamp[timestamp]
            if timestamp not in geometry_cache:
                pose_row = pose_table.loc[timestamp]
                city_positions = np.asarray(frame["translation_m"], dtype=float)
                ego_positions = global_to_ego(city_positions, pose_row, SE3, quat_to_mat)
                roi_mask = roi_mask_for_frame(frame, roi_map)
                geometry_cache[timestamp] = (city_positions, ego_positions, roi_mask, _ego_yaw(pose_row, quat_to_mat))
            city_positions, ego_positions, roi_mask, ego_yaw = geometry_cache[timestamp]
            prompt_annotations = annotation_groups[(prompt, timestamp)]
            matches = match_candidates(
                ego_positions[:, :2],
                np.asarray(frame["name"]),
                prompt_annotations,
                distance_threshold_m=distance_threshold_m,
                linear_sum_assignment=linear_sum_assignment,
            )
            annotations_by_uuid = {str(row.track_uuid): row for row in prompt_annotations.itertuples(index=False)}
            for index in range(len(frame["track_id"])):
                distance_m = float(np.linalg.norm(ego_positions[index, :2]))
                is_in_roi = bool(roi_mask[index])
                if distance_m >= 50.0 or not is_in_roi:
                    continue
                match = matches.get(index)
                annotation = annotations_by_uuid.get(match.gt_track_uuid) if match else None
                label_name = str(annotation.mining_category) if annotation is not None else None
                label = LABEL_TO_ID.get(label_name) if label_name else None
                status = "MATCHED_ANNOTATED" if annotation is not None else "UNMATCHED_TRACK" if match is None else "MATCHED_UNANNOTATED_GT"
                projection_key = (timestamp, index)
                if projection_key not in projection_cache:
                    projection_cache[projection_key] = project_candidate(
                        ego_positions[index],
                        np.asarray(frame["size"])[index],
                        float(frame["yaw"][index]) - ego_yaw,
                        timestamp,
                        sensor_root,
                        camera_models,
                        camera_files,
                        camera_name=DEFAULT_CAMERA,
                        camera_tolerance_ns=camera_tolerance_ns,
                    )
                projection = projection_cache[projection_key]
                records.append({
                    "log_id": log_id,
                    "prompt": prompt,
                    "timestamp_ns": timestamp,
                    "track_id": int(frame["track_id"][index]),
                    "score": float(frame["score"][index]),
                    "label": label,
                    "label_name": label_name,
                    "match_status": status,
                    "matched_track_uuid": match.gt_track_uuid if match else None,
                    "match_distance_m": match.distance_m if match else None,
                    "name": str(frame["name"][index]),
                    "raw_tracker_label": int(frame["label"][index]),
                    "translation_m": ego_positions[index].tolist(),
                    "city_translation_m": city_positions[index].tolist(),
                    "size": np.asarray(frame["size"])[index].tolist(),
                    "yaw": float(frame["yaw"][index]) - ego_yaw,
                    "distance_m": distance_m,
                    # This field follows the existing contract name; it is
                    # the official AV2 scenario-mining ROI mask, not a claim
                    # that the object itself is physically drivable.
                    "is_drivable": is_in_roi,
                    "camera_name": projection.camera_name,
                    "image_path": projection.image_path,
                    "camera_timestamp_ns": projection.image_timestamp_ns,
                    "camera_delta_ns": projection.timestamp_delta_ns,
                    "projected_box": list(projection.projected_box) if projection.projected_box else None,
                    "projection_status": projection.status,
                })
            completed_groups += 1
            if progress and completed_groups % 100 == 0:
                progress(f"{log_id}: {completed_groups} groups completed; {len(records)} candidate rows so far")
        if progress:
            progress(f"Finished {log_id}: {completed_groups} groups")
    return records, summary


def write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write the normalized pilot as a Feather file."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrackerPreparationError("Writing candidate records requires pandas and pyarrow") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_feather(output_path)


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
