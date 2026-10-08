#!/usr/bin/env python3
"""CPU-only official replay and candidate-source audit on the pinned pilot.

Large outputs go outside Git. Existing files are read only. Sources share the
exact native Le3DE2E timestamp grid; the old 10Hz nearest-frame expansion is
reported separately and never mixed into the matched source comparison.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_candidate_sources import (  # noqa: E402
    AUDIT_VERSION, THRESHOLDS_M, associate, box_corners, build_pool,
    choose_decision, count_bin, coverage, pool_hash, project_pool, validate_pool, validate_splits,
)
from driveone.data.refav_tracker import (  # noqa: E402
    RING_CAMERAS, build_camera_file_index, build_camera_models, build_roi_map, load_tracker_pickle,
)
from driveone.eval.refav_metrics import run_control_suite  # noqa: E402

OFFICIAL_COMMIT = "5c5be6439ce59b61a31d56431a79a8a04bba33fa"
SOURCE_COLUMNS = ["log_id", "prompt", "timestamp_ns", "track_uuid", "mining_category", "category",
                  "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_selected(path: Path, logs: list[str], columns: list[str]) -> pd.DataFrame:
    # Arrow filters before pandas conversion; do not materialize 17M Python dictionaries.
    table = feather.read_table(path, columns=columns)
    return table.filter(pc.is_in(table["log_id"], value_set=pa.array(logs))).to_pandas()


def official_namespace(repo: Path, sensor_root: Path) -> dict:
    """Execute only the pinned converter functions, without LLM imports/API calls.

    The source bodies are used unchanged, including the whole-log filter and
    the path-dependent Le3DE2 height adjustment. Runtime dataset/output paths
    are injected into their original global namespace; no upstream file edits.
    """
    from av2.utils.io import read_city_SE3_ego, read_feather

    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_COMMIT:
        raise ValueError(f"RefAV source revision mismatch: {actual}")
    source = repo / "refAV/dataset_conversion.py"
    tracked = subprocess.check_output(["git", "-C", str(repo), "show", f"{OFFICIAL_COMMIT}:refAV/dataset_conversion.py"])
    if source.read_bytes() != tracked:
        raise ValueError("Official converter has local edits")
    tree = ast.parse(source.read_text())
    names = {"filter_ids_by_score", "process_sequences", "add_ego_to_annotation", "euler_to_quaternion"}
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    if len(functions.body) != len(names):
        raise ValueError("Official converter function set changed")
    ns = {"np": np, "pd": pd, "feather": feather, "Path": Path, "Rotation": Rotation,
          "read_city_SE3_ego": read_city_SE3_ego, "read_feather": read_feather,
          "get_log_split": lambda _: "val", "AV2_DATA_DIR": sensor_root.parent}
    exec(compile(functions, str(source), "exec"), ns)
    return ns


def roi_mask(rows: list[dict], avm: object) -> np.ndarray:
    from av2.map.map_api import RasterLayerType

    if not rows:
        return np.empty(0, bool)
    corners = box_corners(rows)
    mask = avm.get_raster_layer_points_boolean(corners.reshape(-1, 3)[:, :2], RasterLayerType.ROI)
    return np.asarray(mask).reshape(-1, 8).any(axis=1)


def raw_gt_rows(df: pd.DataFrame, pose: object, log: str, timestamp: int) -> list[dict]:
    from av2.geometry.geometry import quat_to_mat

    if df.empty:
        return []
    centers = pose.transform_from(df[["tx_m", "ty_m", "tz_m"]].to_numpy(float))
    rotations = np.einsum("ij,njk->nik", pose.rotation, quat_to_mat(df[["qw", "qx", "qy", "qz"]].to_numpy(float)))
    return [{"log_id": log, "timestamp_ns": timestamp, "track_id": str(r.track_uuid), "name": str(r.category),
             "translation_m": centers[i].tolist(), "rotation": rotations[i].tolist(),
            "size": [float(r.length_m), float(r.width_m), float(r.height_m)],
            "score": float(getattr(r, "score", 1.0)),
             "synthetic_ego": str(r.category) == "EGO_VEHICLE"}
            for i, r in enumerate(df.itertuples())]


def filter_rows(rows: list[dict], pose: object, avm: object, *, include_ego: bool = False) -> list[dict]:
    if not rows:
        return []
    names = [r["name"] for r in rows]
    # All classes are supported; no query-dependent class pruning.
    return build_pool(log_id=rows[0]["log_id"], timestamp_ns=rows[0]["timestamp_ns"],
                      ids=[r["track_id"] for r in rows], names=names,
                      centers_city=np.asarray([r["translation_m"] for r in rows]), sizes=np.asarray([r["size"] for r in rows]),
                      rotations_city=np.asarray([r["rotation"] for r in rows]), scores=[r["score"] for r in rows],
                      city_se3_ego=pose, roi=roi_mask(rows, avm), include_ego=include_ego)


def tracker_rows(frame: dict, pose: object, log: str, *, height_adjustment: bool) -> list[dict]:
    centers = np.asarray(frame["translation_m"], float).copy()
    if height_adjustment:
        # Exactly the official convention: add height/2 in ego Z, then map back to city.
        ego = pose.inverse().transform_from(centers)
        ego[:, 2] += np.asarray(frame["size"])[:, 2] / 2
        centers = pose.transform_from(ego)
    rotations = Rotation.from_euler("z", np.asarray(frame["yaw"])).as_matrix()
    return [{"log_id": log, "timestamp_ns": int(frame["timestamp_ns"]), "track_id": str(frame["track_id"][i]),
             "name": str(frame["name"][i]), "translation_m": centers[i].tolist(),
             "rotation": rotations[i].tolist(), "size": np.asarray(frame["size"][i], float).tolist(),
             "score": float(frame["score"][i]), "synthetic_ego": False}
            for i in range(len(centers))]


def template_plan(prompts: list[str]) -> dict:
    """Conservative connected components for exact/numeric/lexical duplicates."""
    from difflib import SequenceMatcher

    normalize = lambda p: " ".join(re.sub(r"[^a-z0-9# ]", " ", re.sub(r"\d+(?:\.\d+)?", "#", p.casefold())).split())
    values = {p: normalize(p) for p in sorted(set(prompts))}
    parents = {p: p for p in values}
    def find(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]
            p = parents[p]
        return p
    pairs = []
    keys = list(values)
    for i, p in enumerate(keys):
        for q in keys[i + 1:]:
            a, b = values[p], values[q]
            ta, tb = set(a.split()), set(b.split())
            jaccard = len(ta & tb) / max(1, len(ta | tb))
            if a == b or jaccard >= 0.85 or SequenceMatcher(None, a, b).ratio() >= 0.90:
                left, right = find(p), find(q)
                parents[max(left, right)] = min(left, right)
                pairs.append([p, q])
    result = {}
    for p in keys:
        root = find(p)
        bucket = int(hashlib.sha256(values[root].encode()).hexdigest(), 16) % 10
        result[p] = {"normalized": values[p], "cluster": root, "split": "train" if bucket < 6 else "validation" if bucket < 8 else "test"}
    return {"rule": "exact/numeric plus token Jaccard >=.85 or character similarity >=.90; connected components; SHA256 60/20/20",
            "semantic_paraphrase_holdout_verified": False, "near_duplicate_pairs": pairs, "prompts": result}


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    return {"n": len(array), "quantiles_10_50_90": np.quantile(array, [0.1, 0.5, 0.9]).tolist() if len(array) else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("/ehsan/m.sadegh/driveone_assets/refav")
    parser.add_argument("--asset-root", type=Path, default=base)
    parser.add_argument("--official-repo", type=Path, default=Path("/data/sadegh/tmp/RefAV"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logs", nargs="+", help="Optional smaller smoke subset; never changes the saved split plan")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or ROOT in out.parents and out.relative_to(ROOT).parts[0] not in {"outputs", "runs", "data"}:
        raise SystemExit("Write audit artifacts outside Git or under an ignored directory")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    plan_path = args.asset_root / "refav_repeated_prompt_log_plan.json"
    plan = json.loads(plan_path.read_text())
    validate_splits(plan["splits"])
    logs = sorted(args.logs or plan["all_selected_logs"])
    if not set(logs).issubset(plan["all_selected_logs"]):
        raise ValueError("Logs must belong to the pinned pilot plan")
    log_split = {log: split for split, values in plan["splits"].items() for log in values}
    sensor_root = args.asset_root / "av2_sensor_clean/val"
    annotation_path = args.asset_root / "official/scenario_mining_val_annotations.feather"
    tracker_path = args.asset_root / "official/Le3DE2E_tracking_predictions_val.pkl"
    legacy_path = args.asset_root / "extracted/data/track/for_val_and_test/val_tracking.pkl"
    old_path = args.asset_root / "official/refav_le3de2e_repaired.feather"
    annotations = load_selected(annotation_path, logs, SOURCE_COLUMNS)
    templates = template_plan(annotations.prompt.unique().tolist())
    write_json(out / "prompt_holdout_plan.json", templates)
    tracker = load_tracker_pickle(tracker_path, logs)
    legacy = load_tracker_pickle(legacy_path, logs)
    old_cols = ["log_id", "timestamp_ns", "tracker_timestamp_ns", "track_id", "name", "translation_m", "size", "yaw", "tracker_score"]
    old = load_selected(old_path, logs, old_cols).drop_duplicates(["log_id", "timestamp_ns", "track_id"])
    official = official_namespace(args.official_repo, sensor_root)
    replay_dir = out / "official_replay/Le3DE2E_Tracking/val"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sources = ["custom_historical", "official_replay", "le3de2e_causal", "valeo_historical", "gt_oracle"]
    collected = {s: [] for s in sources}
    pool_digests = {s: [] for s in sources}
    control_records = {s: [] for s in sources}
    differences, integrity, camera_assets, replay_checks = [], [], [], []
    candidate_files = []
    group_file = (out / "groups.jsonl").open("w")
    row_file = (out / "candidate_source_comparison.csv").open("w", newline="")
    row_writer = csv.DictWriter(row_file, fieldnames=["log_id", "timestamp_ns", "track_id", "in_custom", "in_official", "in_causal"])
    row_writer.writeheader()
    from av2.utils.io import read_city_SE3_ego
    from av2.geometry.geometry import quat_to_mat

    for log in logs:
        print(f"Auditing {log}", flush=True)
        sr = sensor_root / log
        poses = read_city_SE3_ego(sr)
        cameras = build_camera_models(sr)
        camera_files = build_camera_file_index(sr)
        avm = build_roi_map(sr)
        la = annotations[annotations.log_id == log]
        anno_times = set(int(t) for t in la.timestamp_ns.unique())
        # Common native timestamp intersection, selected without mining labels.
        frame_by_time = {int(f["timestamp_ns"]): f for f in tracker[log]}
        legacy_by_time = {int(f["timestamp_ns"]): f for f in legacy[log]}
        grid = sorted(set(frame_by_time) & anno_times & set(poses) & set(legacy_by_time))
        if not grid:
            raise ValueError(f"No common exact native timestamps for {log}")
        camera_assets.append({"log_id": log, "split": log_split[log], "camera_files": {c: len(camera_files.get(c, ([], []))[0]) for c in RING_CAMERAS},
                              "all_seven_downloaded": all(c in camera_files for c in RING_CAMERAS),
                              "native_timestamp_count": len(frame_by_time), "common_timestamp_count": len(grid),
                              "annotation_timestamp_count": len(anno_times)})
        official["process_sequences"](log, tracker[log], sensor_root.parent, replay_dir, filter=True)
        replay = pd.read_feather(replay_dir / log / "sm_annotations.feather")
        replay_by_time = {int(t): frame for t, frame in replay.groupby("timestamp_ns")}
        sensor_gt = pd.read_feather(sr / "annotations.feather")
        gt_by_time = {int(t): frame for t, frame in sensor_gt.groupby("timestamp_ns")}
        label_by_group = {(str(prompt), int(t)): frame for (prompt, t), frame in la.groupby(["prompt", "timestamp_ns"])}
        old_log = old[old.log_id == log]
        old_by_time = {int(t): frame for t, frame in old_log.groupby("timestamp_ns")}
        source_rows = {s: [] for s in sources}
        parity_errors = []
        for timestamp in grid:
            pose = poses[timestamp]
            raw_gt = raw_gt_rows(gt_by_time.get(timestamp, sensor_gt.iloc[:0]), pose, log, timestamp)
            # Oracle membership comes from AV2 sensor GT, never the RefAV prompt table.
            gt_eligible = filter_rows(raw_gt, pose, avm)
            # RefAV includes ego targets; keep those explicit in a separate oracle/synthetic diagnostic.
            raw_targets = raw_gt_rows(la[la.timestamp_ns == timestamp].drop_duplicates("track_uuid"), pose, log, timestamp)
            target_eligible = filter_rows(raw_targets, pose, avm)
            eligible_by_id = {r["track_id"]: r for r in target_eligible}
            target_by_id = {r["track_id"]: r for r in raw_targets}
            fr = frame_by_time[timestamp]
            corrected = filter_rows(tracker_rows(fr, pose, log, height_adjustment=True), pose, avm)
            val = filter_rows(tracker_rows(legacy_by_time[timestamp], pose, log, height_adjustment=False), pose, avm)
            official_rows = raw_gt_rows(replay_by_time[timestamp], pose, log, timestamp)
            official_pool = filter_rows(official_rows, pose, avm)
            for r in official_pool:
                r["synthetic_ego"] = r["name"] == "EGO_VEHICLE"
            historic = []
            for r in old_by_time.get(timestamp, old_log.iloc[:0]).to_dict("records"):
                historic.append({"log_id": log, "timestamp_ns": timestamp, "track_id": str(r["track_id"]), "name": r["name"],
                                 "translation_m": np.asarray(r["translation_m"]).tolist(), "size": np.asarray(r["size"]).tolist(),
                                 "rotation": (pose.rotation @ Rotation.from_euler("z", float(r["yaw"])).as_matrix()).tolist(),
                                 "score": float(r["tracker_score"]), "synthetic_ego": False,
                                 "distance_m": float(np.linalg.norm(pose.inverse().transform_from(np.asarray(r["translation_m"])[None])[0, :2]))})
            historic.sort(key=lambda r: r["track_id"])
            pools = dict(zip(sources, [historic, official_pool, corrected, val, gt_eligible]))
            custom_ids, official_ids, causal_ids = [set(r["track_id"] for r in pools[s]) for s in sources[:3]]
            for track_id in sorted(custom_ids | official_ids | causal_ids):
                row_writer.writerow({"log_id": log, "timestamp_ns": timestamp, "track_id": track_id,
                                     "in_custom": track_id in custom_ids, "in_official": track_id in official_ids, "in_causal": track_id in causal_ids})
            differences.append({"log_id": log, "timestamp_ns": timestamp, "custom_count": len(custom_ids), "official_count": len(official_ids),
                                "causal_count": len(causal_ids), "custom_only": len(custom_ids - official_ids), "official_only": len(official_ids - custom_ids)})
            # Verify exact converter translations against the independently implemented city->ego + height convention.
            independent = {r["track_id"]: r for r in tracker_rows(fr, pose, log, height_adjustment=True)}
            for r in official_rows:
                if r["track_id"] in independent:
                    parity_errors.append(float(np.max(np.abs(np.asarray(r["translation_m"]) - independent[r["track_id"]]["translation_m"]))))
            for source, rows in pools.items():
                validation = validate_pool(rows)
                if validation["duplicate_keys"]:
                    raise ValueError(f"Duplicate candidate keys in {source}/{log}/{timestamp}")
                original_hash = validation["sha256"]
                # Explicit permutation test after association: targets cannot change shared pool bytes.
                perturbed = [dict(r, prompt="unrelated shuffled query", label=0, mining_category="REFERRED_OBJECT") for r in reversed(rows)]
                if pool_hash(perturbed) != original_hash:
                    raise AssertionError("Target permutation altered candidate hash")
                projections = project_pool(rows, cameras, camera_files, poses)
                for r, proj in zip(rows, projections):
                    r["camera_projections"] = proj
                    r["any_projected"] = any(p["status"] == "PROJECTED" for p in proj)
                    r["all_seven_assets"] = all(p["status"] != "MISSING_CAMERA_ASSET" for p in proj)
                    areas = [max(0, p["box"][2] - p["box"][0]) * max(0, p["box"][3] - p["box"][1]) if p["box"] else 0 for p in proj]
                    r["projected_area_sum"] = sum(areas)  # never selects a camera for model input
                if pool_hash(rows) != original_hash:
                    raise AssertionError("Projection altered source geometry")
                source_rows[source].extend(rows)
                if source == "gt_oracle":
                    # Direct UUID oracle association, independent candidate membership.
                    targets_index = {r["track_id"]: i for i, r in enumerate(target_eligible)}
                    matches = {i: {"gt_index": targets_index[r["track_id"]], "distance_m": 0.0, "ambiguous": False}
                               for i, r in enumerate(rows) if r["track_id"] in targets_index}
                    sensitivity = {str(th): matches for th in THRESHOLDS_M}
                else:
                    sensitivity = {str(th): associate(rows, target_eligible, threshold_m=th) for th in THRESHOLDS_M}
                    matches = sensitivity["2.0"]
                for prompt in sorted(la.prompt.unique()):
                    annotation = label_by_group.get((prompt, timestamp))
                    if annotation is None:
                        continue
                    label_by_id = dict(zip(annotation.track_uuid.astype(str), annotation.mining_category))
                    all_positive = {k for k, v in label_by_id.items() if v == "REFERRED_OBJECT"}
                    eligible_positive = all_positive & set(eligible_by_id)
                    labels, match_statuses, distances = [], [], []
                    for i, r in enumerate(rows):
                        match = matches.get(i)
                        target_id = target_eligible[match["gt_index"]]["track_id"] if match else None
                        mining = label_by_id.get(target_id)
                        status = "UNMATCHED_TRACK" if not match else "MATCHED_UNANNOTATED_GT" if mining is None else "AMBIGUOUS_MATCH" if match["ambiguous"] else "MATCHED_ANNOTATED"
                        # Ambiguous assignments do not become confident training labels.
                        labels.append({"REFERRED_OBJECT": 0, "RELATED_OBJECT": 1, "OTHER_OBJECT": 2}.get(mining) if status == "MATCHED_ANNOTATED" else None)
                        match_statuses.append(status)
                        distances.append(match["distance_m"] if match else None)
                    matched_target_ids = []
                    for i in range(len(rows)):
                        match = matches.get(i)
                        matched_target_ids.append(
                            target_eligible[match["gt_index"]]["track_id"] if match else None
                        )
                    positives = [i for i, v in enumerate(labels) if v == 0]
                    external_positive_indices = [
                        i for i in positives
                        if target_by_id.get(matched_target_ids[i], {}).get("name") != "EGO_VEHICLE"
                    ]
                    ego_positive_indices = [i for i in positives if i not in external_positive_indices]
                    all_external_positive = {
                        track_id for track_id in all_positive
                        if target_by_id.get(track_id, {}).get("name") != "EGO_VEHICLE"
                    }
                    eligible_external_positive = all_external_positive & set(eligible_by_id)
                    group = {"source": source, "log_id": log, "split": log_split[log], "prompt": prompt, "timestamp_ns": timestamp,
                             "prompt_split": templates["prompts"][prompt]["split"], "joint_holdout_eligible": log_split[log] == templates["prompts"][prompt]["split"],
                             "candidate_count": len(rows), "candidate_count_range": count_bin(len(rows)), "pool_hash": original_hash,
                             "gt_positive_count": len(all_positive), "eligible_positive_count": len(eligible_positive),
                             "gt_external_positive_count": len(all_external_positive),
                             "eligible_external_positive_count": len(eligible_external_positive),
                             "eligible_ego_positive_count": sum(target_by_id.get(i, {}).get("name") == "EGO_VEHICLE" for i in eligible_positive),
                             "matched_positive_count": len(positives), "matched_external_positive_count": len(external_positive_indices),
                             "ego_positive_count": len(ego_positive_indices),
                             "projected_positive_count": sum(rows[i]["any_projected"] for i in positives),
                             "projected_external_positive_count": sum(rows[i]["any_projected"] for i in external_positive_indices),
                             "positive_count_all_seven_assets": sum(rows[i]["all_seven_assets"] for i in positives),
                             "matched_external_positive_count_all_seven_assets": sum(rows[i]["all_seven_assets"] for i in external_positive_indices),
                             "projected_positive_count_all_seven_assets": sum(rows[i]["any_projected"] and rows[i]["all_seven_assets"] for i in positives),
                             "projected_external_positive_count_all_seven_assets": sum(rows[i]["any_projected"] and rows[i]["all_seven_assets"] for i in external_positive_indices),
                             "unknown_count": labels.count(None), "negative_count": sum(v in {1, 2} for v in labels),
                             "statuses": dict(Counter(match_statuses)), "matching_sensitivity": {}, "labels": labels,
                             "match_distances_m": distances, "positive_categories": sorted({target_by_id.get(matched_target_ids[i], {}).get("name") for i in positives}),
                             "track_family": "ego_or_scenario" if not external_positive_indices else "mixed" if ego_positive_indices else "external_object"}
                    for th, mapping in sensitivity.items():
                        confirmed = [i for i, m in mapping.items() if not m["ambiguous"] and target_eligible[m["gt_index"]]["track_id"] in eligible_external_positive]
                        possible = [i for i, m in mapping.items() if target_eligible[m["gt_index"]]["track_id"] in eligible_external_positive]
                        group["matching_sensitivity"][th] = {"confirmed": len(confirmed), "including_ambiguous": len(possible)}
                    group_file.write(json.dumps(group, sort_keys=True, separators=(",", ":")) + "\n")
                    # Group-local rows are shared by all controls; unknowns remain explicit.
                    # Keep the original semantic label and match status. For
                    # ranking only, an unmatched detector/tracker row is an
                    # explicit non-referred false positive, not OTHER_OBJECT.
                    ranking_labels = [1 if label is None else label for label in labels]
                    control_rows = [dict(r, prompt=prompt, label=ranking_label, source_label=label,
                                         projected_box=[0, 0, r["projected_area_sum"], 1], match_distance_m=d,
                                         visibility=float(r["any_projected"])) for r, ranking_label, label, d in zip(rows, ranking_labels, labels, distances)]
                    control_records[source].extend(control_rows)
                    collected[source].append({k: v for k, v in group.items() if k not in {"labels", "match_distances_m"}})
        for source, rows in source_rows.items():
            digest = validate_pool(rows)
            if digest["duplicate_keys"]:
                raise ValueError("Cross-timestamp pool duplicate")
            pool_digests[source].append({"log_id": log, **digest})
            serial = [dict(r, camera_projections=json.dumps(r["camera_projections"], sort_keys=True, separators=(",", ":"))) for r in rows]
            file_path = out / f"{source}_{log}.feather"
            pd.DataFrame(serial).to_feather(file_path)
            candidate_files.append(str(file_path))
        replay_checks.append({"log_id": log, "independent_converter_max_xyz_error_m": max(parity_errors, default=0),
                              "replay_frame_count": len(replay_by_time), "replay_score_filter_uses_future_log": True,
                              "le3de2e_height_adjustment_m": "height/2 in ego Z", "ego_injected": True})
        integrity.append({"log_id": log, "pool_target_permutation_invariance": True, "candidate_uniqueness": True})
        print(f"Finished {log}: {len(grid)} common native timestamps", flush=True)
    group_file.close()
    row_file.close()
    infrastructure_complete = all(r["all_seven_downloaded"] for r in camera_assets)
    comparisons = {}
    for source in sources:
        groups = collected[source]
        full = coverage(groups)
        allseven_pos = sum(g["matched_external_positive_count_all_seven_assets"] for g in groups)
        full["positive_projection_rate_complete_camera_inputs"] = sum(g["projected_external_positive_count_all_seven_assets"] for g in groups) / allseven_pos if allseven_pos else None
        full["complete_camera_matched_positive_count"] = allseven_pos
        stratifications = {}
        for field in ("log_id", "split", "candidate_count_range", "prompt_split"):
            stratifications[field] = {value: coverage([g for g in groups if g[field] == value]) for value in sorted({g[field] for g in groups})}
        sensitivity = {}
        positive_groups = [g for g in groups if g.get("eligible_external_positive_count", g["eligible_positive_count"]) > 0]
        for th in ("1.0", "2.0", "4.0"):
            sensitivity[th] = {"conditional_confirmed_group_recall": sum(g["matching_sensitivity"][th]["confirmed"] > 0 for g in positive_groups) / len(positive_groups) if positive_groups else None,
                               "conditional_group_recall_including_ambiguity": sum(g["matching_sensitivity"][th]["including_ambiguous"] > 0 for g in positive_groups) / len(positive_groups) if positive_groups else None}
        print(f"Controls: {source}, {len(control_records[source])} prompt-expanded rows", flush=True)
        controls = run_control_suite(control_records[source], seeds=(0, 1))
        for part in ("results",):
            controls[part] = {k: v for k, v in controls[part].items()}
        # Recompute controls per log and per log-disjoint split (same rows, no training).
        controls["by_log"] = {}
        controls["by_split"] = {}
        for field, values, output_field in (("log_id", logs, "by_log"), ("split", list(plan["splits"]), "by_split")):
            for value in values:
                subset = [r for r in control_records[source] if (r["log_id"] == value if field == "log_id" else log_split[r["log_id"]] == value)]
                controls[output_field][value] = run_control_suite(subset, seeds=(0,))["results"]
        controls["metrics_caveat"] = "AP/R@1 conditional on observed external positives+negatives; UNMATCHED_TRACK rows retain their status but are explicitly treated as non-referred detector false positives for ranking. This is not OTHER_OBJECT relabeling, not official HOTA, and not comparable to the old label-selected 500 groups."
        controls["learned_controls"] = "candidate-only/metadata-only and PE not rerun: no training authorized for this audit; old scores have different rows"
        write_json(out / f"controls_{source}.json", controls)
        distributions = {}
        for sign, labels in (("positive", {0}), ("negative", {1, 2}), ("unknown", {None})):
            selected = [r for r in control_records[source] if r["label"] in labels]
            distributions[sign] = {field: distribution([r[field] for r in selected if r.get(field) is not None]) for field in ("score", "distance_m", "projected_area_sum", "match_distance_m")}
            distributions[sign]["size_volume_m3"] = distribution([float(np.prod(r["size"])) for r in selected])
            distributions[sign]["category"] = dict(Counter(r["name"] for r in selected))
        near_perfect = any((v[metric] or 0) >= 0.90 for k, v in controls["results"].items() if not k.startswith("oracle")
                           for metric in ("mean_average_precision", "recall_at_1"))
        comparisons[source] = {"coverage": full, "stratifications": stratifications, "matching_sensitivity": sensitivity,
                               "distributions": distributions, "pool_hashes": pool_digests[source], "near_perfect_deterministic_control": near_perfect,
                               "controls_path": str(out / f"controls_{source}.json"), "joint_log_template_holdout": coverage([g for g in groups if g["joint_holdout_eligible"]])}
        del control_records[source]
    # Availability conditional on eligible GT event-positive groups, never all timestamps.
    causal = comparisons["le3de2e_causal"]
    oracle = comparisons["gt_oracle"]
    rates = [v["conditional_confirmed_group_recall"] for v in causal["matching_sensitivity"].values()]
    stability = None if any(r is None for r in rates) else max(rates) - min(rates)
    gate = {"infrastructure_complete": infrastructure_complete, "pool_independent": True,
            "coverage_ge_0_80": (causal["coverage"]["positive_availability_given_eligible_gt_positive"] or 0) >= 0.80,
            "projection_ge_0_90": infrastructure_complete and (causal["coverage"]["positive_projection_rate_downloaded_cameras"] or 0) >= 0.90,
            "matching_stability_le_0_10": stability is not None and stability <= 0.10,
            "no_near_perfect_metadata_control": not causal["near_perfect_deterministic_control"],
            "unknown_target_semantics_resolved": True,
            "second_seed_pooled_model_and_matched_learned_controls": False}
    oracle_pass = None if not infrastructure_complete else (oracle["coverage"]["positive_availability_given_eligible_gt_positive"] or 0) >= .80 and (oracle["coverage"]["positive_projection_rate_downloaded_cameras"] or 0) >= .90
    decision, reason = choose_decision(infrastructure_complete=infrastructure_complete, deployable_pass=all(gate.values()),
                                       oracle_pass=oracle_pass, shortcut_dominated=causal["near_perfect_deterministic_control"])
    source_paths = [annotation_path, tracker_path, legacy_path, old_path, plan_path, args.official_repo / "refAV/dataset_conversion.py"]
    # Hash only small required sensor metadata; images are inventory evidence, not copied or downloaded.
    source_paths += [sensor_root / log / rel for log in logs for rel in ("annotations.feather", "city_SE3_egovehicle.feather", "calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather")]
    source_paths += [p for log in logs for p in sorted((sensor_root / log / "map").glob("*.json"))]
    provenance = [{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in source_paths]
    report = {"audit_version": AUDIT_VERSION, "search_date": "2026-10-08", "decision": decision, "reason": reason,
              "proceed_to_training": False, "gate_checks": gate, "candidate_sources": comparisons, "camera_assets": camera_assets,
              "official_replay_checks": replay_checks, "pool_integrity": integrity, "pool_differences": differences,
              "matching_stability_max_change": stability, "oracle_gate": "unresolved_missing_camera_assets" if oracle_pass is None else oracle_pass,
              "source_revision": OFFICIAL_COMMIT, "provenance": provenance,
              "source_urls": {"official_repo": f"https://github.com/CainanD/RefAV/tree/{OFFICIAL_COMMIT}",
                              "tracker": "https://huggingface.co/datasets/CainanD/AV2_Tracker_Predictions",
                              "annotations": "https://huggingface.co/datasets/CainanD/RefAV", "AV2": "https://argoverse.github.io/user-guide/datasets/sensor.html"},
              "licenses": "RefAV code MIT; AV2 Sensor data CC BY-NC-SA 4.0 terms apply separately; tracker upstream license not independently established",
              "independent_source_inventory": [str(p) for p in sorted(args.asset_root.rglob("*.pkl"))],
              "system": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "device": "cpu", "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "elapsed_seconds": time.monotonic() - start},
              "limitations": ["Only the fixed nine logs; not an exhaustive tracker-source search.", "Camera geometric intersection does not establish actual visibility/occlusion.", "Whole-log official score filtering uses future data and cannot be a causal candidate generator.", "Unknown/ambiguous candidates are not true negatives; ranking numbers are conditional conservative diagnostics.", "Held-out lexical clusters do not establish semantic paraphrase holdout.", "Source comparison uses exact native 2Hz grid; it cannot prove one-frame observability of motion prompts.", "No new model training; learned-control comparison and model gate remain pending."]}
    # Record output hash; reruns can compare pool hashes even though elapsed time changes.
    manifest = {"audit_version": AUDIT_VERSION, "source_revision": OFFICIAL_COMMIT, "provenance": provenance,
                "candidate_pool_hashes": pool_digests, "source_splits": plan["splits"], "selected_logs": logs,
                "output_hashes": [{"path": p, "sha256": sha256(Path(p))} for p in candidate_files],
                "groups_path": str(out / "groups.jsonl"), "groups_sha256": sha256(out / "groups.jsonl"),
                "no_downloads": True, "no_training": True}
    write_json(out / "manifest.json", manifest)
    write_json(out / "candidate_source_audit.json", report)
    with (out / "per_log_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "log_id", *coverage([]).keys()])
        writer.writeheader()
        for source in sources:
            for log, values in comparisons[source]["stratifications"]["log_id"].items():
                writer.writerow({"source": source, "log_id": log, **values})
    print(json.dumps({"decision": decision, "reason": reason, "gate_checks": gate,
                      "report": str(out / "candidate_source_audit.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
