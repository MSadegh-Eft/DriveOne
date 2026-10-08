#!/usr/bin/env python3
"""Create the external report for the corrected frozen pooled-PE gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mAP", "Recall@1", "labeled_only_mAP", "labeled_only_Recall@1",
        "NLL", "Brier", "ECE", "positive_count", "negative_count",
        "unknown_count", "rankable_group_count", "group_count",
    )
    return {key: metrics.get(key) for key in keys}


def stream_split(path: Path) -> dict[str, Any]:
    rows = 0
    logs: set[str] = set()
    groups: set[tuple[str, str, int]] = set()
    pool_keys: set[tuple[str, int, str]] = set()
    repeated_pool_rows = 0
    positives = 0
    projected_positives = 0
    projected_rows = 0
    unknown = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows += 1
        logs.add(str(row["log_id"]))
        group = (str(row["log_id"]), str(row["prompt"]), int(row["timestamp_ns"]))
        groups.add(group)
        pool_key = (str(row["log_id"]), int(row["timestamp_ns"]), str(row["track_id"]))
        if pool_key in pool_keys:
            repeated_pool_rows += 1
        pool_keys.add(pool_key)
        if row.get("label") == 0:
            positives += 1
            projected_positives += row.get("projected_box") is not None
        projected_rows += row.get("projected_box") is not None
        unknown += row.get("label") is None
    return {
        "rows": rows,
        "logs": sorted(logs),
        "prompt_groups": len(groups),
        "unique_pool_keys_after_prompt_expansion": len(pool_keys),
        "prompt_repeat_rows_expected_from_shared_pools": repeated_pool_rows,
        "positive_rows": positives,
        "front_center_projected_positive_rows": projected_positives,
        "front_center_positive_projection_rate": projected_positives / positives if positives else None,
        "all_candidate_rows_with_front_center_box_rate": projected_rows / rows if rows else None,
        "unknown_rows": unknown,
    }


def controls_summary(path: Path, split: str) -> dict[str, Any]:
    data = read_json(path)
    output: dict[str, Any] = {}
    for name, value in sorted(data["results"].items()):
        if not name.endswith("seed_0"):
            continue
        output[name.removesuffix(":seed_0")] = {
            "mAP": value["mean_average_precision"],
            "Recall@1": value["recall_at_1"],
            "labeled_only_mAP": value["labeled_only_mean_average_precision"],
            "labeled_only_Recall@1": value["labeled_only_recall_at_1"],
            "rankable_group_count": value["rankable_group_count"],
            "group_count": value["group_count"],
        }
    return {"split": split, "seed": 0, "methods": output}


def model_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "protocol": data["protocol"],
        "models": {
            model: {
                "parameter_count": value["parameter_count"],
                "train_seconds": value["train_seconds"],
                "score_seconds": value["score_seconds"],
                "metrics": {split: compact_metrics(value["metrics"][split]) for split in ("train", "validation", "test")},
                "stratified_metrics": value["stratified_metrics"],
            }
            for model, value in sorted(data["models"].items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset = args.asset_dir
    audit = read_json(args.audit_report)
    export_manifest = read_json(asset / "refav_baseline_export_manifest.json")
    feature_report = read_json(asset / "pe_core_pooled_features.pt.json")
    seed0_path = asset / "baseline_results_seed0.json"
    seed1_path = asset / "baseline_results_seed1.json"
    comparison_path = asset / "seed_comparison.json"

    split_data = {
        split: stream_split(asset / f"refav_causal_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    split_logs = {split: set(value["logs"]) for split, value in split_data.items()}
    log_disjoint = all(
        split_logs[left].isdisjoint(split_logs[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    seed0 = model_summary(seed0_path)
    seed1 = model_summary(seed1_path)
    controls = {
        split: controls_summary(asset / f"controls_{split}.json", split)
        for split in ("train", "validation", "test")
    }
    test_pooled = seed0["models"]["pooled_pe"]["metrics"]["test"]
    test_task = seed0["models"]["task_id"]["metrics"]["test"]
    test_metadata = seed0["models"]["metadata_only"]["metrics"]["test"]
    test_area = controls["test"]["methods"]["projected_box_area"]
    test_comparison = read_json(comparison_path)

    le3de2e_coverage = audit["candidate_sources"]["le3de2e_causal"]["coverage"]
    groups_path = Path(export_manifest["audit_dir"]) / "groups.jsonl"
    data_checks = {
        "source_pool_hashes_unchanged": export_manifest["candidate_pool"]["candidate_membership_changed"] is False and export_manifest["details"]["source_pool_hashes_unchanged"],
        "source_duplicate_keys_absent": all(not item["duplicate_keys"] for item in audit["candidate_sources"]["le3de2e_causal"]["pool_hashes"]),
        "log_disjoint_splits": log_disjoint,
        "label_independent_candidate_membership": export_manifest["candidate_pool"]["construction"] == "read v6 causal pool before prompt labels",
        "unknown_labels_retained": str(export_manifest["unknown_labels"]).startswith("None;"),
        "all_camera_positive_projection_rate": le3de2e_coverage["positive_projection_rate_downloaded_cameras"],
        "conditional_positive_availability": le3de2e_coverage["positive_availability_given_eligible_gt_positive"],
        "all_timestamp_positive_availability": le3de2e_coverage["positive_availability_all_groups"],
        "fixed_front_center_test_positive_projection_rate": split_data["test"]["front_center_positive_projection_rate"],
    }
    pooled_vs_task_test_seed0 = test_pooled["mAP"] - test_task["mAP"]
    pooled_vs_task_test_seed1 = seed1["models"]["pooled_pe"]["metrics"]["test"]["mAP"] - seed1["models"]["task_id"]["metrics"]["test"]["mAP"]
    gate_checks = {
        "pooled_beats_task_id_mAP_by_0_03": pooled_vs_task_test_seed0 >= 0.03 and pooled_vs_task_test_seed1 >= 0.03,
        "pooled_beats_task_id_recall_by_0_05": (
            test_pooled["Recall@1"] - test_task["Recall@1"] >= 0.05
            and seed1["models"]["pooled_pe"]["metrics"]["test"]["Recall@1"] - seed1["models"]["task_id"]["metrics"]["test"]["Recall@1"] >= 0.05
        ),
        "pooled_beats_metadata_control": test_pooled["mAP"] > test_metadata["mAP"] and test_pooled["Recall@1"] > test_metadata["Recall@1"],
        "pooled_beats_fixed_front_center_area": test_pooled["mAP"] > test_area["mAP"] and test_pooled["Recall@1"] > test_area["Recall@1"],
        "pooled_ece_not_materially_worse_than_task_id": test_pooled["ECE"] <= test_task["ECE"] + 0.03,
        "fixed_front_center_coverage_ge_0_90": split_data["test"]["front_center_positive_projection_rate"] >= 0.90,
        "all_camera_coverage_ge_0_90": le3de2e_coverage["positive_projection_rate_downloaded_cameras"] >= 0.90,
    }
    report: dict[str, Any] = {
        "report_version": "refav-frozen-pooled-baseline-gate-v1",
        "search_date": "2026-10-08",
        "scope": "Corrected Le3DE2E causal candidate pool; frozen PE-Core-L14-336 pooled features; no patch, temporal, Qwen, trajectory, or distillation work.",
        "source_and_artifacts": {
            "audit_report": {"path": str(args.audit_report), "sha256": sha256(args.audit_report)},
            "export_manifest": {"path": str(asset / "refav_baseline_export_manifest.json"), "sha256": sha256(asset / "refav_baseline_export_manifest.json")},
            "tracker": {"path": export_manifest["tracker"], "sha256": export_manifest["tracker_sha256"]},
            "groups": {"path": str(groups_path), "sha256": export_manifest["groups_sha256"]},
            "split_plan": {"path": export_manifest["split_plan"], "sha256": export_manifest["split_plan_sha256"]},
            "feature_cache": feature_report,
            "seed0": {"path": str(seed0_path), "sha256": sha256(seed0_path)},
            "seed1": {"path": str(seed1_path), "sha256": sha256(seed1_path)},
            "seed_comparison": {"path": str(comparison_path), "sha256": sha256(comparison_path)},
        },
        "protocol": {
            "candidate_pool": export_manifest["candidate_pool"],
            "camera_policy": export_manifest["camera_policy"],
            "unknown_label_policy": export_manifest["unknown_labels"],
            "split_counts": export_manifest["outputs"]["splits"],
            "source": export_manifest["source"],
            "adapter_timing_seconds": export_manifest.get("timings_seconds", {}),
        },
        "data_checks": data_checks,
        "split_stream_checks": split_data,
        "controls_seed0": controls,
        "learned_seed0": seed0,
        "learned_seed1": seed1,
        "paired_bootstrap": test_comparison,
        "gate_checks": gate_checks,
        "interpretation": {
            "data_gate": "The source candidate pool remains label-independent and hash-stable; source-level duplicate checks pass. Positive availability is high conditional on eligible ground-truth events, but only 23.1% of all timestamp groups contain an available positive. The fixed front-center view contains a projected positive in 46.0% of test positive rows; seven-camera coverage is 100% in the audit.",
            "model_gate": "The pooled image-plus-question model does not beat task ID, metadata-only, candidate-only, or fixed front-center projected-area controls. Its test ECE is materially worse than task ID. The paired bootstrap intervals do not support the required positive margin.",
            "prompt_holdout": "The main log-disjoint split has exact prompt overlap across logs. joint_holdout_eligible is reported as a secondary subgroup and is small; it must not be described as the primary template-held-out result.",
            "latency": "The reported learned-model timing covers JSONL parsing, image-file validation, cached feature loading, tensor input construction, training, and cached-feature scoring. It excludes PE image encoding for the full set, image decoding during deployment, candidate construction, calibration, and postprocessing. The missing-feature CPU extraction took 157.4 seconds for 86 images and 33 prompts.",
        },
        "decision_basis": {
            "required_margin": "at least +0.05 absolute Recall@1 or +0.03 absolute mAP over the strongest matched baseline, with two-seed support and no material calibration regression",
            "test_pooled_vs_task_id_mAP_seed0": pooled_vs_task_test_seed0,
            "test_pooled_vs_task_id_mAP_seed1": pooled_vs_task_test_seed1,
            "test_pooled_vs_task_id_recall_seed0": test_pooled["Recall@1"] - test_task["Recall@1"],
            "test_pooled_vs_task_id_recall_seed1": seed1["models"]["pooled_pe"]["metrics"]["test"]["Recall@1"] - seed1["models"]["task_id"]["metrics"]["test"]["Recall@1"],
            "test_pooled_mAP": test_pooled["mAP"],
            "test_task_id_mAP": test_task["mAP"],
            "test_metadata_only_mAP": test_metadata["mAP"],
            "test_fixed_front_center_area_mAP": test_area["mAP"],
            "test_pooled_ECE": test_pooled["ECE"],
            "test_task_id_ECE": test_task["ECE"],
        },
        "decision": "POOLED_BASELINE_GATE_FAILED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "test_pooled_mAP": test_pooled["mAP"], "test_fixed_front_center_area_mAP": test_area["mAP"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
