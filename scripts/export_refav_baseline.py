#!/usr/bin/env python3
"""Export the v6 causal RefAV pool for the frozen pooled-PE baseline gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_baseline_export import (  # noqa: E402
    BaselineExportError,
    build_baseline_rows,
    load_source_rows,
    load_tracker_index,
    read_groups,
    sha256_file,
    write_baseline_exports,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_split_plan(groups: list[dict], split_plan: dict) -> None:
    owners = {}
    for split, logs in split_plan["splits"].items():
        for log in logs:
            log = str(log)
            if log in owners:
                raise BaselineExportError(f"Log appears in multiple split-plan entries: {log}")
            owners[log] = str(split)
    for group in groups:
        expected = owners.get(str(group["log_id"]))
        if expected is None:
            raise BaselineExportError(f"Group log is absent from split plan: {group['log_id']}")
        if expected != str(group["split"]):
            raise BaselineExportError(
                f"Group split disagrees with split plan for {group['log_id']}: "
                f"{group['split']} != {expected}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True, help="AV2 validation sensor root")
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="le3de2e_causal")
    parser.add_argument("--camera", default="ring_front_center")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    adapter_start = time.perf_counter()
    groups = read_groups(args.audit_dir / "groups.jsonl", args.source)
    if not groups:
        raise SystemExit(f"No groups for source {args.source}")
    split_plan = read_json(args.split_plan)
    validate_split_plan(groups, split_plan)
    logs = sorted({str(group["log_id"]) for group in groups})
    source_rows = load_source_rows(args.audit_dir, logs, args.source)
    tracker_index = load_tracker_index(args.tracker, logs)
    rows_by_split, details = build_baseline_rows(
        groups=groups,
        source_rows=source_rows,
        sensor_root=args.sensor_root,
        tracker_index=tracker_index,
        shuffle_seed=args.shuffle_seed,
        camera_name=args.camera,
    )
    output_manifest = write_baseline_exports(rows_by_split, args.output_dir)
    adapter_elapsed = time.perf_counter() - adapter_start
    manifest = {
        "protocol": "refav-frozen-pooled-baseline-v1",
        "source": args.source,
        "audit_dir": str(args.audit_dir.resolve()),
        "audit_report": str((args.audit_dir / "candidate_source_audit.json").resolve()),
        "audit_report_sha256": sha256_file(args.audit_dir / "candidate_source_audit.json"),
        "groups": str((args.audit_dir / "groups.jsonl").resolve()),
        "groups_sha256": sha256_file(args.audit_dir / "groups.jsonl"),
        "tracker": str(args.tracker.resolve()),
        "tracker_sha256": sha256_file(args.tracker),
        "split_plan": str(args.split_plan.resolve()),
        "split_plan_sha256": sha256_file(args.split_plan),
        "sensor_root": str(args.sensor_root.resolve()),
        "camera_policy": {
            "camera": args.camera,
            "selection": "fixed camera; nearest source image already recorded by v6; no visibility-based camera selection",
            "out_of_view_rows_retained": True,
        },
        "candidate_pool": {
            "construction": "read v6 causal pool before prompt labels",
            "hash_checked_before_label_attachment": True,
            "candidate_membership_changed": False,
            "candidate_ids_shuffled_per_group": True,
        },
        "unknown_labels": "None; retained in evaluation and omitted from BCE loss",
        "timings_seconds": {"adapter_total": adapter_elapsed},
        "geometry": "v6 city-frame translation transformed to ego frame using official AV2 pose",
        "groups": len(groups),
        "logs": logs,
        "details": details,
        "outputs": output_manifest,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "refav_baseline_export_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaselineExportError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
