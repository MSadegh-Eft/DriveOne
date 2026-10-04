#!/usr/bin/env python3
"""Build the small RefAV referred-track ranking pilot from public tracks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_tracker import prepare_records, write_records, write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True, help="AV2 split directory containing log directories")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    parser.add_argument("--camera-tolerance-ms", type=float, default=100.0)
    args = parser.parse_args()

    roots = {log_id: args.sensor_root / log_id for log_id in args.logs}
    records, summary = prepare_records(
        tracker_path=args.tracker,
        annotation_path=args.annotations,
        sensor_roots=roots,
        log_ids=args.logs,
        distance_threshold_m=args.distance_threshold_m,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000),
    )
    summary = dict(summary)
    summary.update({
        "tracker_path": str(args.tracker.resolve()),
        "annotation_path": str(args.annotations.resolve()),
        "sensor_root": str(args.sensor_root.resolve()),
        "record_count": len(records),
        "camera_associated_count": sum(bool(row.get("image_path")) for row in records),
        "projected_count": sum(row.get("projection_status") == "PROJECTED" for row in records),
        "matched_label_count": sum(row.get("label") is not None for row in records),
        "unmatched_track_count": sum(row.get("match_status") == "UNMATCHED_TRACK" for row in records),
    })
    write_records(records, args.output)
    write_summary(summary, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
