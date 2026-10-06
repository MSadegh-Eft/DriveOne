#!/usr/bin/env python3
"""Run deterministic RefAV ranking controls on one prepared manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_contract import load_records
from driveone.eval.refav_metrics import run_control_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--hard-negative-delta", type=float, default=0.05)
    parser.add_argument("--size-matched-log-area-delta", type=float, default=0.2)
    parser.add_argument("--split-plan", type=Path, help="Optional JSON plan with train/validation/test log lists")
    parser.add_argument("--splits-only", action="store_true", help="Skip the full-pool result when --split-plan is provided")
    args = parser.parse_args()

    records = list(load_records(args.records))
    result = {}
    if not args.splits_only:
        result = run_control_suite(
            records,
            seeds=args.seeds,
            hard_negative_delta=args.hard_negative_delta,
            size_matched_log_area_delta=args.size_matched_log_area_delta,
        )
    result["source_records"] = str(args.records.resolve())
    if args.split_plan:
        split_plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
        result["split_plan"] = str(args.split_plan.resolve())
        result["split_results"] = {}
        for split_name, log_ids in split_plan["splits"].items():
            allowed = {str(log_id) for log_id in log_ids}
            split_records = [row for row in records if str(row.get("log_id")) in allowed]
            split_result = run_control_suite(
                split_records,
                seeds=args.seeds,
                hard_negative_delta=args.hard_negative_delta,
                size_matched_log_area_delta=args.size_matched_log_area_delta,
            )
            split_result["log_ids"] = sorted(allowed)
            result["split_results"][split_name] = split_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
