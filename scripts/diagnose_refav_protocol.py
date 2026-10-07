#!/usr/bin/env python3
"""Diagnose geometry, visibility, and metadata shortcuts in fixed RefAV groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.eval.refav_diagnosis import diagnose_split, decide_report, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_learned_results(baseline: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "group_count",
        "rankable_group_count",
        "mAP",
        "Recall@1",
        "NLL",
        "Brier",
        "ECE",
        "calibration_labeled_candidate_count",
        "calibration_positive_fraction",
    )
    return {
        model: {
            split: {key: values[key] for key in keys if key in values}
            for split, values in data.get("metrics", {}).items()
        }
        for model, data in baseline.get("models", {}).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, required=True, help="Directory containing refav_{train,validation,test}_subset.jsonl")
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, help="Directory containing controls_{split}.json; defaults to subset root")
    parser.add_argument("--subset-manifest", type=Path, help="Optional fixed-subset manifest")
    parser.add_argument("--split-plan", type=Path, help="Optional repeated-prompt split plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility-gap-threshold", type=float, default=0.10)
    parser.add_argument("--area-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--control-margin", type=float, default=0.03)
    parser.add_argument("--minimum-rankable-groups", type=int, default=50)
    args = parser.parse_args()
    controls_dir = args.controls_dir or args.subset_root
    baseline = read_json(args.baseline_results)

    report: dict[str, Any] = {
        "protocol": {
            "task": "RefAV referred-track ranking",
            "purpose": "CPU-only protocol diagnosis; no model training or candidate-pool changes",
            "fixed_subset": str(args.subset_root.resolve()),
            "baseline_results": str(args.baseline_results.resolve()),
            "baseline_results_sha256": sha256_file(args.baseline_results),
            "thresholds": {
                "visibility_gap": args.visibility_gap_threshold,
                "projected_area_median_ratio": args.area_ratio_threshold,
                "control_mAP_margin": args.control_margin,
                "minimum_rankable_groups": args.minimum_rankable_groups,
            },
        },
        "artifacts": {},
        "learned_results": compact_learned_results(baseline),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows_path = args.subset_root / f"refav_{split}_subset.jsonl"
        controls_path = controls_dir / f"controls_{split}.json"
        if not rows_path.is_file():
            raise SystemExit(f"Missing subset rows: {rows_path}")
        if not controls_path.is_file():
            raise SystemExit(f"Missing controls artifact: {controls_path}")
        rows = read_jsonl(rows_path)
        controls = read_json(controls_path)
        report["artifacts"][split] = {
            "rows": str(rows_path.resolve()),
            "rows_sha256": sha256_file(rows_path),
            "controls": str(controls_path.resolve()),
            "controls_sha256": sha256_file(controls_path),
        }
        report["splits"][split] = diagnose_split(
            rows,
            baseline,
            controls,
            split,
            visibility_gap_threshold=args.visibility_gap_threshold,
            area_ratio_threshold=args.area_ratio_threshold,
            control_margin=args.control_margin,
        )

    for optional_name, optional_path in (("subset_manifest", args.subset_manifest), ("split_plan", args.split_plan)):
        if optional_path is not None:
            if not optional_path.is_file():
                raise SystemExit(f"Missing {optional_name}: {optional_path}")
            report["artifacts"][optional_name] = {
                "path": str(optional_path.resolve()),
                "sha256": sha256_file(optional_path),
            }
            report["protocol"][optional_name] = str(optional_path.resolve())

    report["decision"] = decide_report(report["splits"], minimum_rankable_groups=args.minimum_rankable_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
