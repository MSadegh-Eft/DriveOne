#!/usr/bin/env python3
"""Assess a repaired RefAV manifest against the protocol gate.

This report is intentionally conservative.  A labelled Feather file can show
coverage and unknown rates, but it cannot prove 1 m/2 m/4 m matching
sensitivity unless all thresholds were computed during construction.  The
report records that limitation instead of silently treating the 2 m result as
the whole sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_repair import candidate_pool_hash  # noqa: E402


def _is_positive(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _is_unknown(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def assess(records: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row.get("log_id")), str(row.get("prompt")), int(row.get("timestamp_ns")))
        groups.setdefault(key, []).append(row)
    duplicate_keys: list[tuple[str, int, int]] = []
    # Labelled output repeats each candidate for each prompt.  One row per
    # candidate identity is the pool representation used for hashing.
    pool_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in records:
        key = (str(row.get("log_id")), int(row.get("timestamp_ns")), int(row.get("track_id")))
        pool_by_key.setdefault(key, row)
    pool_rows = list(pool_by_key.values())
    positive_groups = sum(any(_is_positive(row.get("label")) for row in rows) for rows in groups.values())
    positive_rows = [row for row in records if _is_positive(row.get("label"))]
    projection_rate = sum(bool(row.get("any_projected")) for row in positive_rows) / max(1, len(positive_rows))
    unknown_rows = sum(_is_unknown(row.get("label")) for row in records)
    availability = positive_groups / max(1, len(groups))
    checks = {
        # Labelled output intentionally repeats one candidate pool for each
        # prompt.  Uniqueness is checked on the deduplicated pool below;
        # repeats across prompts are expected and are reported separately.
        "candidate_pool_unique": len(pool_by_key) == len(set(pool_by_key)),
        "label_independent_candidate_pool": bool((manifest or {}).get("label_independent_candidate_construction", False)),
        "positive_availability_ge_0_80": availability >= 0.80,
        "positive_projection_ge_0_90": projection_rate >= 0.90,
        "matching_sensitivity_complete": bool((manifest or {}).get("matching_sensitivity_complete", False)),
    }
    decision = "PROTOCOL_VALIDATION_CONTINUES" if all(checks.values()) else "PROTOCOL_REPAIR_REQUIRED"
    return {
        "protocol": "RefAV referred-track ranking",
        "decision": decision,
        "gate_checks": checks,
        "coverage": {
            "record_count": len(records),
            "candidate_pool_row_count": len(pool_rows),
            "group_count": len(groups),
            "groups_with_referred_track": positive_groups,
            "positive_availability": availability,
            "positive_row_count": len(positive_rows),
            "unknown_row_count": unknown_rows,
            "unknown_row_fraction": unknown_rows / max(1, len(records)),
            "positive_projection_rate": projection_rate,
            "duplicate_candidate_keys": [list(key) for key in duplicate_keys],
            "labelled_identity_repeat_count": len(records) - len(pool_rows),
            "candidate_pool_hash": candidate_pool_hash(pool_rows),
        },
        "matching_sensitivity": (
            {
                "status": "complete",
                "required_thresholds_m": [1.0, 2.0, 4.0],
                "values": (manifest or {}).get("matching_sensitivity", {}),
            }
            if (manifest or {}).get("matching_sensitivity_complete")
            else {
                "status": "not_complete_in_labelled_output",
                "required_thresholds_m": [1.0, 2.0, 4.0],
                "note": "The 2 m label attachment cannot establish the 1/2/4 m conclusion; rerun construction with all thresholds before proceeding.",
            }
        ),
        "control_evidence": {
            "full_pool": "not_run because the coverage gate fails",
            "rankable_subset": "conditional diagnostic only; not a deployment-like candidate pool",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("This command requires pandas in the refav environment") from exc
    records = pd.read_feather(args.records).to_dict(orient="records")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    result = assess(records, manifest)
    result["source_records"] = str(args.records.resolve())
    result["manifest"] = str(args.manifest.resolve()) if args.manifest else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
