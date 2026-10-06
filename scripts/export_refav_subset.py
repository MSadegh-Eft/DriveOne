#!/usr/bin/env python3
"""Export a small, fixed RefAV ranking subset for an early model smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def select_rankable_group_keys(
    frame: pd.DataFrame,
    log_ids: Iterable[str],
    group_count: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    """Select rankable groups with all candidates retained.

    Selection is deterministic after sorting the group keys and applying a
    seeded permutation. Unknown candidates stay in the exported groups.
    """

    if group_count < 1:
        raise ValueError("group_count must be positive")
    selected_logs = {str(log_id) for log_id in log_ids}
    subset = frame[frame["log_id"].astype(str).isin(selected_logs)]
    keys: list[tuple[str, str, int]] = []
    for key, group in subset.groupby(["log_id", "prompt", "timestamp_ns"], sort=True):
        labels = pd.to_numeric(group["label"], errors="coerce")
        if (labels == 0).any() and labels.isin([1, 2]).any():
            keys.append((str(key[0]), str(key[1]), int(key[2])))
    keys.sort()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    return [keys[int(index)] for index in order[: min(group_count, len(keys))]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups-per-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_feather(args.records)
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_records": str(args.records.resolve()),
        "source_sha256": sha256_file(args.records),
        "split_plan": str(args.split_plan.resolve()),
        "groups_per_split_requested": args.groups_per_split,
        "selection_seed": args.seed,
        "selection_rule": "seeded selection from rankable groups; all candidate rows retained, including unknown labels",
        "splits": {},
    }
    for split_name, log_ids in plan["splits"].items():
        keys = select_rankable_group_keys(frame, log_ids, args.groups_per_split, args.seed)
        key_set = set(keys)
        row_keys = pd.MultiIndex.from_frame(
            frame[["log_id", "prompt", "timestamp_ns"]].assign(
                log_id=frame["log_id"].astype(str),
                prompt=frame["prompt"].astype(str),
                timestamp_ns=pd.to_numeric(frame["timestamp_ns"], errors="raise").astype("int64"),
            )
        )
        group_frame = frame[row_keys.isin(key_set)].copy()
        output = args.output_dir / f"refav_{split_name}_subset.feather"
        group_frame.to_feather(output)
        jsonl_output = args.output_dir / f"refav_{split_name}_subset.jsonl"
        with jsonl_output.open("w", encoding="utf-8") as handle:
            for row in group_frame.to_dict("records"):
                handle.write(json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True) + "\n")
        manifest["splits"][split_name] = {
            "log_ids": sorted(str(log_id) for log_id in log_ids),
            "group_count": len(keys),
            "record_count": len(group_frame),
            "rankable_group_count": len(keys),
            "output": str(output.resolve()),
            "output_sha256": sha256_file(output),
            "jsonl_output": str(jsonl_output.resolve()),
            "jsonl_output_sha256": sha256_file(jsonl_output),
            "group_keys": [list(key) for key in keys],
        }
    manifest_path = args.output_dir / "refav_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
