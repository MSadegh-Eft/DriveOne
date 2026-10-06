#!/usr/bin/env python3
"""Select a log-disjoint RefAV plan with exact prompt overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from driveone.data.refav_splits import build_prompt_log_sets, make_three_way_plan, select_disjoint_log_triplets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triplets", type=int, default=3)
    parser.add_argument("--min-shared-prompts", type=int, default=2)
    args = parser.parse_args()

    frame = pd.read_feather(args.annotations, columns=["log_id", "prompt"])
    prompt_logs = build_prompt_log_sets(frame.to_dict("records"))
    triplets = select_disjoint_log_triplets(
        prompt_logs,
        triplet_count=args.triplets,
        min_shared_prompts=args.min_shared_prompts,
    )
    plan = make_three_way_plan(triplets)
    plan.update(
        {
            "source_annotation": str(args.annotations.resolve()),
            "source_sha256": sha256_file(args.annotations),
            "selection_rule": "exact prompt string overlap; greedy disjoint triples; sorted logs assigned train/validation/test",
            "template_rule": "exact prompt strings only; no paraphrase or numeric normalization",
            "annotation_log_count": int(frame["log_id"].nunique()),
            "annotation_prompt_count": len(prompt_logs),
            "candidate_triplet_count": len(triplets),
            "min_shared_prompts": args.min_shared_prompts,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
