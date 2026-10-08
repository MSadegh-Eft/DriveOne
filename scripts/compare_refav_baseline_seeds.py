#!/usr/bin/env python3
"""Compare two frozen RefAV baseline seeds with paired group bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.eval.bootstrap import paired_metric_bootstrap  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(results: dict[str, dict], model: str, baseline: str, split: str, samples: int, seed: int) -> dict:
    left = results["seed0"]["models"][model]["per_group_metrics"][split]
    right = results["seed0"]["models"][baseline]["per_group_metrics"][split]
    seed1_left = results["seed1"]["models"][model]["per_group_metrics"][split]
    seed1_right = results["seed1"]["models"][baseline]["per_group_metrics"][split]
    output = {"model": model, "baseline": baseline, "split": split, "metrics": {}}
    for metric in ("average_precision", "recall_at_1", "labeled_average_precision", "labeled_recall_at_1"):
        output["metrics"][metric] = {
            "seed0": paired_metric_bootstrap(left, right, metric=metric, samples=samples, seed=seed),
            "seed1": paired_metric_bootstrap(seed1_left, seed1_right, metric=metric, samples=samples, seed=seed),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0", type=Path, required=True)
    parser.add_argument("--seed1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["pooled_pe"])
    parser.add_argument("--baselines", nargs="+", default=["task_id", "candidate_only", "metadata_only"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = {"seed0": read(args.seed0), "seed1": read(args.seed1)}
    output = {
        "protocol": "paired per-group bootstrap; operational and labeled-only group metrics",
        "seed0": str(args.seed0.resolve()), "seed1": str(args.seed1.resolve()),
        "comparisons": [],
    }
    for model in args.models:
        for baseline in args.baselines:
            for split in ("validation", "test"):
                output["comparisons"].append(compare(results, model, baseline, split, args.samples, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
