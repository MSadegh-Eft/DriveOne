#!/usr/bin/env python3
"""Audit a local RefAV export without downloading large datasets by default."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from driveone.data.refav_contract import (
    build_manifest,
    inspect_records,
    load_records,
    validate_candidate_feature_schema,
    validate_log_disjoint,
)

SUPPORTED = {".json", ".jsonl", ".csv", ".feather", ".pkl", ".pickle"}
DEFAULT_SOURCE = "https://github.com/cainand/refav"


def parse_simple_config(path: Path) -> dict[str, Any]:
    """Parse the small scalar/list subset used by configs/refav_pilot.yaml.

    This avoids making the verifier depend on PyYAML. Users may also pass a
    JSON file, which is valid YAML and is parsed by the same routine.
    """
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    def atom(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
            return raw[1:-1]
        if re.fullmatch(r"[-+]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", raw):
            return float(raw)
        return raw

    result: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.endswith(":") and indent == 0:
            key = stripped[:-1].strip()
            section = {}
            result[key] = section
            continue
        if ":" not in stripped:
            continue
        key, raw_value = (part.strip() for part in stripped.split(":", 1))
        if raw_value in {"", "null", "~"}:
            value = None
        elif raw_value.lower() in {"true", "false"}:
            value = raw_value.lower() == "true"
        elif raw_value.startswith("[") and raw_value.endswith("]"):
            value = [atom(item) for item in raw_value[1:-1].split(",") if item.strip()]
        else:
            value = atom(raw_value)
        target = section if indent else result
        target[key] = value
    return result


def discover_files(data_root: Path, explicit: str | None) -> list[Path]:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            # Accept both repository-relative invocations (the normal CLI
            # form) and paths relative to the configured data root.
            candidates = [ROOT / path, data_root / path, Path.cwd() / path]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        return [path] if path.is_file() else []
    if not data_root.exists():
        return []
    return sorted(path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED)


def run(args: argparse.Namespace) -> int:
    config = parse_simple_config(Path(args.config)) if args.config else {}
    source = config.get("source", {}) if isinstance(config.get("source"), dict) else {}
    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    data_root = Path(args.data_root or data.get("root", "data/refav"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    explicit = args.records or data.get("records_file")
    files = discover_files(data_root, explicit)

    result: dict[str, Any] = {
        "status": "ok" if files else "not_ready",
        "source": source.get("url", DEFAULT_SOURCE),
        "data_root": str(data_root.resolve()),
        "candidate_protocol": config.get("candidate_protocol", {}),
        "split_protocol": config.get("split_protocol", {}),
        "files_discovered": [str(path.resolve()) for path in files],
        "warnings": [],
    }
    provenance = [
        source.get("repository_commit"),
        source.get("dataset_artifact"),
        source.get("argoverse_api_version"),
        source.get("tracker_artifact"),
        source.get("split_manifest_sha256"),
    ]
    result["provenance_placeholders"] = [value for value in provenance if value in {None, "", "PIN_BEFORE_RUN"}]
    result["provenance_unavailable"] = [
        value for value in provenance if isinstance(value, str) and value.startswith("UNAVAILABLE_")
    ]
    if result["provenance_placeholders"]:
        result["warnings"].append("RefAV/Argoverse/tracker provenance is not pinned; do not treat results as reproducible yet.")
    if result["provenance_unavailable"]:
        result["warnings"].append("One or more required artifacts are explicitly unavailable in the downloaded export; the feasibility gate cannot pass.")
    allowed = (config.get("candidate_protocol", {}) or {}).get("candidate_features_allowed", [])
    result["candidate_feature_schema_errors"] = validate_candidate_feature_schema(allowed)
    if result["candidate_feature_schema_errors"]:
        result["warnings"].append("Candidate feature exclusion list contains forbidden fields; verify the model adapter before training.")
    if not files:
        result["warnings"].extend([
            "No local RefAV record file was found; no download was attempted.",
            "Provide --records or place a small official scenario-mining export under the configured data root.",
            "Do not point this verifier at the full sensor download until the schema gate passes.",
        ])
    else:
        all_records = []
        inspections = []
        for path in files:
            try:
                records = load_records(path)
            except Exception as exc:  # report the file-level failure, keep auditing others
                inspections.append({"source_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            all_records.extend(records)
            inspections.append(inspect_records(records, str(path)))
        result["file_inspections"] = inspections
        result["file_errors"] = [item for item in inspections if "error" in item]
        result["inspection"] = inspect_records(all_records, "<all discovered files>")
        result["manifest"] = build_manifest(files=files, config=config, inspection=result["inspection"])
        if result["inspection"]["missing_required_fields"]:
            result["warnings"].append("Required fields are missing from one or more records.")
        if not result["inspection"]["fixed_frame_ranking_feasible"]:
            result["warnings"].append("No candidate group contains both a referred object and a negative.")
        if result["inspection"]["camera_association_fraction"] < 1.0:
            result["warnings"].append("Not every record has explicit camera/frame association; projection must be verified separately.")
        if not result["inspection"]["official_filter_reproducible"]:
            result["warnings"].append("Official 50 m and drivable-area filtering cannot yet be reproduced from the local records.")
        if result["inspection"]["duplicate_candidate_key_count"]:
            result["warnings"].append("Duplicate (log_id, timestamp_ns, track_id) candidates were found.")
        if result["inspection"]["timestamp_order_violations"]:
            result["warnings"].append("Timestamps are not monotonic within at least one log/prompt sequence.")

    split_manifest_file = data.get("split_manifest_file")
    if split_manifest_file:
        split_path = Path(split_manifest_file)
        if not split_path.is_absolute():
            split_path = data_root / split_path
        try:
            split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
            result["split_manifest"] = {
                "path": str(split_path.resolve()),
                "overlap_errors": validate_log_disjoint(split_manifest),
            }
        except Exception as exc:
            result["split_manifest"] = {"path": str(split_path), "error": f"{type(exc).__name__}: {exc}"}
            result["warnings"].append("The configured split manifest could not be read.")
    else:
        result["warnings"].append("No official log split manifest supplied; only a unique-log feasibility check was performed.")

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    if args.strict:
        inspection = result.get("inspection", {})
        if (
            result["status"] != "ok"
            or result["candidate_feature_schema_errors"]
            or inspection.get("missing_required_fields")
            or inspection.get("duplicate_candidate_key_count")
            or inspection.get("timestamp_order_violations")
            or inspection.get("invalid_label_values")
            or inspection.get("nonfinite_numeric_fields")
            or result.get("file_errors")
            or result.get("provenance_placeholders")
            or result.get("provenance_unavailable")
            or (
                (config.get("verification", {}) or {}).get("require_official_split_manifest")
                and not result.get("split_manifest")
            )
            or result.get("split_manifest", {}).get("overlap_errors")
            or result.get("split_manifest", {}).get("error")
            or not inspection.get("fixed_frame_ranking_feasible")
            or (
                (config.get("verification", {}) or {}).get("require_camera_association")
                and inspection.get("camera_association_fraction", 0.0) < 1.0
            )
            or (
                (config.get("candidate_protocol", {}) or {}).get("official_filter")
                and not inspection.get("official_filter_reproducible")
            )
        ):
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/refav_pilot.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--records", help="Explicit JSON, JSONL, CSV, Feather, or trusted local pickle file")
    parser.add_argument("--output", help="Write the JSON audit/manifest to this path")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless the ranking feasibility gate passes")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
