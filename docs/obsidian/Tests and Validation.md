# Tests and Validation

## Automated tests

The repository currently uses Python’s built-in `unittest` runner. Run:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/verify_refav.py scripts/smoke_test_pe.py src/driveone/data/refav_contract.py
git diff --check
```

## `test_refav_contract.py`

This file tests the reusable contract logic with synthetic records.

### Official pickle flattening

`make_payload` creates the official-style tuple-keyed, multi-frame structure. The test checks that it becomes six flat rows, preserves both logs, and maps labels correctly.

### Malformed arrays

The test shortens one `yaw` array and expects `ValueError`. This verifies that the loader rejects shape mismatches rather than silently truncating.

### Ranking feasibility and camera association

Synthetic records contain one referred and one negative candidate per group plus camera paths. The test confirms positive/negative groups, log disjoint feasibility, prompt-template diversity, and 100% camera association.

### Duplicate and invalid-label detection

Two identical rows must produce one duplicate candidate-key report and two invalid-label counts for label `9`.

### Feature redaction

The test confirms that `label` and `track_id` are rejected from the declared learned feature allow-list while `projected_box` and `raw_category` pass the declaration check.

### Split overlap

The test creates a train/validation overlap and confirms that the validator reports it.

### Prompt normalization and JSON

The final tests check conservative replacement of numbers and UUID-like values and string-to-integer label conversion in JSON.

## `test_verify_script.py`

Runs the CLI against an empty temporary directory. It verifies two properties: the command has no network side effect, and it emits a machine-readable `not_ready` status with an actionable warning.

## What is not tested yet

There are currently no tests for:

- AV2 cuboid projection and camera timestamp matching;
- official tracker-output conversion;
- AP or calibration calculations;
- PE tensor shapes;
- GPU latency;
- training or model quality;
- end-to-end candidate crop construction.

Those tests should be added only after the missing tracker/camera candidate artifact exists. Tests that merely mirror a future implementation would not resolve the current feasibility blocker.
