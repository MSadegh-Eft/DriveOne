# Current Status and Next Steps

## What has been completed

- The repository was created as an isolated DriveOne project.
- The data contract, verifier, PE smoke-test scaffold, configuration, tests, and documentation exist.
- Official RefAV metadata and a validation annotation artifact were inspected.
- Official train/validation/test log indices were compared and found disjoint.
- A small camera/calibration/ego-pose sample was downloaded and projection was demonstrated.
- The code passes eleven unit tests and compilation checks.

## Why the first data attempt was insufficient

The annotation-only Feather is not a tracker candidate artifact. It has relevance labels and geometry, but no tracker confidence, camera/image association, or 2-D boxes. It stores quaternion pose rather than the scalar `yaw` field expected by the normalized tracker record. Treating it as a candidate file would make a later ranking result hard to interpret.

## Public tracker pilot

The public Valeo4Cast repository was pinned and its validation tracking artifact was extracted outside Git. The adapter in `src/driveone/data/refav_tracker.py` converts city-frame tracks to the ego frame, performs deterministic same-category 2 m matching, applies the AV2 ROI/range filter, and associates retained candidates with a ring-camera image and projected box.

The repaired two-log pilot contains 446,810 candidates in 1,660 prompt/frame groups. It has 165 groups with both a referred object and a labeled negative. It has 165 referred, 161 related, and 14,344 other prompt-specific labels; 432,140 candidates remain explicitly unmatched. The strict verifier passes its configured structural checks, but the two-log scope and sparse labels are still limitations. This is not a model result and is not enough for the final claim.

## PE result

The smoke test passes through the official CLIP path on host GPU 2. `PE-Core-L14-336` exposes 576 patch tokens of width 1024, pooled image features of width 1024, and text features of width 1024. The official text context is 32 tokens. The standalone tokenizer's default of 77 is not the model context used in this run.

## Immediate next step

Run a small control-only ranking experiment using the repaired manifest, not the raw tracker pickle. The manifest is external and identified by its SHA-256 in `configs/refav_pilot.yaml`.

Before calling the result a benchmark, fix or explicitly label the current target-informed timestamp selection: `select_decision_timestamps` chooses a frame because it contains both a referred and a negative match. That is acceptable for a feasibility check, but it is not yet an unbiased final sampling rule.

## Control order

1. Reproduce the prepared manifest and strict audit.
2. Fix the timestamp selection protocol, or label the run as feasibility-only.
3. Implement random, tracker-score, candidate-only, metadata-only, task-ID, and pooled-PE controls.
4. Report labeled and unmatched candidate strata separately.
5. Only after these controls are reproducible, add the patch-token scorer.

## Stop rules

Stop and redesign if the candidate pool cannot be defined independently of relevance labels, referred tracks cannot be rendered in a camera frame, the tracker artifact cannot be reproduced, or candidate-only metadata explains the result.

## Commit history

- `4976bff` — initial repository layout.
- `a07a511` — RefAV data contract and audit primitives.
- `4c4a249` — RefAV feasibility and PE smoke-test workflow.
- `2531f0d` — public tracker adapter, pilot manifest, and strict audit updates.
- `db7d19b` — PE smoke findings.
- The current vault update should be committed separately.
