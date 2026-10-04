# Current Status and Next Steps

## What has been completed

- The repository was created as an isolated DriveOne project.
- The first commits establish a data contract, verifier, PE smoke-test scaffold, configuration, tests, and documentation.
- Official RefAV metadata and a validation annotation artifact were downloaded locally.
- The official RefAV repository and AV2 API were inspected.
- Official train/validation/test log indices were compared and found disjoint.
- A small camera/calibration/ego-pose sample was downloaded and basic projection was demonstrated.
- The code passes eleven unit tests and compilation checks.

## What failed, and why that is useful

The strict gate is not passed because the official annotation Feather is not the required tracker candidate artifact. It lacks tracker confidence, camera/image association, and 2D boxes. It also stores quaternion pose rather than the scalar `yaw` field expected by the normalized tracker record.

This is not a model failure. It is an upstream data-contract failure. Training now would risk measuring annotation availability or candidate leakage instead of language-conditioned visual ranking.

## Public tracker pilot now completed

The public Valeo4Cast repository was pinned and its validation tracking
artifact was extracted outside Git. The new adapter in
`src/driveone/data/refav_tracker.py` converts city-frame tracks back to the
ego frame, performs deterministic same-category 2 m matching, applies the AV2
ROI/range filter, and associates each retained candidate with a ring-camera
image and projected box.

The two-log pilot contains 2,129 candidates in 10 prompt/frame groups. It has
10 referred, 4 related, and 75 other prompt-specific labels; 2,040 candidates
remain explicitly unmatched. The strict verifier passes, but this sparse
label coverage is a material limitation that must be handled by the controls.

The PE smoke test also passes. `PE-Core-L14-336` exposes 576 patch tokens of
width 1024 and a 1024-dimensional pooled output. It has no native text tower;
the tokenizer is an external 77-token interface, so the question encoder is a
separate DriveOne design choice.

## Immediate next dependency

The data and PE feasibility artifacts now exist. The next dependency is a
small control-only ranking experiment. It must use the generated pilot
manifest, not the raw tracker pickle. The manifest is external and identified
by its SHA-256 in `configs/refav_pilot.yaml`.

The adapter's source artifact contains, at minimum:

- tracker candidate IDs and confidence scores;
- timestamps aligned to the camera stream;
- 3D geometry sufficient to project candidate cuboids;
- a reproducible mapping from each candidate to the camera frame/crop;
- the exact tracker code/checkpoint and hash.

The official RefAV repository mentions tracker outputs and camera-only trackers, but the downloaded validation annotation release does not itself provide the needed candidate file. Do not label ground-truth annotations as tracker candidates without explicitly redesigning the experiment.

## Then run the controls

1. Reproduce the prepared manifest and strict audit.
2. Implement random, tracker-score, candidate-only, metadata-only, task-ID, and pooled-PE controls.
3. Report metrics separately for matched candidates and the explicit unmatched-candidate stratum.
4. Only after those controls are reproducible, add the patch-token scorer.

## Stop rules

Stop and redesign if the candidate pool cannot be defined independently of the relevance labels, if referred tracks cannot be rendered in a camera frame, if the tracker artifact cannot be reproduced, or if candidate-only metadata explains the result.

## Commit history

- `4976bff` — initial repository layout.
- `a07a511` — RefAV data contract and audit primitives.
- `4c4a249` — RefAV feasibility and PE smoke-test workflow.
- Next — public Valeo4Cast adapter, strict pilot manifest, and PE checkpoint findings.
- The vault itself should be committed separately after these notes are reviewed.
