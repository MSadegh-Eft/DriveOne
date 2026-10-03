# Current Status and Next Steps

## What has been completed

- The repository was created as an isolated DriveOne project.
- The first commits establish a data contract, verifier, PE smoke-test scaffold, configuration, tests, and documentation.
- Official RefAV metadata and a validation annotation artifact were downloaded locally.
- The official RefAV repository and AV2 API were inspected.
- Official train/validation/test log indices were compared and found disjoint.
- A small camera/calibration/ego-pose sample was downloaded and basic projection was demonstrated.
- The code passes its current nine unit tests and compilation checks.

## What failed, and why that is useful

The strict gate is not passed because the official annotation Feather is not the required tracker candidate artifact. It lacks tracker confidence, camera/image association, and 2D boxes. It also stores quaternion pose rather than the scalar `yaw` field expected by the normalized tracker record.

This is not a model failure. It is an upstream data-contract failure. Training now would risk measuring annotation availability or candidate leakage instead of language-conditioned visual ranking.

## Immediate next dependency

Obtain or generate a pinned tracker-output artifact on the selected RefAV logs. The artifact must contain, at minimum:

- tracker candidate IDs and confidence scores;
- timestamps aligned to the camera stream;
- 3D geometry sufficient to project candidate cuboids;
- a reproducible mapping from each candidate to the camera frame/crop;
- the exact tracker code/checkpoint and hash.

The official RefAV repository mentions tracker outputs and camera-only trackers, but the downloaded validation annotation release does not itself provide the needed candidate file. Do not label ground-truth annotations as tracker candidates without explicitly redesigning the experiment.

## Then run the gate again

1. Add the tracker artifact outside Git.
2. Add its path and hash to `configs/refav_pilot.yaml`.
3. Build the camera association/crop manifest.
4. Run `verify_refav.py --strict` on a tiny log-disjoint sample.
5. Run the official PE environment smoke test with a real image and `--pretrained`.
6. Only after both gates pass, implement random, tracker-score, candidate-only, metadata-only, task-ID, and pooled-PE controls.

## Stop rules

Stop and redesign if the candidate pool cannot be defined independently of the relevance labels, if referred tracks cannot be rendered in a camera frame, if the tracker artifact cannot be reproduced, or if candidate-only metadata explains the result.

## Commit history

- `4976bff` — initial repository layout.
- `a07a511` — RefAV data contract and audit primitives.
- `4c4a249` — RefAV feasibility and PE smoke-test workflow.
- The vault itself should be committed separately after these notes are reviewed.
