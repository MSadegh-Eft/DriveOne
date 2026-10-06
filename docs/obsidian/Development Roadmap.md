# Development Roadmap

This page answers two questions: **what should I read?** and **why are we doing this step?** The order is deliberate. Each stage can stop the project before the next, more expensive stage begins.

## The research path

| Stage | What we do | Why it matters | Output | Move on when |
| --- | --- | --- | --- | --- |
| 0. Understand the contract | Read the proposal, [[Project Overview]], and [[RefAV Data Contract]] | A wrong task definition makes every later score meaningless | A written definition of one ranking group | We agree that the task is referred-track ranking |
| 1. Prepare data | Run `prepare_refav_tracker.py` on pinned tracker, annotation, pose, calibration, and camera files | The scorer needs a candidate pool that exists before the relevance label is known | A hashed Feather manifest and summary JSON | Candidate construction can be repeated from the same inputs |
| 2. Audit data | Run `verify_refav.py --strict` and inspect the warnings | This catches missing fields, duplicate candidates, split overlap, and obvious leakage | Machine-readable audit | Structural checks pass and limitations are recorded |
| 3. Check PE | Run `smoke_test_pe.py` with the official PE environment | We need to know what tensors the backbone actually returns before writing a fusion model | Shape/configuration report | Patch and pooled paths are both available and measured |
| 4. Run controls | Compare random, tracker-score, candidate geometry, projected area, and category-frequency ranking | These controls tell us whether the candidate pool already contains an easy answer | Control metrics and coverage report | Shortcut controls are recorded and hard negatives are defined |
| 5. Shortcut audit | Add score-matched and size-matched hard negatives on repeated prompt families | The first controls found tracker-confidence and projected-size shortcuts | Shortcut report and fixed hard-negative protocol | No forbidden shortcut explains the planned comparison |
| 6. Minimal DriveOne | Add frozen PE patch tokens, one question encoder, one fusion block, and one scorer | This is the cheapest test of the central mechanism | Two-seed ranking result | Patch model beats pooled PE and task ID on held-out logs/templates by the preregistered margin |
| 7. Stress the result | Add candidate-count shift and log/template holdout | A gain only in the easy pilot is weak evidence | Shift and calibration report | The gain survives the planned shifts |
| 8. Expand carefully | Only then consider more frames, PE-Spatial, other cameras, or other datasets | Each extension changes compute or task semantics and needs its own baseline | Separate evaluation track | The added question is justified by the previous result |
| 9. Distill last | Compare Qwen or another teacher after the non-distilled baseline is fixed | Distillation can hide whether the small model itself works | Teacher comparison and cost report | The teacher target is valid and the baseline is reproducible |

## The file reading order

1. `README.md`: the short command list and repository rules.
2. `configs/refav_pilot.yaml`: the experiment contract written as values.
3. `scripts/prepare_refav_tracker.py`: the command that creates the derived manifest.
4. `src/driveone/data/refav_tracker.py`: the functions used by that command.
5. `scripts/verify_refav.py`: the command that audits the manifest.
6. `src/driveone/data/refav_contract.py`: the reusable loader and checks.
7. `scripts/smoke_test_pe.py`: the PE interface check.
8. `tests/`: small examples that show expected behavior.
9. `docs/initial-experiment.md`: the planned control order.
10. `CONTRIBUTING.md` and `.gitignore`: rules that keep invalid or private artifacts out of the project.

## How the files relate

`configs/refav_pilot.yaml` is the written contract. The two scripts are entry points. The two modules under `src/driveone/data/` contain reusable code called by those scripts. The preparation script writes a manifest; the verifier reads it and writes an audit. Tests check small pieces with synthetic data. The vault explains the decisions and records what the current run actually proved.

## What you should understand before the first model

- A **candidate pool** is the list of objects the scorer is allowed to choose from at one timestamp.
- A **positive** is a candidate marked `REFERRED_OBJECT` by the prompt annotation.
- A **negative** is a candidate marked `RELATED_OBJECT` or `OTHER_OBJECT`.
- An **unmatched** tracker object is still part of the pool, but has no prompt-specific label in the current derived file. It must not silently become a negative.
- A structural pass is not a performance result.

## Current project boundary

Do not implement Qwen, distillation, temporal frames, trajectories, NAVSIM/GTRS, Waymo, PE-Spatial, or deployment optimization before the shortcut controls, score-matched hard negatives, and log-disjoint data pass. Those tasks answer different scientific questions and would make it harder to identify why a result changed.
