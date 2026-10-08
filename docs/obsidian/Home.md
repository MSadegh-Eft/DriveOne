# DriveOne

This repository is the **small feasibility study** for DriveOne. It is not a driving model yet. We first ask a narrower question:

> Can a model use a RefAV question and an image to rank the referred object above the other tracked objects in the same scene?

This first question is useful because it can disprove the proposed candidate interface before we spend time on temporal models, trajectories, Qwen, or distillation.

## Start here

If you prefer one document instead of a note-by-note vault, start with the
[standalone project guide](../DriveOne_Project_Guide.md). It explains the whole
project in order and uses simple English. The generated PDF is kept locally at
`reports/DriveOne_Project_Guide.pdf` and is intentionally ignored by Git.

Read [[Development Roadmap]] first. It gives both reading order and experiment order. Then read:

1. [[Project Overview]] — what we are trying to learn and what we are deliberately not claiming.
2. [[Repository Map]] — one-line purpose of every important file.
3. [[Architecture and Data Flow]] — how the files connect when a command runs.
4. [[RefAV Data Contract]] — the exact meaning of one candidate row and the leakage rules.
5. [[Code Walkthrough - refav_tracker]] — how public tracker output becomes the pilot manifest.
6. [[Code Walkthrough - refav_contract]] — how files are loaded, normalized, checked, and hashed.
7. [[Code Walkthrough - verify_refav]] — the command that turns those checks into a pass/fail audit.
8. [[Code Walkthrough - smoke_test_pe]] — what the PE interface check really measures.
9. [[Tests and Validation]] — what is tested and what is still untested.
10. [[Current Status and Next Steps]] — the current evidence and the next experiment.
11. [[Code Walkthrough - refav_splits]] — how repeated-prompt log splits are selected.
12. [[Code Walkthrough - frozen baselines]] — how the first pooled-PE and shortcut controls run.
13. [Protocol-diagnosis record](../refav_protocol_diagnosis.md) — historical reason the first pilot was paused.
14. [Corrected candidate-source decision](../refav_candidate_pool_decision.md) — why the frozen pooled baseline is now authorized.

## Current state in plain language

The public tracker and camera assets were used to build a nine-log,
log-disjoint repeated-prompt manifest. The frozen-PE baseline was tested on
200 and then 500 groups per split; those results remain historical because the
evaluator was later corrected. The corrected CPU-only audit preserves unknown
labels, separates global assignment ties from local nearby matches, excludes
synthetic ego rows from external controls, and reports
`POOLED_BASELINE_GATE_FAILED`. The corrected frozen pooled-PE baseline has now
run with two seeds. Pooled PE did not beat task ID, metadata-only, or the
fixed projected-area control, so patch-token work is still deferred. Read the
[pooled baseline gate](../refav_pooled_baseline_gate.md) for the numbers and
the next decision.

The PE smoke test found 576 visual patch tokens of width 1024, a
1024-dimensional pooled image output, and a 1024-dimensional text output for
`PE-Core-L14-336`. The official CLIP text context is 32 tokens. This is an
interface check, not evidence that the features improve ranking.

## One-sentence mental model

Raw tracker and camera files go through the preparation adapter, become a fixed candidate manifest, pass through the contract checker, and then feed simple ranking controls. Only if those controls are trustworthy do we add the small DriveOne scorer.

## Important links

- [[Current Status and Next Steps]]
- [[Decision Log]]
- [[Glossary]]
- [Repository README](../../README.md)
- [RefAV contract](../../docs/refav_data_contract.md)
