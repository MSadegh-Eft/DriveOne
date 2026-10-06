# DriveOne

This repository is the **small feasibility study** for DriveOne. It is not a driving model yet. We first ask a narrower question:

> Can a model use a RefAV question and an image to rank the referred object above the other tracked objects in the same scene?

This first question is useful because it can disprove the proposed candidate interface before we spend time on temporal models, trajectories, Qwen, or distillation.

## Start here

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

## Current state in plain language

The public tracker and camera assets were used to build a six-log manifest. It
contains 2,143,270 candidates in 9,840 groups, with 1,816,120 unmatched rows.
The strict structural audit passes, but deterministic controls still show a
projected-size shortcut. This is **not evidence for a model result**. The next
step is a repeated-prompt, log-disjoint control split before model training.

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
