# DriveOne

DriveOne is currently a **feasibility-gate repository**, not a trained driving model. Its immediate purpose is to determine whether a valid, reproducible RefAV referred-track ranking experiment can be constructed before implementing a scorer, temporal model, trajectory model, or distillation pipeline.

## Read in this order

1. [[Project Overview]] — the research question and the deliberately narrow scope.
2. [[Repository Map]] — where each source file fits.
3. [[Architecture and Data Flow]] — how a command moves through configuration, loading, validation, and reporting.
4. [[RefAV Data Contract]] — what a candidate record means and what is forbidden as model input.
5. [[Code Walkthrough - refav_contract]] — the reusable Python library.
6. [[Code Walkthrough - verify_refav]] — the command-line audit program.
7. [[Code Walkthrough - smoke_test_pe]] — the optional Perception Encoder interface check.
8. [[Tests and Validation]] — what is actually tested today.
9. [[Current Status and Next Steps]] — what passed, what failed, and the next dependency.
10. [[Supporting Files]] — the small packaging, policy, documentation, and Git files.

## The one-sentence mental model

The repository takes a local RefAV-like export, normalizes it into explicit candidate records, checks whether the records support a leakage-controlled ranking study, records provenance and hashes, and refuses strict readiness when critical evidence is missing.

## Current decision

The data gate is **not passed**. The official scenario-mining annotation file that was inspected contains ground-truth relevance labels and geometry, but it is not a tracker-produced candidate artifact and has no camera association or tracker confidence. The repository records this result instead of silently treating annotations as model candidates.

## Important links

- [[Current Status and Next Steps]]
- [[Decision Log]]
- [[Glossary]]
- [Repository README](../../README.md)
- [RefAV contract](../../docs/refav_data_contract.md)
