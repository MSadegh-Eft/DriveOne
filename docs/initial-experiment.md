# Initial experiment protocol

The initial experiment evaluates a **custom RefAV referred-track ranking diagnostic**. RefAV is officially a multi-object, multi-timestamp scenario-mining benchmark; it is not a yielding-action or single-winner classification dataset. The pilot therefore must preserve multi-positive referred objects and report the official scenario-mining metrics separately from custom ranking diagnostics.

## Gate order

1. Pin the RefAV repository commit, scenario-mining artifact, Argoverse 2 API/split manifest, tracker-prediction artifact, and SHA-256 hashes.
2. Run `python scripts/verify_refav.py` on a small official export.
3. Confirm candidate coverage, camera/frame association, log-disjoint splits, prompt-family provenance, and the absence of forbidden target fields in model inputs.
4. Run frozen-PE pooled and patch-token smoke tests.
5. Run controls before adding the question-conditioned scorer.

## Candidate contract

- Every prompt timestamp aligned to tracker, pose, and the shared camera frame is retained initially; timestamp selection must not inspect relevance labels.
- Candidate pool: all eligible tracker-produced tracks at that timestamp in the same log.
- Positive label: `REFERRED_OBJECT` (label `0`); multiple positives are allowed.
- Hard negatives: `RELATED_OBJECT` (label `1`) and nearby `OTHER_OBJECT` (label `2`).
- Candidate IDs are randomized before model input.
- Every candidate in a ranking group uses the same `ring_front_center` image and camera timestamp; out-of-view candidates remain explicit records.
- Relevance labels, relevance names, tracker confidence, timestamps, future values, and IDs are excluded from learned candidate features; confidence and metadata may appear only in explicitly named shortcut controls.
- Apply the official 50 m and drivable-area filtering when the required AV2 geometry is available. Otherwise label the result custom and report oracle candidate coverage.

## Required comparisons

1. Random ranking
2. Tracker-score/frequency ranking
3. Candidate-only baseline
4. Metadata-only baseline
5. Learned task-ID baseline
6. Question-conditioned pooled PE baseline
7. Question-conditioned patch-token scorer
8. Oracle ranking upper bound

The direct Qwen baseline is deferred until the frozen student and controls are reproducible. Distillation is deferred until after the non-distilled baseline.

## Metrics

Use official RefAV/Argoverse metrics where the official evaluator can be reproduced:

- HOTA-Temporal
- HOTA-Track
- Timestamp Balanced Accuracy
- Log Balanced Accuracy

Use custom diagnostics for the fixed-frame ranking study:

- per-frame multi-positive average precision;
- per-track average precision;
- Recall@K, with the multi-positive rule declared in advance;
- NLL, Brier score, ECE, and reliability diagrams;
- candidate-count-shift calibration;
- oracle candidate coverage and candidate-count distributions.

Negative prompts and frames with ambiguous `None` targets are not coerced into a winner. Negative/abstention behavior is evaluated separately; ambiguous frames remain in audit statistics and are excluded from the primary ranking metric under a preregistered rule.

## Controls and splits

Required controls include log-disjoint train/validation/test splits, prompt-template provenance, held-out prompt families where externally defined, matched candidate sets, hard negatives, two random seeds after a one-seed smoke test, and end-to-end latency measurement.

A lexical prompt normalization is only a prompt-disjoint diagnostic. It must not be called unseen-template generalization unless template-family IDs or a preregistered clustering method are available.

## Current control-only result

The repaired six-log manifest was used for deterministic controls before any
learned model. On the full candidate pool, tracker-score ranking reached about
0.286 mAP and projected box area about 0.134 mAP, versus about 0.056 for
random ranking. On 2,405 score-matched groups, tracker score fell to about
0.452 while projected box area remained about 0.556. With a fixed 0.20
log-area tolerance in addition to the score match, projected box area remained
about 0.605 mAP versus about 0.530 for random ranking across 1,826 groups.
These are shortcut diagnostics, not DriveOne evidence. The nine-log,
log-disjoint repeated-prompt plan has now been built and audited. The next
step is to run frozen-PE pooled and learned task-ID controls on those exact
splits before adding patch-token fusion.
