# Code Walkthrough: `refav_metrics.py` and `run_refav_controls.py`

## Why these files exist

These files measure simple ranking rules before a learned model is added. They
answer whether the candidate pool already contains an easy answer.

`run_refav_controls.py` is the command-line entry point. It loads one prepared
Feather manifest and writes a JSON result. `refav_metrics.py` groups rows by
`(log_id, prompt, timestamp_ns)`, creates rankings, and calculates metrics.

## Controls

- `random`: a seeded random order;
- `tracker_score`: descending tracker confidence; this is an explicit shortcut
  control because the field is forbidden to the main model;
- `candidate_distance`: nearer objects first;
- `projected_box_area`: larger projected boxes first;
- `category_frequency`: common tracker categories first;
- `oracle`: positives first, used only as an upper bound.

The same candidate rows are used for every control. Ties use a seeded order so
the result can be reproduced.

## Unknown labels

An unmatched tracker candidate has `label=None`. It remains in the candidate
pool and can appear above a labeled positive, but it is not counted as a
positive or negative. The report gives both full-pool metrics and
labeled-only metrics. This separates two questions:

1. Can the method rank the target while unknown candidates are present?
2. Can it rank the target among candidates that have a prompt label?

Primary ranking groups must have at least one referred object and one labeled
negative. Groups without both are reported as coverage information, not
silently discarded from the dataset summary.

## Current result

On the nine-log pilot, tracker-score ranking reaches about 0.281 full-pool mAP
and projected-box area about 0.129 mAP, versus about 0.030 for random ranking.
On 2,029 groups matched on score and `log1p(projected pixel area)` within 0.20,
tracker score is about 0.515, projected-box area about 0.511, and random about
0.495. These are shortcut diagnostics, not DriveOne results. The split-level
values vary widely, so the next experiment must use the fixed train/validation/
test plan and report each split separately.
