# Current Status and Next Steps

## Latest gate: 2026-10-08

The source-level candidate protocol passed its data checks. We then ran the
corrected frozen pooled-PE baseline on the official Le3DE2E causal pool with
seeds 0 and 1. This model gate failed:

- pooled PE test mAP: 0.0668 (seed 0), 0.0569 (seed 1);
- task ID test mAP: 0.0787, 0.0718;
- metadata-only test mAP: 0.0830, 0.0810;
- fixed front-center projected-area control: 0.1964 mAP;
- pooled PE seed-0 ECE: 0.1583 versus 0.0468 for task ID.

The source pool is still label-independent and hash-stable. Unknown candidates
remain in the ranking pool. All-camera positive projection is 100% in the
audit, but the fixed front-center image contains a projected positive in only
46.0% of test positive rows. The full explanation is in
`docs/refav_pooled_baseline_gate.md`; the machine-readable report is kept
outside Git with the data.

**Current decision: `POOLED_BASELINE_GATE_FAILED`.** Do not add patch tokens,
Qwen, temporal frames, trajectories, or distillation. First choose between a
fair fixed multi-camera input redesign and stopping the RefAV branch.

## What has been completed

- The repository was created as an isolated DriveOne project.
- The data contract, verifier, tracker adapter, PE smoke test, controls, tests, and documentation exist.
- Official RefAV metadata, annotations, tracker output, and AV2 sensor assets were inspected.
- The code passes 35 tests in the `refav` environment.
- `PE-Core-L14-336` was tested through the official CLIP image and text path on host GPU 2.

## Repaired candidate protocol

The adapter now keeps every prompt timestamp that is aligned with tracker,
pose, and the shared camera. It does not inspect relevance labels when choosing
timestamps. Every candidate in a group uses the same `ring_front_center` image.
Candidates outside that view remain in the pool with `OUT_OF_VIEW` status.

The six-log manifest contains 2,143,270 candidates in 9,840 groups. It has
2,502 groups with both a referred object and a labeled negative. It has 15,804
referred, 11,040 related, and 300,306 other prompt-specific labels; 1,816,120
candidates remain unmatched. The strict audit passes. The manifest has six
logs, but only seven prompts repeat across more than one selected log, so it is
still a control pilot rather than a strong language-generalization benchmark.

## PE result

The smoke test passes through the official CLIP path on host GPU 2.
`PE-Core-L14-336` exposes 576 patch tokens of width 1024, pooled image
features of width 1024, and text features of width 1024. The official text
context is 32 tokens. The image input was synthetic, so this proves the
interface and records timing; it does not prove visual quality.

## Control result and current stop

The six-log deterministic controls show a strong shortcut:

- tracker-score ranking: 0.286 full-pool mAP and 0.309 labeled-only mAP;
- projected-box area: 0.134 full-pool mAP and 0.377 labeled-only mAP;
- nearest-candidate distance: 0.071 full-pool mAP;
- random ranking: 0.056 full-pool mAP;
- oracle ranking: 1.0 mAP.

On 2,405 score-matched hard-negative groups, tracker score falls to 0.452
mAP, random ranking is 0.467, nearest distance is 0.484, and projected-box
area is 0.556. On 1,826 groups matched on score and log-area within 0.20,
tracker score is 0.548, random ranking is 0.530, and projected-box area is
0.605. The tracker-score advantage is reduced by score matching, but projected
size remains a shortcut even after the fixed size match.

This is not DriveOne evidence. Tracker confidence is excluded from learned
features, and the current result says that candidate availability and visual
size are correlated with the labels. Stop before training a question-conditioned
PE scorer. The next task is to select more logs containing repeated prompt
families and rerun this fixed control protocol. Do not add a learned PE scorer
until the geometry shortcut is either explained by the candidate construction
or removed by a defensible matched-negative design.

The six-log manifest hash is
`865779aa04c48b8a0a889e13598354d6eb183d4c759915e00120fce6f7f8dd45`.
The control result hash is
`bae30b14125ffa814624dd4a47c4b5981ed957a7d347aa2752aa75ba3a80cebf`.

A deterministic repeated-prompt plan now exists outside Git. It selects nine
additional logs as three disjoint train/validation/test triplets. Each triplet
shares two exact prompt strings across all three logs. The plan uses exact
prompt text only; it does not claim paraphrase or template-family equivalence.
Its SHA-256 is
`cba73f6f34a5373bf3a5f69765b78f5eeebf48ad2ebdab38032a204cd7c60f85`.

## Repeated-prompt nine-log result

The nine-log manifest passed the strict audit. It contains 3,338,670 candidates
in 14,120 groups, with 3,242 rankable groups. Every row has a shared camera
image; 856,830 rows have a projected box and 2,987,540 are unmatched tracker
rows. The manifest SHA-256 is
`af5142e17c955333ea4429e55eee23269c4d14e843bed29e57b2f714830f0dbb` and the
audit SHA-256 is
`f8011df7017b75372c2b0cbf08a9763f7e7f46c68dedbe63a70cfcde19aad851`.

On the full nine-log pool, random, tracker-score, and projected-area controls
reach 0.030, 0.281, and 0.129 mAP. On 2,029 groups matched on tracker score
and log-area, they reach 0.495, 0.515, and 0.511 mAP. The score-and-size
matched control result SHA-256 is
`3d865f67f1b9ecb56c35376f3e975f39ce371dfabc762ce99cd6d8eb518aff5c`.

The split controls are not uniform: size-matched tracker/area mAP is
0.335/0.376 on train, 0.795/0.642 on validation, and 0.548/0.607 on test.
These per-split numbers must remain visible; pooling them would hide a major
log-dependent shortcut. The split-control SHA-256 is
`780816682c52483ce8f262d7e3bab088c133619b36b2ae15ada6ff5263c86844`.

## Next task

The corrected frozen-PE baseline smoke test is complete. It used 200 rankable
groups per split from the nine-log plan and one random seed. Pooled PE scored
0.0319 test mAP; task ID with the same pooled image scored 0.0581. On the exact
same groups, deterministic tracker-score and projected-box-area ranking scored
0.1674 and 0.1678 mAP. Pooled PE also had test ECE 0.1322, compared with
0.0559 for metadata-only.

This is an early stop signal. It is not a final rejection because the run is
small and uses one seed. It does mean that adding patch tokens now would hide
an unresolved baseline problem. Read [[Code Walkthrough - frozen baselines]]
and `docs/refav_baseline_smoke.md` before running more model code.

The corrected 500-group-per-split replication is now complete. Pooled PE scored
0.0476 test mAP and task ID with the same pooled image scored 0.0576;
tracker-score and projected-box-area controls scored 0.1813 and 0.1729 on the
same candidates. This confirms the early stop at a larger subset. Do not add
patch tokens or a second seed yet.

The corrected result hashes are `7bd361d9988cc1fd3db90299d4e5c6b346174265bfcce6f4b25e161ac272f164`
(200 groups per split) and
`3d9f83776f3ed7713b00a474a940e149319c1b0086addf13bf770c659e3605b8`
(500 groups per split). The earlier v1 hashes remain listed in the config only
to show which artifacts were superseded.

## Protocol-diagnosis result

The CPU-only diagnosis of the 500-group artifacts is complete. The full report
is in `docs/refav_protocol_diagnosis.md`; the detailed JSON is outside Git at
`/ehsan/m.sadegh/driveone_assets/refav/refav_protocol_diagnosis_500.json`.

The selected groups have 100% oracle candidate coverage because the exporter
deliberately selected rankable groups. This does not describe the full
manifest. Unknown rows make up 84.4% of train and about 89.8% of validation
and test. Referred-object visibility is 38.6% versus 16.4% for labeled
negatives in validation, and 46.9% versus 21.0% in test. The direction changes
in train, which shows that the shared camera signal is unstable across logs.

Non-oracle deterministic controls remain much stronger than pooled PE on the
held-out splits, including the score-and-size-matched diagnostic subset. The
diagnosis decision is **`PROTOCOL_REPAIR_REQUIRED`**.

The next action is to design one label-independent, deterministic, hashable
candidate-pool repair and rerun the controls on that exact pool. Do not add
patch tokens, Qwen, temporal frames, trajectories, or distillation yet.

Qwen, temporal input, PE-Spatial, trajectories, distillation, and deployment
optimization remain deferred.

## Stop rules

Stop and redesign if candidate construction cannot be independent of relevance
labels, target coverage is poor, the shared camera removes most targets, or a
candidate-only shortcut remains competitive after matched hard negatives.

## Official Le3DE2E repair result (2026-10-07)

We rebuilt the pilot from the official RefAV scenario annotations and the
official Le3DE2E tracker.  The nine-log output is outside Git at
`/ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired.feather`.

The candidate pool itself is reproducible and label-independent.  However, only
2,935 of 14,110 prompt/timestamp groups (20.8%) contain a matched referred
track.  Only 33.2% of matched referred rows project into at least one camera,
and 90.5% of rows remain unknown after matching.  These numbers fail the
80% positive-availability and 90% camera-coverage gates.

The official repair assessment is
`PROTOCOL_REPAIR_REQUIRED`.  A 500-rankable-group control run exists only as a
conditional diagnostic; it selected groups using transferred labels and is not
a deployment-like result.  The next decision is to repair the tracker/target
interface or stop the RefAV branch.  Do not begin patch-token training, Qwen,
temporal frames, trajectories, or distillation.

## Commit history

- `4976bff` — initial repository layout.
- `a07a511` — RefAV data contract and audit primitives.
- `4c4a249` — RefAV feasibility and PE smoke-test workflow.
- `2531f0d` — public tracker adapter, pilot manifest, and strict audit updates.
- `db7d19b` — PE smoke findings.
- `7930e77` — simplified Obsidian roadmap and corrected PE documentation.
- `cfb2d1e` — label-independent timestamps and shared-camera protocol.
- `a75b1b8` — leakage-aware deterministic control suite.
- `19c8859` — frozen-PE subset export, feature extraction, and baseline controls.
- `3bd966b` — baseline helper tests for fixed feature shape and unknown-row ranking.

## Candidate-source audit result (historical v3)

The official Le3DE2E conversion was replayed on the nine native timestamp
grids. The city-to-ego conversion matched exactly. We downloaded only the
nearest frames for all seven cameras at those timestamps, so camera coverage
is now a real measurement rather than a missing-file artifact.

For external-object prompts, official Le3DE2E has 68.6% conservative
referred-track availability at 2 m. The result changes from 71.7% at 1 m to
59.2% at 4 m. The ground-truth oracle reaches 100% availability and 100%
projection, which means the annotations and camera geometry are usable, but
the oracle is not a deployable candidate generator.

Tracker confidence reaches 0.533 mAP and candidate distance reaches 0.446 mAP
on the official replay pool. These are data-source controls, not DriveOne
results. Unmatched rows stay `UNMATCHED_TRACK`; the ranking diagnostic treats
them as explicit non-referred tracker false positives without calling them
`OTHER_OBJECT`.

The v3 conclusion was conservative because it treated every locally ambiguous
geometric edge as an unknown and included synthetic ego rows in the controls.
It remains unchanged as historical evidence.

See [RefAV candidate-pool decision](../refav_candidate_pool_decision.md) and
the external report at
`/ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v3/candidate_source_audit.json`.

## Evaluation-correctness gate (v6 report)

The evaluator was corrected without changing candidate membership. A match is
now globally ambiguous only when an equal-cardinality, equal-cost assignment
exists; local multiple-edge cases remain diagnostics. Unknown labels remain
explicit, and synthetic ego rows are excluded from external controls.

The corrected v6 report is outside Git at
`/ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v6/candidate_source_audit.json`.
The causal pool has 98.3%, 99.0%, and 99.1% availability at 1 m, 2 m, and 4
m, with a 1.07-point range. Candidate hashes match the v3 report. External
controls are no longer near-perfect after correction. The decision is
`POOLED_BASELINE_GATE_READY`.

Next: rerun candidate-only, metadata-only, task-ID, and pooled-PE controls on
the causal pool with one seed, then a second seed if reproducible. Report both
operational unknown-retained and labeled-only metric bounds. Do not add patch
tokens, Qwen, temporal input, trajectories, or distillation yet.
