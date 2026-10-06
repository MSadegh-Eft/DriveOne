# Current Status and Next Steps

## What has been completed

- The repository was created as an isolated DriveOne project.
- The data contract, verifier, tracker adapter, PE smoke test, controls, tests, and documentation exist.
- Official RefAV metadata, annotations, tracker output, and AV2 sensor assets were inspected.
- The code passes fifteen tests in the `refav` environment.
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

## Next task

1. Download only the nine planned logs' front-center camera, pose, calibration, map, and
   annotation assets.
2. Re-run the fixed score-and-size-matched controls on those groups.
3. Only if shortcut controls are no longer sufficient should the pooled-PE,
   task-ID, and patch-token models be implemented.

## Stop rules

Stop and redesign if candidate construction cannot be independent of relevance
labels, target coverage is poor, the shared camera removes most targets, or a
candidate-only shortcut remains competitive after matched hard negatives.

## Commit history

- `4976bff` — initial repository layout.
- `a07a511` — RefAV data contract and audit primitives.
- `4c4a249` — RefAV feasibility and PE smoke-test workflow.
- `2531f0d` — public tracker adapter, pilot manifest, and strict audit updates.
- `db7d19b` — PE smoke findings.
- `7930e77` — simplified Obsidian roadmap and corrected PE documentation.
- `cfb2d1e` — label-independent timestamps and shared-camera protocol.
- `a75b1b8` — leakage-aware deterministic control suite.
