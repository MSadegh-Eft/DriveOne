# Corrected frozen pooled-PE gate

Date: 2026-10-08

This step tested the smallest learned DriveOne-like model on the repaired
Le3DE2E RefAV candidate pool. It used one front-center image, frozen
`PE-Core-L14-336` features, and a small scorer. It did not use patch tokens,
temporal frames, Qwen, trajectories, or distillation.

## Why this test was run

The previous data audit said that the Le3DE2E candidate pool was label
independent and reproducible. Before adding a more complex model, we needed to
check whether pooled image and question features help at all. The test used
the same candidate rows for every method and kept unknown candidates in the
ranking pool.

Unknown rows were not treated as negative labels. They stayed in the ranking
list, but they were omitted from the BCE training loss and from the
labeled-only calibration denominator.

## Inputs and code

- `src/driveone/data/refav_baseline_export.py` joins the v6 candidate pool,
  labels, pose data, tracker metadata, and the fixed `ring_front_center`
  image. It verifies the pool hash before labels are attached.
- `scripts/export_refav_baseline.py` creates the three prompt-expanded
  JSONL/Feather splits and the manifest.
- `scripts/train_refav_baselines.py` trains four frozen-feature controls:
  candidate-only, metadata-only, task ID plus pooled image, and natural
  language plus pooled image/text.
- `src/driveone/eval/bootstrap.py` computes paired per-group bootstrap
  intervals.
- `scripts/compare_refav_baseline_seeds.py` compares the two seeds.
- `scripts/report_refav_baseline_gate.py` creates the external JSON report.

The complete data and result files are outside Git at:

`/ehsan/m.sadegh/driveone_assets/refav/refav_baseline_gate_v1/`

The main report is `baseline_gate_report.json`. It records paths, hashes,
timings, stratified metrics, and the final decision.
Its SHA-256 is
`6f686a4d4c3a8a9c923bc4809737a51d1cbdeca8b13c2f5e7fb958509da3640c`.

## Protocol

There are 2,880 groups: 960 train, 960 validation, and 960 test. The split
unit is `log_id`, so no log occurs in more than one split. Each group has one
prompt and one timestamp. Its candidates are the complete v6 causal tracker
pool at that timestamp. Prompt expansion repeats the same pool for several
queries; these repeated rows are expected and do not mean that the source pool
has duplicate candidates.

The source-level pool has no duplicate
`(log_id, timestamp_ns, track_id)` keys. The export kept the source pool hash
unchanged after adding prompts and labels. Candidate rows were shuffled in a
fixed way before every model saw them.

The fixed image policy uses the nearest available `ring_front_center` frame.
It never chooses a camera because its box is largest. Out-of-view candidates
remain in the pool.

## Results on the test split

Operational mAP and Recall@1 keep unknown candidates as distractors. The
labeled-only metrics remove unknown candidates only for that diagnostic.

| Method | Seed | mAP | Recall@1 | labeled-only mAP | ECE |
|---|---:|---:|---:|---:|---:|
| Candidate-only | 0 | 0.0780 | 0.0405 | 0.2321 | 0.0559 |
| Metadata-only | 0 | 0.0830 | 0.0270 | 0.3242 | 0.0402 |
| Task ID + pooled image | 0 | 0.0787 | 0.0495 | 0.2737 | 0.0468 |
| Question + pooled PE image/text | 0 | 0.0668 | 0.0360 | 0.2277 | 0.1583 |
| Candidate-only | 1 | 0.0710 | 0.0315 | 0.2500 | 0.0405 |
| Metadata-only | 1 | 0.0810 | 0.0405 | 0.2631 | 0.0282 |
| Task ID + pooled image | 1 | 0.0718 | 0.0315 | 0.2190 | 0.2182 |
| Question + pooled PE image/text | 1 | 0.0569 | 0.0135 | 0.2197 | 0.1048 |
| Fixed front-center projected area | deterministic | 0.1964 | 0.1712 | 0.3554 | — |

The strongest deterministic control is projected box area. It is much stronger
than the learned pooled-PE model. The metadata-only learned control is also
slightly stronger than pooled PE on operational test mAP.

The required margin was at least +0.03 mAP or +0.05 Recall@1 over the strongest
matched baseline, with two-seed support. Pooled PE was below task ID by
0.0119 mAP (seed 0) and 0.0149 mAP (seed 1). It was below task ID in
Recall@1 by 0.0135 and 0.0180. The paired bootstrap intervals do not support a
positive margin. Pooled PE also had worse seed-0 ECE than task ID (0.1583 vs
0.0468).

## Coverage and what it means

The repaired source association is strong when an eligible ground-truth event
exists: 98.99% positive availability at the 2 m match threshold and 100%
positive projection across the downloaded seven-camera assets in the audit.
Across all timestamp groups, however, only 23.8% contain an available
positive; this is a property of the event sampling, not a tracker recall claim.

The fixed front-center baseline sees a projected positive in 46.0% of test
positive rows. This is much lower than seven-camera coverage. Therefore the
front-center result is a valid fixed-view diagnostic, but it is a weak basis
for claims about all-camera grounded ranking.

The main log-disjoint split has exact prompt overlap across logs. The
`joint_holdout_eligible` subgroup is reported separately and is small. It must
not be described as a full held-out-template evaluation.

## Timing limits

The baseline JSON records JSONL parsing, image-file validation, cached feature
loading, tensor construction, training, and cached-feature scoring. It does
not measure end-to-end deployment latency. In particular, it excludes image
decoding during deployment, PE image encoding, candidate construction,
calibration, and postprocessing. The adapter itself took 120.1 seconds to
build the prompt-expanded export. CPU extraction of the missing PE cache took
157.4 seconds for 86 images and 33 prompts.

## Decision

`POOLED_BASELINE_GATE_FAILED`

This does not prove that patch tokens can never help. It means that the
current pooled representation has not shown a useful gain, and the fixed-view
candidate protocol has limited visual coverage. Adding patch tokens now would
hide the simpler result instead of testing a clear hypothesis.

The next action is a design decision: either define a fair multi-camera
candidate representation and rerun the pooled baseline with that same input,
or record this RefAV branch as a negative result and move to a task with a
better candidate interface. Do not start Qwen, distillation, temporal input,
trajectories, or deployment optimization from this failed gate.
