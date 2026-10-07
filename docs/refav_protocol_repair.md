# RefAV protocol repair

## Why this step exists

The first 500-group diagnosis used a Valeo4Cast-derived validation tracker
file.  Tracker score and projected area were strong predictors of the transferred
RefAV labels, and most tracker rows were unmatched.  Adding patch tokens would
not tell us whether the improvement came from the image or from the candidate
construction.

The official RefAV tutorial uses the Le3DE2E validation predictions from the
`CainanD/AV2_Tracker_Predictions` dataset.  This repair rebuilds the pilot from
that source before any new model is trained.

## Contract

The pipeline has two separate stages:

1. **Candidate construction.** For each aligned log and timestamp, keep every
   eligible tracker row after only fixed infrastructure filters: valid geometry,
   supported taxonomy, distance under 50 m, fixed ROI, and available pose or
   camera information.  Prompt text and RefAV relevance labels are not read in
   this stage.
2. **Target association.** Match the already-built pool to ground-truth boxes
   with a fixed semantic taxonomy map and one-to-one Hungarian assignment.  The
   prompt-specific `REFERRED_OBJECT`, `RELATED_OBJECT`, and `OTHER_OBJECT`
   labels are copied only after matching.

Unmatched rows remain explicit unknowns.  They are not converted to
`OTHER_OBJECT`.  The primary pool is hashed from candidate identity and
construction fields before labels are attached.

## Camera policy

Every candidate is projected into the seven ring cameras in fixed order.  For
each camera, the nearest image within 100 ms is selected.  Le3DE2E tracker
frames are sampled more slowly than the scenario annotations, so each
annotation timestamp is paired with the nearest tracker frame within a fixed
250 ms tolerance.  This timestamp-only association is recorded explicitly and
does not inspect relevance labels.  Out-of-view and
missing-frame statuses remain in the output.  The pipeline never chooses the
camera with the largest box or best visibility.  The front-center projection is
kept only as a compatibility field for older readers; `camera_projections` is
the authoritative field.

## Source pins

| Artifact | Source | Required record |
| --- | --- | --- |
| Scenario annotations | [RefAV dataset](https://huggingface.co/datasets/CainanD/RefAV) | file URL, revision, SHA-256, license |
| Tracker predictions | [AV2 Tracker Predictions](https://huggingface.co/datasets/CainanD/AV2_Tracker_Predictions) | `Le3DE2E_tracking_predictions_val.pkl`, revision, SHA-256 |
| Protocol reference | [RefAV repository](https://github.com/cainand/refav) | repository commit |

The expected tracker revision and hash are pinned in
`configs/refav_pilot.yaml`; the repair manifest records the actual values used.

## Reproduction

Use the official tracker and scenario-mining files outside Git.  Select the
same nine logs as the repeated-prompt pilot and provide their AV2 sensor
directories:

```bash
conda run -n refav python scripts/repair_refav_protocol.py \
  --tracker /ehsan/m.sadegh/driveone_assets/refav/official/Le3DE2E_tracking_predictions_val.pkl \
  --annotations /ehsan/m.sadegh/driveone_assets/refav/official/scenario_mining_val_annotations.feather \
  --sensor-root /ehsan/m.sadegh/driveone_assets/refav/av2_sensor_clean/val \
  --logs <nine-log-ids> \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_le3de2e_repaired.feather \
  --summary /ehsan/m.sadegh/driveone_assets/refav/refav_le3de2e_repaired_summary.json \
  --manifest /ehsan/m.sadegh/driveone_assets/refav/refav_le3de2e_repaired_manifest.json
```

The output is a derived research artifact, not an official RefAV leaderboard
submission.  The old Valeo manifest stays available for comparison.

## Acceptance gate

Continue to the second-seed pooled baseline only when the manifest shows:

- no duplicate candidate keys and a stable candidate-pool hash;
- no prompt or relevance-label influence on candidate membership or count;
- at least 80% positive availability on selected timestamp groups;
- at least 90% of matched positives projected in one or more cameras;
- no conclusion-changing sensitivity to 1 m, 2 m, and 4 m matching thresholds;
- no near-perfect score, distance, size, area, category, or visibility control.

If these conditions fail, keep the result as a protocol diagnosis and stop the
RefAV model branch.  Do not add patch tokens, temporal input, Qwen, trajectories,
or distillation before this gate passes.

## Official nine-log result

The repair was run on the nine logs from the repeated-prompt plan using the
official Le3DE2E tracker and official RefAV validation annotations.  The derived
Feather file is stored outside Git:

```text
/ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired.feather
SHA-256: b8955f0b811d3a350c667d3ccae6f2b6cc344dd2f6537ce001b7f1a36839ad3d
```

The candidate pool contained 333,255 unique tracker candidates.  The labelled
view contains 14,110 prompt/timestamp groups and 3,332,550 candidate rows,
because the same pool is repeated for each prompt.  Unknown rows account for
90.5% of the labelled view.  Only 2,935 of 14,110 groups (20.8%) had at least
one tracker candidate matched to a `REFERRED_OBJECT`.  Only 33.2% of the
matched referred rows projected into at least one camera.  Matching-threshold
sensitivity produced positive availability of 18.9%, 20.8%, and 21.8% at 1 m,
2 m, and 4 m, respectively.  These values are
well below the 80% positive-availability and 90% positive-projection gates.

The full-pool ranking controls were therefore not presented as a valid
benchmark result.  A conditional 500-rankable-group diagnostic was run only to
show that the repaired artifact can be consumed by the existing control code;
its results are not deployment-like because selecting rankable groups uses the
transferred labels.

The machine-readable assessment is:

```text
/ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_assessment.json
decision: PROTOCOL_REPAIR_REQUIRED
```

The official tracker-to-referred coverage and camera coverage already fail the
gate.  Do not begin patch-token modeling or distillation on this branch.
