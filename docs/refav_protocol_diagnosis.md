# RefAV protocol-diagnosis gate

This note records the CPU-only diagnosis of the fixed 500-group-per-split
RefAV referred-track ranking subset. It does not train a model and it does not
change the candidate pool. Its purpose is to check whether tracker metadata,
visibility, or geometry can predict the transferred RefAV label.

## Inputs

The run used the existing external artifacts:

- corrected frozen-PE baseline results, SHA-256
  `3d9f83776f3ed7713b00a474a940e149319c1b0086addf13bf770c659e3605b8`;
- the three fixed JSONL subsets and their control JSON files under
  `/ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500`;
- the fixed-subset manifest;
- the repeated-prompt nine-log split plan.

The complete machine-readable output is outside Git:

```text
/ehsan/m.sadegh/driveone_assets/refav/refav_protocol_diagnosis_500.json
SHA-256: 8c705b9aba767253bf3fa5c279e32d03768c1684d4476a095a058b004ebdc0ad
```

Run command:

```bash
conda run -n refav python scripts/diagnose_refav_protocol.py \
  --subset-root /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500 \
  --baseline-results /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/baseline_results_seed0_v2.json \
  --controls-dir /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500 \
  --subset-manifest /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/refav_subset_manifest.json \
  --split-plan /ehsan/m.sadegh/driveone_assets/refav/refav_repeated_prompt_log_plan.json \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_protocol_diagnosis_500.json
```

## What the script checks

For each split it reports:

- group count, rankable-group count, candidate-count distribution, and unknown
  row rate;
- oracle candidate coverage, meaning the fraction of selected groups containing
  at least one referred candidate;
- referred, related, other, and unknown row counts;
- projected-box status, match status, and object-category counts by label;
- distributions of projected area, tracker score, distance, match distance,
  and cuboid size by label;
- deterministic control mAP and Recall@1 per log and candidate-count range;
- full-pool, score-matched, and score-plus-size-matched control results;
- the existing learned-baseline results by log and candidate-count range.

The oracle ranking is reported as an upper bound. It is never used as evidence
that a shortcut beats the model.

## Main observations

| split | groups | rows | unknown rows | referred visible | labeled negatives visible | visibility gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 500 | 144,580 | 84.4% | 15.3% | 26.6% | -11.3 points |
| validation | 500 | 112,276 | 89.8% | 38.6% | 16.4% | +22.2 points |
| test | 500 | 100,192 | 89.8% | 46.9% | 21.0% | +26.0 points |

The visibility relationship changes by split. This means the shared front-center
camera is not a stable source of evidence: in validation and test, referred
objects are much more likely to have a projected box than labeled negatives;
in train, the direction is reversed. This variation is itself a warning sign
for generalization.

The selected subset has 100% oracle candidate coverage because the exporter
intentionally selected rankable groups. This does **not** describe the full
manifest; it only says that the selected groups contain a positive and a
labeled negative. Full-manifest candidate coverage must remain a separate
measurement.

### Full-pool controls

| split | random | tracker score | candidate distance | projected area | category frequency | pooled PE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.0275 | 0.1562 | 0.1200 | 0.0422 | 0.0201 | 0.2669 |
| validation | 0.0393 | 0.5341 | 0.0870 | 0.1682 | 0.0142 | 0.0531 |
| test | 0.0395 | 0.1813 | 0.0285 | 0.1729 | 0.0885 | 0.0476 |

The train pooled-PE score is a training-set score and is not evidence of
generalization. On validation and test, deterministic tracker/geometry
controls are much stronger than pooled PE. The test values match the corrected
baseline record.

### Score-and-size-matched controls

The score-and-size subset is an evaluation diagnostic. It is not presented as a
deployment-like candidate pool because it is constructed using labeled
positive/negative rows.

| split | random | tracker score | candidate distance | projected area | category frequency |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.3835 | 0.3281 | 0.4295 | 0.3870 | 0.3391 |
| validation | 0.6630 | 0.8138 | 0.6421 | 0.6699 | 0.6786 |
| test | 0.5587 | 0.5294 | 0.4020 | 0.6069 | 0.6133 |

The matched subset has far fewer groups than the full subset. Its purpose is to
ask whether simple metadata remains predictive after observable score and size
are made more similar. The strongest deterministic controls remain much above
the pooled-PE test mAP, so the current result still cannot identify a visual
language gain.

## Decision

The diagnosis reports:

```text
PROTOCOL_REPAIR_REQUIRED
```

This is not a final claim that RefAV is unusable. It is a stop on the current
model path. The next work must design and test a candidate-pool repair that is
label-independent, deterministic, hashable, and shared by every method. If no
credible repair exists, stop the RefAV branch rather than adding more model
capacity.

Do not start patch tokens, PE-Spatial, temporal frames, Qwen, distillation,
trajectories, or deployment optimization yet.

## Next gate

1. Inspect whether label transfer and tracker matching create the visibility
   difference.
2. Define one candidate-pool repair without looking at relevance labels.
3. Rebuild or re-export the repaired rows and hash them.
4. Rerun deterministic controls on exactly those rows.
5. Only if the controls are no longer competitive, run the second-seed pooled
   baseline and confidence-interval check.

PROTOCOL_REPAIR_REQUIRED
