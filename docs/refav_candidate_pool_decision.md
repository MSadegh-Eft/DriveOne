# RefAV candidate-pool decision

## Current status

The evaluation-correctness gate has been rerun on the same nine logs and the
same candidate rows. No model was trained and no new data was downloaded.

```text
POOLED_BASELINE_GATE_READY
```

This means the **data and association checks are now good enough to run the
frozen pooled-PE baseline**. It does not mean that DriveOne has passed its model
quality gate.

The machine-readable report is outside Git:

```text
/ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v6/candidate_source_audit.json
SHA-256 52aff5c738bd674969c302f0fde3572b2e21e134a99e843b1182f7290157e9e7
```

The previous v3 report remains unchanged. Its conclusion was too conservative
because it treated every locally ambiguous match as an unknown label and mixed
synthetic ego rows into the external controls. It is retained as an audit
history, not overwritten.

## What was corrected

1. **Global assignment ambiguity.** A candidate with more than one valid local
   geometric edge is now recorded as a diagnostic (`multiple_valid_edges`). It
   becomes `AMBIGUOUS_MATCH` only when removing its assigned edge leaves another
   equal-cardinality, equal-cost global assignment. Nearby objects therefore do
   not automatically count as tracker misses.
2. **Unknown labels.** Candidate rows retain `label=None` and their original
   `match_status`. The controls report two bounds: labeled-only metrics exclude
   unknown rows from the metric denominator, while pessimistic operational
   metrics keep unknown rows in ranking order so they can push a positive down.
3. **Ego separation.** Synthetic `EGO_VEHICLE` rows remain in the candidate
   pool and its hash, but are excluded from external-object controls. Ego
   scenarios are counted separately.
4. **Data/model separation.** The report has a data gate and a model gate. The
   second-seed pooled model is marked `PENDING`; it cannot cause a data audit to
   fail before training is authorized.
5. **Threshold arithmetic.** The official replay changes from 71.7% at 1 m to
   59.2% at 4 m in the historical v3 report, a 12.4-point range. The 20.8-point
   range belonged to the old causal matching policy. Under the corrected global
   assignment, the causal range is 1.07 points and the official replay range is
   0.87 points.

## Candidate and camera policy

Candidate membership was not changed. The audit still uses all eligible rows
at each native tracker timestamp, finite positive geometry, the fixed 50 m and
ROI filters, and a deterministic pool hash. Labels are attached only after the
pool is complete. All seven ring cameras use the fixed nearest-frame policy
within 100 ms, and out-of-view rows remain in the pool.

The official RefAV repository is pinned to commit
`5c5be6439ce59b61a31d56431a79a8a04bba33fa`. The official tutorial uses
Le3DE2E validation predictions:
<https://raw.githubusercontent.com/CainanD/RefAV/main/run/tutorial.ipynb>.
The source repository is <https://github.com/cainand/refav>.

The final input artifacts remain:

- `scenario_mining_val_annotations.feather`, SHA-256
  `e461e51057fdf347a11bdd60609e7de0b0bd8d0eb199314b3fd73c50106d48c9`;
- `Le3DE2E_tracking_predictions_val.pkl`, SHA-256
  `fd702ade8b640d325e5096e90f63cf1bf43f94a87a6aabfb52e71aa7b8470875`;
- repeated-prompt plan, SHA-256
  `cba73f6f34a5373bf3a5f69765b78f5eeebf48ad2ebdab38032a204cd7c60f85`.

The corrected per-log candidate hashes are identical to v3 for all five source views.
This confirms that the evaluator correction did not change candidate
membership.

## Corrected coverage and association

The audit contains 2,880 prompt/timestamp groups. External availability is
conditional on an eligible external ground-truth referred object; it is not the
fraction of all timestamps that contain a scenario event.

| source | external referred availability at 2 m | positive projection | unknown candidate fraction |
| --- | ---: | ---: | ---: |
| causal Le3DE2E pool | 99.0% | 100.0% | 88.7% |
| official replay | 97.0% | 100.0% | 74.1% |
| historical pool | 99.0% | 100.0% | 88.7% |
| ground-truth oracle | 100.0% | 100.0% | 0.0% |

The causal pool’s conservative group availability is 98.3%, 99.0%, and 99.1%
at 1 m, 2 m, and 4 m. The maximum change is 1.07 percentage points. The
official replay is 96.4%, 97.0%, and 97.3%, with a 0.87-point range.

The causal pool still has many unmatched tracker rows: 603,230 of 680,040
candidate rows (88.7%). This is not the same as saying that 88.7% of referred
objects are missing. Most eligible referred groups have at least one tracker
candidate assigned under the corrected global matching. Unmatched rows remain
important because they enlarge the candidate list and can affect operational
ranking.

There are many local geometric alternatives, but no equal-cost global
assignment ties in the corrected nine-log report. At 2 m the causal pool has
3,964 assigned candidates with more than one local valid edge; the corresponding
number for the official replay is 3,528. These are crowded-scene diagnostics,
not proof that the rows are duplicate tracker proposals. No NMS has been
applied.

## Corrected external controls

Controls were recomputed after removing synthetic ego rows. Each control uses
the same external candidate rows. The controls report both metric bounds:

- **operational:** unknown rows stay in the ranking order;
- **labeled-only:** unknown and ambiguous rows are excluded from the metric
  denominator.

For the causal Le3DE2E pool:

| control | operational mAP | labeled-only mAP | operational Recall@1 | labeled-only Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| tracker confidence | 0.264 | 0.274 | 0.166 | 0.169 |
| projected box area | 0.143 | 0.289 | 0.110 | 0.164 |
| candidate distance | 0.078 | 0.242 | 0.056 | 0.123 |
| category frequency | 0.027 | 0.226 | 0.006 | 0.129 |
| random | 0.035 | 0.222 | 0.010 | 0.114 |

The official replay has the same qualitative pattern: tracker confidence is the
strongest operational shortcut (mAP 0.272), while projected area is strong on
labeled-only rows (mAP 0.295). None of these controls is near-perfect after the
evaluator correction. They are still mandatory baselines for the pooled model.

The output also includes per-log, split, candidate-count, and prompt-holdout
stratifications. The strongest controls vary by log, so later model results must
not be reported only as one pooled number.

## Decision and next step

The corrected data gate passes:

- candidate membership is independent and hash-stable;
- projection coverage is complete for the selected camera assets;
- external referred availability exceeds 80%;
- matching stability is within 10 percentage points;
- unknown statuses are explicit;
- synthetic ego rows no longer inflate external controls;
- the model gate is separately marked pending.

Therefore the next experiment is the **frozen pooled-PE baseline gate** on the
causal all-tracker pool, with the same candidate rows for every method:

1. reproduce the one-seed pooled-PE, task-ID, candidate-only, and metadata-only
   controls on the corrected external groups;
2. run a second seed only after the one-seed result is reproducible;
3. report both operational and labeled-only mAP/Recall@1, per-log and
   candidate-count results, hard-negative results, NLL/Brier/ECE where the label
   policy is defined, and end-to-end timing scope;
4. compare against tracker score, distance, projected area, category, and
   random controls;
5. test patch tokens only if pooled PE beats the strongest matched baseline by
   at least +5 Recall@1 points or +0.03 mAP with two-seed confidence intervals
   excluding zero and no material calibration regression.

Do not apply NMS merely because local matching alternatives exist. First inspect
crowded examples. If a future NMS or track-consistency variant is tested, it
must be label-independent, deterministic, separately hashed, and compared as a
candidate-pool variant.

Do not run PE-Spatial, temporal frames, Qwen, distillation, trajectories, or
deployment optimization before the corrected pooled baseline gate passes.

## Reproduction

```bash
TMPDIR=/data/sadegh/tmp conda run --no-capture-output -n refav \
  python scripts/audit_refav_candidate_sources.py \
  --output-dir /ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v6
```

The audit code, tests, configuration, and this decision record are tracked in
Git. Large candidate files, camera images, controls, and JSON reports remain
outside Git.
