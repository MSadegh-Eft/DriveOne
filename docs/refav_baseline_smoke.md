# RefAV frozen-PE baseline smoke test

This note records the first learned-baseline check for the custom **RefAV
referred-track ranking** task. It is a small diagnostic, not a DriveOne result
and not evidence about driving safety, planning, or grounding.

## What was run

- Nine-log, log-disjoint repeated-prompt plan.
- 200 rankable groups per split: 600 groups total.
- All tracker candidates in each selected group were retained, including
  unmatched candidates.
- One shared `ring_front_center` image was used for each group.
- Frozen `PE-Core-L14-336` image and text features were extracted once on host
  GPU 2. The model was not fine-tuned.
- Four controls were trained with one seed (`0`): candidate-only,
  metadata-only, learned task ID, and pooled PE.
- The loss was binary cross-entropy on labeled candidates. Unknown candidates
  were omitted from the loss but retained when ranking the full candidate pool.
- Metrics were computed per group: multi-positive mAP, Recall@1, NLL, Brier,
  and ECE.

The exact subset and feature files remain outside Git. Their paths and hashes
are recorded in `configs/refav_pilot.yaml` and in the external artifact
directory under `$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200`.

## PE interface used

The real-image smoke report confirms that `PE-Core-L14-336` returns 576 patch
tokens of width 1024, a pooled image vector of width 1024, and a text vector of
width 1024. The official CLIP text context is 32 tokens. This test used pooled
image and text vectors only; it did not train a patch-token fusion block.

## Results

| model | train mAP | validation mAP | test mAP | test Recall@1 | test NLL | test Brier | test ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate-only | 0.0849 | 0.0479 | 0.0391 | 0.000 | 0.3450 | 0.0845 | 0.0817 |
| metadata-only | 0.0511 | 0.0803 | 0.0491 | 0.000 | 0.3276 | 0.0808 | 0.0538 |
| task ID | 0.1633 | 0.0600 | 0.0855 | 0.025 | 0.4078 | 0.0987 | 0.1156 |
| pooled PE | 0.1841 | 0.0873 | 0.0967 | 0.035 | 0.4334 | 0.1111 | 0.1522 |

The deterministic controls on the same exact test groups reach mAP 0.1674 for
tracker score and 0.1678 for projected box area. Therefore pooled PE is ahead
of the learned task-ID and simple learned controls in this smoke test, but it
does not beat the strongest deterministic controls. Its test calibration is
also worse: ECE is 0.1522 versus 0.0538 for the metadata-only model.

## Larger one-seed replication

The same protocol was rerun with 500 groups per split. This uses a new fixed
subset from the same nine logs, not a second random seed. The held-out test
results are:

| model/control | test mAP | test Recall@1 |
| --- | ---: | ---: |
| random ranking | 0.0395 | 0.012 |
| candidate-only | 0.0782 | 0.038 |
| metadata-only | 0.0379 | 0.000 |
| learned task ID | 0.0437 | 0.006 |
| pooled PE | 0.0511 | 0.012 |
| tracker-score ranking | 0.1813 | 0.046 |
| projected-box-area ranking | 0.1729 | 0.158 |

The pooled-PE result is lower than both deterministic controls, and the
candidate-only model is also far below them. This makes the first negative
signal reproducible at a larger subset. It still does not prove that visual
features can never help; it shows that this candidate protocol and current
pooled baseline do not support the proposed gain.

## Decision

Do **not** add the patch-token scorer. The larger one-seed replication confirms
the initial stop signal. This is not a final rejection of every possible
language-conditioned ranker, but it is a failure of the current first-stage
gate: pooled PE does not beat the strongest matched deterministic controls.
The next work should be protocol and candidate redesign, not more model
capacity.

The next allowed experiment is a baseline replication/debugging gate:

1. Audit per-log and candidate-count strata for the 500-group result.
2. Inspect whether the label transfer and candidate generator make visual
   appearance redundant or whether tracker geometry is acting as a target
   proxy.
3. Redesign the candidate pool or label protocol only if a concrete,
   label-independent repair is available. Re-run deterministic controls after
   that repair.
4. Add patch tokens only after a repaired pooled baseline beats the strongest
   deterministic control under a preregistered margin.

## Reproduction commands

The following commands assume the external files already exist and the
`driveone-pe` environment is active. The paths are examples; use the paths in
the external manifest on the current machine.

```bash
python scripts/export_refav_subset.py \
  --records "$DRIVEONE_ASSET_ROOT/refav/refav_tracker_repeated_prompt9_repaired.feather" \
  --split-plan "$DRIVEONE_ASSET_ROOT/refav/refav_repeated_prompt_log_plan.json" \
  --output-dir "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200" \
  --groups-per-split 200 --seed 0

python scripts/extract_pe_features.py \
  --inputs "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200"/*.jsonl \
  --output "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/pe_core_pooled_features.pt" \
  --report "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/pe_core_pooled_features.json" \
  --config PE-Core-L14-336 --device cuda --batch-size 16 --pretrained

python scripts/train_refav_baselines.py \
  --train "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/refav_train_subset.jsonl" \
  --validation "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/refav_validation_subset.jsonl" \
  --test "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/refav_test_subset.jsonl" \
  --features "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/pe_core_pooled_features.pt" \
  --output "$DRIVEONE_ASSET_ROOT/refav/refav_baseline_subset_200/baseline_results_seed0.json" \
  --device cuda --epochs 12 --seed 0
```
