# DriveOne

A standalone research repository for the DriveOne bounded-candidate autonomous-driving scorer.

The first milestone is deliberately narrow: establish a reproducible, non-distilled RefAV referred-track ranking diagnostic with frozen PE-Core-L14-336 features. Distillation, PE-Spatial, temporal frames, trajectory ranking, and deployment optimization come later, only after the data contract, candidate leakage controls, and frozen-feature smoke tests pass.

For a complete explanation in simple English, start with the [DriveOne Project Guide](docs/DriveOne_Project_Guide.md). A locally generated PDF copy is available at `reports/DriveOne_Project_Guide.pdf`; the PDF is ignored by Git so it is not uploaded with the source repository.

## Current next step

The evaluation-correctness gate passed on the nine-log candidate pool. It
separates global assignment certainty from local nearby matches, preserves
unknown labels, excludes synthetic ego rows from external controls, and keeps
the learned-model gate separate. The corrected two-seed frozen pooled-PE gate
has now run and returns `POOLED_BASELINE_GATE_FAILED`.

Pooled PE reached 0.0668/0.0569 test mAP for seeds 0/1. Task ID reached
0.0787/0.0718, metadata-only reached 0.0830/0.0810, and fixed front-center
projected area reached 0.1964. Pooled PE also had worse seed-0 calibration
than task ID. Do not add patch tokens, Qwen, temporal frames, trajectories, or
distillation yet.

The full gate record is in
[`docs/refav_pooled_baseline_gate.md`](docs/refav_pooled_baseline_gate.md).
The detailed JSON report and large result files remain outside Git under
`/ehsan/m.sadegh/driveone_assets/refav/refav_baseline_gate_v1/`.

The repair entry point is:

```bash
conda run -n refav python scripts/repair_refav_protocol.py \
  --tracker /ehsan/m.sadegh/driveone_assets/refav/official/Le3DE2E_tracking_predictions_val.pkl \
  --annotations /ehsan/m.sadegh/driveone_assets/refav/official/scenario_mining_val_annotations.feather \
  --sensor-root /ehsan/m.sadegh/driveone_assets/refav/av2_sensor_clean/val \
  --logs <nine-log-ids> \
  --output /ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired.feather \
  --summary /ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired_summary.json \
  --manifest /ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired_manifest.json
```

The previous Valeo4Cast artifacts and controls remain available as historical
diagnostics.  They are not the final source for the protocol gate.

The exact reading and experiment order is in [`docs/obsidian/Development Roadmap.md`](docs/obsidian/Development%20Roadmap.md).

The corrected official Le3DE2E candidate-source audit reproduces the official
conversion and uses all seven camera views at the native tracker timestamps.
The causal pool provides 99.0% external referred-object availability at 2 m;
the ground-truth oracle provides 100%. Tracker confidence remains a required
control, but no external deterministic control is near-perfect after ego rows
are excluded. See
[`docs/refav_candidate_pool_decision.md`](docs/refav_candidate_pool_decision.md).

The six-log deterministic control run is available outside Git:

```bash
conda run -n refav python scripts/run_refav_controls.py \
  --records /ehsan/m.sadegh/driveone_assets/refav/refav_tracker_sixlog_repaired.feather \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_controls_sixlog_v3.json \
  --seeds 0 1 \
  --hard-negative-delta 0.05 \
  --size-matched-log-area-delta 0.2
```

It found strong tracker-confidence and projected-size shortcuts. The repeated-
prompt nine-log split is now built and audited; shortcut strength varies by
split, so results must remain stratified. A first frozen-PE pooled and task-ID
smoke test is recorded in
[`docs/refav_baseline_smoke.md`](docs/refav_baseline_smoke.md). Pooled PE does
not yet beat the strongest deterministic controls, so patch-token modeling is
paused while the baseline and shortcut protocol are replicated on a larger
fixed subset.

The corrected 500-group-per-split replication is retained as historical
evidence: pooled PE
reaches 0.0476 test mAP and task ID with the same pooled image reaches 0.0576,
while tracker-score and projected-box-area controls reach 0.1813 and 0.1729 on
the same candidates. The CPU-only protocol diagnosis is now complete and is
recorded in [`docs/refav_protocol_diagnosis.md`](docs/refav_protocol_diagnosis.md).
The new evaluator-correctness report supersedes that data stop; the next work is
the corrected pooled baseline, not a larger model.

Prepare the pinned public Valeo4Cast validation tracks into the derived pilot
manifest, then run the dependency-light RefAV feasibility verifier:

```bash
conda run -n refav python scripts/prepare_refav_tracker.py \
  --tracker /ehsan/m.sadegh/driveone_assets/refav/extracted/data/track/for_val_and_test/val_tracking.pkl \
  --annotations /data/sadegh/driveone/data/refav/pilot/refav_val_two_logs.feather \
  --sensor-root /ehsan/m.sadegh/driveone_assets/refav/av2_sensor_clean/val \
  --logs 20dd185d-b4eb-3024-a17a-b4e5d8b15b65 02a00399-3857-444e-8db3-a8f58489c394 \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_tracker_pilot_repaired.feather \
  --summary /ehsan/m.sadegh/driveone_assets/refav/refav_tracker_pilot_repaired_summary.json

conda run -n refav python scripts/verify_refav.py \
  --records /ehsan/m.sadegh/driveone_assets/refav/refav_tracker_pilot_repaired.feather \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_tracker_pilot_repaired_audit.json \
  --strict
```

The verifier does not download the full Argoverse 2 sensor dataset. The
manifest is a custom referred-track ranking diagnostic, not an official RefAV
leaderboard submission.

## Initial milestone

- Pin the RefAV artifact, repository commit, AV2 split manifest, and file hashes.
- Verify the official multi-object, multi-timestamp schema.
- Freeze one tracker candidate artifact and construct identical candidate pools.
- Use log-disjoint splits and held-out query templates.
- Compare data-only controls before implementing the patch-token scorer.
- Report candidate oracle coverage, ranking quality, calibration, leakage controls, and end-to-end timing.

## Repository layout

- `src/driveone/` — model and data code
- `configs/` — versioned experiment configurations
- `scripts/` — reproducible entry points
- `tests/` — unit and data-contract tests
- `docs/` — experiment notes and protocol decisions
- `experiments/` — run metadata and results (large artifacts stay outside Git)

Dataset files, model weights, caches, and generated artifacts are intentionally excluded from this repository.

## PE smoke test

After installing the official `perception_models` environment, run the interface check with an actual image:

```bash
python scripts/smoke_test_pe.py --image /path/to/frame.jpg --pretrained --output runs/pe_smoke.json --strict
```

The output records the pre-pooling sequence shape, the sequence with the class token removed, the pooled output shape, dtype, device, and measured forward times. Do not treat a run without `--pretrained` as evidence about the released checkpoint.
