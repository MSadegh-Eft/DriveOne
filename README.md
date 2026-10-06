# DriveOne

A standalone research repository for the DriveOne bounded-candidate autonomous-driving scorer.

The first milestone is deliberately narrow: establish a reproducible, non-distilled RefAV referred-track ranking diagnostic with frozen PE-Core-L14-336 features. Distillation, PE-Spatial, temporal frames, trajectory ranking, and deployment optimization come later, only after the data contract, candidate leakage controls, and frozen-feature smoke tests pass.

## Current next step

The public-tracker pilot manifest now exists outside Git. Before adding a
DriveOne scorer, reproduce that manifest, inspect its audit, and run the
control-only ranking study. The pilot has many unmatched tracker rows, so do
not treat an unmatched row as a negative without an explicit protocol.

The exact reading and experiment order is in [`docs/obsidian/Development Roadmap.md`](docs/obsidian/Development%20Roadmap.md).

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
split, so results must remain stratified. The next step is the frozen-PE pooled
and task-ID controls before adding patch-token fusion.

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
