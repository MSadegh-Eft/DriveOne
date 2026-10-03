# DriveOne

A standalone research repository for the DriveOne bounded-candidate autonomous-driving scorer.

The first milestone is deliberately narrow: establish a reproducible, non-distilled RefAV referred-track ranking diagnostic with frozen PE-Core-L14-336 features. Distillation, PE-Spatial, temporal frames, trajectory ranking, and deployment optimization come later, only after the data contract, candidate leakage controls, and frozen-feature smoke tests pass.

## Current next step

Run the dependency-light RefAV feasibility verifier before downloading large data or training a model:

```bash
python scripts/verify_refav.py
python scripts/verify_refav.py --records /path/to/local/refav_export.pkl --output runs/refav_manifest.json --strict
```

Before the strict command can pass, replace the `PIN_BEFORE_RUN` values in `configs/refav_pilot.yaml` and provide the official log split manifest. The first command is expected to report `not_ready` until a small official scenario-mining export is placed under `data/refav/`. The verifier does not download the full Argoverse 2 sensor dataset.

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
