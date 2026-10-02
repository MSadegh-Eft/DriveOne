# DriveOne

A standalone research repository for the DriveOne bounded-candidate autonomous-driving scorer.

The first milestone is deliberately narrow: establish a reproducible, non-distilled RefAV track-ranking baseline with frozen PE-Core-L14-336 features. Distillation, PE-Spatial, temporal frames, trajectory ranking, and deployment optimization come later, only after the baseline and leakage controls are validated.

## Initial milestone

- Freeze the dataset manifest and candidate construction.
- Use log-disjoint splits and held-out query templates.
- Compare candidate-only, task-ID, pooled-PE, and patch-token scorers.
- Report ranking quality, calibration, shortcut controls, and end-to-end timing.

## Repository layout

- `src/driveone/` — model and data code
- `configs/` — versioned experiment configurations
- `scripts/` — reproducible entry points
- `tests/` — unit and data-contract tests
- `docs/` — experiment notes and protocol decisions
- `experiments/` — run metadata and results (large artifacts stay outside Git)

Dataset files, model weights, caches, and generated artifacts are intentionally excluded from this repository.
