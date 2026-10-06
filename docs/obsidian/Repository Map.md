# Repository Map

| Path | Purpose | Depends on |
| --- | --- | --- |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata and optional development/PE dependencies | Python packaging |
| [`src/driveone/data/refav_contract.py`](../../src/driveone/data/refav_contract.py) | Normalization, loading, schema inspection, leakage checks, split checks, and manifest hashing | Python standard library; pandas/pyarrow only for Feather |
| [`src/driveone/data/refav_tracker.py`](../../src/driveone/data/refav_tracker.py) | Converts public Valeo4Cast city-frame tracks to ego-frame candidates, transfers prompt labels, applies AV2 ROI/range filtering, and projects candidates into ring-camera images | `numpy`, `pandas`, `scipy`, `av2` in the `refav` environment |
| [`src/driveone/eval/refav_metrics.py`](../../src/driveone/eval/refav_metrics.py) | Computes deterministic ranking controls while keeping unmatched candidates explicit | `numpy` |
| [`src/driveone/data/__init__.py`](../../src/driveone/data/__init__.py) | Public exports for the data helpers | `refav_contract.py` |
| [`scripts/verify_refav.py`](../../scripts/verify_refav.py) | CLI wrapper that reads config, audits files, writes JSON, and enforces strict mode | `driveone.data.refav_contract` |
| [`scripts/prepare_refav_tracker.py`](../../scripts/prepare_refav_tracker.py) | Builds a reproducible derived candidate manifest from public tracker output and AV2 assets | `driveone.data.refav_tracker` |
| [`src/driveone/data/refav_tracker.py`](../../src/driveone/data/refav_tracker.py) | Performs coordinate conversion, matching, filtering, timestamp selection, and camera projection for the derived pilot | `numpy`, `pandas`, `scipy`, `av2` |
| [`scripts/smoke_test_pe.py`](../../scripts/smoke_test_pe.py) | Optional official PE-Core interface check | `torch`, Pillow, official `perception_models` package |
| [`scripts/run_refav_controls.py`](../../scripts/run_refav_controls.py) | Runs the deterministic RefAV control suite and writes a JSON result | `driveone.data.refav_contract`, `driveone.eval.refav_metrics` |
| [`configs/refav_pilot.yaml`](../../configs/refav_pilot.yaml) | Protocol, provenance, candidate restrictions, split rules, and strict requirements | Verifier |
| [`tests/test_refav_contract.py`](../../tests/test_refav_contract.py) | Unit tests for normalization and contract logic | Data helpers |
| [`tests/test_refav_tracker.py`](../../tests/test_refav_tracker.py) | Unit tests for one-to-one candidate/annotation matching | Optional tracker dependencies |
| [`tests/test_verify_script.py`](../../tests/test_verify_script.py) | CLI test for an empty data root | Verifier |
| [`docs/initial-experiment.md`](../../docs/initial-experiment.md) | Human-readable experiment order and controls | Config and contract |
| [`docs/obsidian/Development Roadmap.md`](Development%20Roadmap.md) | Plain-language reading order and staged research plan | Repository state |
| [`docs/refav_data_contract.md`](../../docs/refav_data_contract.md) | Formal contract and recorded real-data audit result | Official RefAV/AV2 facts |
| [`README.md`](../../README.md) | Short entry point for contributors | All of the above |
| [`CONTRIBUTING.md`](../../CONTRIBUTING.md) | Rules that prevent invalid experiments and untracked artifacts | Project policy |
| [`.gitignore`](../../.gitignore) | Keeps data, weights, reports, manifests, and build artifacts out of Git | Git |

## Empty directories

`configs/`, `experiments/`, `scripts/`, `src/driveone/`, and `tests/` contain `.gitkeep` files from the initial repository layout. They reserve places for future work; they are not executable components.

## Runtime versus documentation

The data runtime path is:

`public Valeo4Cast pickle + AV2 assets` → `scripts/prepare_refav_tracker.py` → `src/driveone/data/refav_tracker.py` → `derived Feather manifest` → `scripts/verify_refav.py` → `src/driveone/data/refav_contract.py` → JSON audit/manifest.

The PE smoke test is a separate path because it depends on a heavy external
model environment. It verifies the released visual checkpoint and records the
external tokenizer interface; it does not implement the DriveOne question
encoder. The Markdown docs explain the intended scientific protocol but do not
run automatically.
