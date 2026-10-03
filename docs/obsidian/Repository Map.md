# Repository Map

| Path | Purpose | Depends on |
| --- | --- | --- |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata and optional development/PE dependencies | Python packaging |
| [`src/driveone/data/refav_contract.py`](../../src/driveone/data/refav_contract.py) | Normalization, loading, schema inspection, leakage checks, split checks, and manifest hashing | Python standard library; pandas/pyarrow only for Feather |
| [`src/driveone/data/__init__.py`](../../src/driveone/data/__init__.py) | Public exports for the data helpers | `refav_contract.py` |
| [`scripts/verify_refav.py`](../../scripts/verify_refav.py) | CLI wrapper that reads config, audits files, writes JSON, and enforces strict mode | `driveone.data.refav_contract` |
| [`scripts/smoke_test_pe.py`](../../scripts/smoke_test_pe.py) | Optional official PE-Core interface check | `torch`, Pillow, official `perception_models` package |
| [`configs/refav_pilot.yaml`](../../configs/refav_pilot.yaml) | Protocol, provenance, candidate restrictions, split rules, and strict requirements | Verifier |
| [`tests/test_refav_contract.py`](../../tests/test_refav_contract.py) | Unit tests for normalization and contract logic | Data helpers |
| [`tests/test_verify_script.py`](../../tests/test_verify_script.py) | CLI test for an empty data root | Verifier |
| [`docs/initial-experiment.md`](../../docs/initial-experiment.md) | Human-readable experiment order and controls | Config and contract |
| [`docs/refav_data_contract.md`](../../docs/refav_data_contract.md) | Formal contract and recorded real-data audit result | Official RefAV/AV2 facts |
| [`README.md`](../../README.md) | Short entry point for contributors | All of the above |
| [`CONTRIBUTING.md`](../../CONTRIBUTING.md) | Rules that prevent invalid experiments and untracked artifacts | Project policy |
| [`.gitignore`](../../.gitignore) | Keeps data, weights, reports, manifests, and build artifacts out of Git | Git |

## Empty directories

`configs/`, `experiments/`, `scripts/`, `src/driveone/`, and `tests/` contain `.gitkeep` files from the initial repository layout. They reserve places for future work; they are not executable components.

## Runtime versus documentation

The runtime path is:

`configs/refav_pilot.yaml` → `scripts/verify_refav.py` → `src/driveone/data/refav_contract.py` → JSON audit/manifest.

The PE smoke test is a separate path because it depends on a heavy external model environment. The Markdown docs explain the intended scientific protocol but do not run automatically.
