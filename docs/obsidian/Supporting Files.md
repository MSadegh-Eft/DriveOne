# Supporting Files Walkthrough

## `pyproject.toml`

The build-system section tells modern Python packaging tools to use setuptools. The project section names DriveOne, sets version `0.1.0`, describes the repository, and declares Python `>=3.10`. Runtime dependencies are empty because the contract library is standard-library based for JSON/CSV/pickle inputs.

The `dev` extra installs pytest for future test suites. The `pe` extra lists PyTorch and Pillow, but the official `perception_models` repository has additional requirements and currently requires Python >=3.11. Installing the `pe` extra alone does not complete the official PE environment.

The setuptools package-discovery setting tells the build to find packages under `src/`.

## `src/driveone/__init__.py`

This is the package marker and contains only the package docstring. It intentionally exposes no model API yet because no scorer has been implemented.

## `src/driveone/data/__init__.py`

This file re-exports the public contract functions and constants. Code can import `driveone.data.load_records` rather than reaching into a private module path. The `__all__` list documents the supported public surface.

## `README.md`

The README is the short project entry point. It states the narrow first milestone, gives the verifier and PE smoke-test commands, lists the initial experiment order, describes the repository layout, and warns that data/weights/outputs stay outside Git.

The README is intentionally shorter than this vault. It should answer “what is this and how do I start?”; the vault answers “how does every part work?”

## `CONTRIBUTING.md`

The contribution rules are scientific guardrails rather than style rules. They require pinned provenance, multi-positive RefAV semantics, forbidden-feature removal, measurement protocols for strong claims, and external storage for large artifacts.

## `docs/initial-experiment.md`

This is the human protocol for the first experiment. It defines gate order, candidate semantics, required controls, official versus custom metrics, split rules, and the decision to defer Qwen/distillation until the frozen student and controls are reproducible.

## `docs/refav_data_contract.md`

This is the formal contract and the record of the actual 2026-10-03 audit. Unlike a README promise, it records the observed Feather schema, artifact hashes, projection sanity check, PE environment blocker, and the conclusion that the data gate is not passed.

## `.gitignore`

The first section ignores Python/editor artifacts. The local-artifact section excludes data, checkpoints, weights, caches, outputs, runs, and W&B files. The report section excludes due-diligence PDFs. The manifest section excludes generated JSON audits. Build metadata is also ignored.

This is why the repository can contain code and protocol while local RefAV files and model weights remain on disk but absent from commits.

## `.gitkeep` files

Git does not track empty directories, so `.gitkeep` reserves future locations. It has no runtime behavior and can remain until real files replace it.

## Tests package marker

`tests/__init__.py` marks the test directory as a Python package. It has no test logic.
