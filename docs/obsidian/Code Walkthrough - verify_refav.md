# Code Walkthrough: `verify_refav.py`

Source: [`scripts/verify_refav.py`](../../scripts/verify_refav.py)

This script is the command-line orchestration layer around [[Code Walkthrough - refav_contract]].

## Startup and imports

The script computes the repository root from its own location and inserts `src/` into `sys.path`. That makes this command runnable directly from a checkout without requiring `pip install -e .` first.

It imports the contract functions and defines supported input suffixes. The supported list includes Feather because the official RefAV annotation release uses that format.

## `parse_simple_config`

First it attempts JSON parsing; JSON is valid YAML for the subset used here. If that fails, it parses a small indentation-aware subset:

- top-level `section:` creates a dictionary;
- `key: value` entries are assigned to the current section;
- quoted strings, integers, floats, booleans, null, and simple bracket lists are supported;
- comments after `#` are stripped.

This avoids a PyYAML dependency for a small verifier, but it is not a complete YAML implementation. The pilot config intentionally stays within the supported subset.

## `discover_files`

An explicit path is resolved first relative to the repository, then the data root, then the current working directory. Without an explicit path, the configured data root is recursively scanned for supported file extensions. This function does not download anything; network acquisition is deliberately outside the verifier.

## `run`

`run` builds the complete result dictionary and returns a shell exit code.

### Configuration and provenance

It loads the source and data sections, resolves the data root, and records the declared candidate/split protocols. Placeholder values such as `PIN_BEFORE_RUN` become warnings. Values beginning with `UNAVAILABLE_` are recorded as an explicit blocker, which is how the current missing tracker artifact is represented.

### Feature and file checks

The declared allow-list is checked with `validate_candidate_feature_schema`. If no records are found, the result is `not_ready` and the script gives instructions without network side effects.

For each discovered file, `load_records` and `inspect_records` run independently. One malformed file is recorded as a file-level error while the remaining files can still be audited. If files load successfully, the script also creates a combined inspection and a hashed manifest.

### Split manifest

If configured, the split JSON is loaded and checked for log overlap. A missing or unreadable manifest is warned about; strict mode can turn that into failure when required by config.

### Strict mode

The command returns `2` if any required condition fails: missing records, forbidden declared features, missing required fields, duplicates, timestamp errors, invalid labels, nonfinite values, file errors, missing/unavailable provenance, split errors, no positive/negative group, absent camera association, or unreproducible official filters. Otherwise it returns `0`.

Strict mode is a gate, not a quality score. Passing means the input is structurally eligible for the next experiment; it does not mean the future model will be good.

## `main`

Defines `--config`, `--data-root`, `--records`, `--output`, and `--strict`, then delegates to `run`. The command prints JSON even when the gate fails, so the failure is inspectable rather than just a shell status.

## Typical invocations

```bash
python scripts/verify_refav.py
python scripts/verify_refav.py --records data/refav/pilot/tracks.pkl
python scripts/verify_refav.py --records data/refav/pilot/tracks.pkl --output manifests/audit.json --strict
```

The current local annotation sample produces useful label/group counts but fails strict mode because it is not a tracker candidate artifact and lacks score, camera association, and reproducible filter fields.
