# Code Walkthrough: `refav_contract.py`

Source: [`src/driveone/data/refav_contract.py`](../../src/driveone/data/refav_contract.py)

This module is the reusable, dependency-light core. It knows about record formats and audit rules, but it does not know how to train a model.

## Imports and constants

The standard-library imports cover CSV/JSON/pickle I/O, hashing, regular expressions, counters, grouped records, paths, and typing. `pandas` is imported only inside the Feather branch so JSON/pickle checks do not require the large data stack.

`REQUIRED_FIELDS` is the canonical schema. `LABEL_NAMES` maps numeric RefAV labels to their official names. `LABEL_ALIASES` lets string exports such as `REFERRED_OBJECT` become numeric labels. `ALIASES` maps common source names such as `track_uuid`, `mining_category`, and `confidence` into canonical names. `CAMERA_FIELDS` lists the names that count as an explicit camera/frame association.

These constants are policy. Changing them changes what the verifier considers a valid experiment.

## `sha256_file`

Reads a file in 1 MiB chunks and returns a SHA-256 digest. Chunking avoids reading a large annotation or tracker file into memory just to hash it. Hashes make a future run refer to exact bytes rather than a vague dataset name.

## `_coerce_label` and `normalize_record`

`_coerce_label` converts known strings and numeric strings into integer labels. Unknown values are preserved so the inspector can report them as invalid rather than hiding them.

`normalize_record` copies the input mapping, fills canonical aliases only when the canonical key is absent, converts timestamps to integers when possible, and normalizes labels. It preserves unknown fields. That last behavior is important: an audit should not destroy evidence just because a field is not part of the canonical schema.

## `_flatten`

Handles ordinary JSON structures. Lists are recursively flattened. Mappings that already look like track records are emitted as rows. Otherwise child mappings are traversed while retaining context. A string key containing `log_id|prompt` is split into context because JSON cannot preserve Python tuple keys.

The function deliberately does not pretend that every arbitrary string representation of a tuple is safely parseable. The official tuple-key case is handled by the pickle-specific adapter.

## `_as_list` and `_flatten_refav_pickle`

The official RefAV spatio-temporal pickle is a mapping from `(log_id, prompt)` to frame dictionaries. Each frame stores N-length arrays for `track_id`, `score`, `label`, `name`, `translation_m`, `size`, and `yaw`.

`_flatten_refav_pickle`:

1. Extracts the two-part mapping key.
2. Ensures a frame is treated as a sequence even if only one frame is supplied.
3. Converts array-like values with `.tolist()`.
4. Checks every vector field has exactly the same length as `track_id`.
5. Emits one flat row per candidate in the frame.
6. Preserves optional `is_positive` and `raw_category` fields for audit purposes.

The array-length check is a central correctness guard. Truncating mismatched arrays would silently associate one candidate’s geometry with another candidate’s label.

## `load_records`

Chooses an adapter based on the filename suffix:

- JSONL: one JSON object per line;
- JSON: recursive generic flattening;
- CSV: `csv.DictReader` rows;
- Feather: pandas reads the table, then official scalar position/dimension columns are assembled into `translation_m` and `size` while original fields remain;
- pickle: trusted local pickle loading followed by the official RefAV flattening.

Pickle is intentionally local-only. Python pickle can execute code during deserialization, so the verifier does not fetch or execute remote pickle files.

The current Feather path loads the whole table into memory. That is acceptable for a small pilot sample but is a known scalability limitation for the full multi-gigabyte artifact.

## `template_key`

Lowercases and whitespace-normalizes prompts, replaces UUID-like strings and numbers with placeholders, and returns a conservative lexical key. This can find obvious duplicates but cannot establish semantic template independence. The inspection report explicitly says external family IDs are still required.

## `_group_key`

Defines the fixed-frame decision unit as `(log_id, prompt, timestamp_ns)`. Candidate ranking must happen within these groups, never across unrelated logs or prompts.

## `inspect_records`

This is the main audit routine. It first materializes the iterable because the current reports need repeated access to rows. It then:

### Schema and provenance counters

Counts missing required fields, logs, prompts, normalized templates, labels, camera-associated rows, optional temporal targets, and official-filter geometry availability.

### Leakage report

Counts whether rows contain `label`, `name`, `score`, or `timestamp_ns`. These fields may be present for evaluation and controls, but their presence is a warning if they are accidentally passed to the model.

### Duplicate detection

Uses `(log_id, prompt, timestamp_ns, track_id)` as the duplicate key. The prompt is included because the official table legitimately repeats the same track under different prompts. Duplicate IDs within one candidate group remain invalid.

### Numeric and temporal checks

Reports nonfinite score/timestamp/yaw values and detects non-monotonic timestamp order within each log/prompt sequence.

### Candidate-group checks

For every fixed-frame group, counts referred candidates and negative candidates. It reports whether at least one group has both. It does not collapse multiple positives into one label.

### Returned report

The returned dictionary is JSON-friendly and includes counts, examples, feasibility booleans, official label names, and notes. `official_filter_reproducible` is false unless every inspected row supplied usable `distance_m` and boolean `is_drivable` values.

## `validate_candidate_feature_schema`

Takes the declared feature allow-list and returns forbidden fields such as labels, names, IDs, timestamps, and `is_positive`. It is a declaration check, not a neural-network firewall. A future dataset adapter must construct only approved tensors.

## `validate_log_disjoint`

Converts each split to a set and reports pairwise overlaps. It does not require exactly train/val/test names, which makes it reusable for pilot subsets.

## `build_manifest`

Hashes every existing input file and stores absolute path, byte count, digest, configuration, and inspection output. Absolute paths are useful locally but should be treated as machine-specific metadata when sharing a manifest.

## What this module does not do

It does not project cuboids into images, construct tracker outputs, calculate AP, load PE, or train DriveOne. Those omissions are intentional: this module establishes whether those later steps would have a valid input contract.
