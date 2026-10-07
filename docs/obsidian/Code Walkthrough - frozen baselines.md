# Code Walkthrough: frozen baseline smoke test

This note explains the first small model run in simple terms. The goal was to
check whether a frozen PE image/text representation can beat easy shortcuts
before we write a patch-token fusion model.

## 1. `scripts/export_refav_subset.py`

This script makes the exact small dataset used by the smoke test.

- `sha256_file` records the source-file hash, so a later run can prove it used
  the same manifest.
- `select_rankable_group_keys` groups rows by `(log_id, prompt,
  timestamp_ns)`. A group is kept only when it has at least one referred row
  (`label == 0`) and one labeled negative (`label == 1` or `2`). It chooses a
  seeded, deterministic sample of groups.
- The script then copies **all rows** in each selected group. This matters:
  unknown tracker candidates remain in the ranking pool and are not silently
  treated as negatives.
- It writes Feather for table work, JSONL for the model scripts, and a JSON
  manifest containing split log IDs, group keys, row counts, and output hashes.

## 2. `scripts/extract_pe_features.py`

This script calls the official PE implementation once and saves reusable
features outside Git.

- It reads image paths and prompts from the three JSONL files.
- It loads `PE-Core-L14-336` with the official configuration and preprocessing.
- `encode_image` produces one pooled image vector per frame. `encode_text`
  produces one text vector per prompt. The model is frozen; no training occurs
  here.
- Features are saved in a PyTorch file. The report records device, input
  hashes, tensor shapes, image size, context length, and image failures.
- The successful image key list is kept aligned with the saved feature rows.
  This prevents a failed image decode from shifting every later feature.

This script does not test the DriveOne fusion model. It only makes a fair,
reusable pooled representation for the first baseline.

## 3. `scripts/train_refav_baselines.py`

This script trains four small controls using the same candidate rows.

### Input preparation

`read_jsonl` loads records and `group_rows` rebuilds the ranking groups. The
model sees candidates in each group, not isolated rows. `box_features` creates
12 simple candidate features: normalized projected-box coordinates, visibility,
distance, object size, and ego-frame translation. `metadata_features` creates a
three-value shortcut vector from tracker score, tracker label, and distance.

`build_inputs` attaches each candidate to its category, prompt ID, pooled PE
image vector, and pooled PE text vector. Image and text features are normalized
before they enter the learned model.

### The four controls

- `candidate_only`: candidate geometry and category, no question or image.
- `metadata_only`: tracker metadata and category. This is an explicit shortcut
  control, not a permitted DriveOne input.
- `task_id`: candidate geometry, the pooled PE image vector, and a learned
  embedding for the prompt ID. This tests task/query identity against the same
  visual input used by pooled PE, without reading the prompt words.
- `pooled_pe`: candidate geometry plus pooled PE image and text features. This
  is the first question-conditioned image baseline, but it still has no patch
  tokens.

All four use the same small MLP-style scorer and the same candidate groups.
The task-ID and pooled-PE context widths are matched; their parameter counts
can still differ because a prompt embedding and a text projection have
different vocabulary sizes.
The only intended change is the context input. Unknown labels are omitted from
the binary cross-entropy training loss, but they remain in the ranking pool.

### Metrics

`average_precision` ranks referred candidates above labeled negatives inside
one group and allows more than one positive. `ranking_metrics` averages AP over
rankable groups and computes Recall@1. It also computes NLL, Brier, and ECE on
labeled candidates. These calibration numbers are diagnostic because the
training loss and candidate pool are not yet a complete abstention protocol.

### Training loop

`train_one` uses AdamW and BCE for a small fixed number of epochs. It skips a
group only when it has no positive or no labeled negative. `score_groups` then
scores every candidate, including unknown rows, so evaluation matches the
candidate-ranking definition.

## 4. What the result means

On the corrected 200-group-per-split smoke subset, pooled PE scored 0.0319 test
mAP and task ID with the same pooled image scored 0.0581. The same test groups
scored 0.1674 with tracker-score ranking and 0.1678 with projected-box-area
ranking. The corrected 500-group-per-split replication then scored 0.0476 for
pooled PE and 0.0576 for task ID, versus 0.1813 and 0.1729 for the two
deterministic controls. Therefore pooled PE has not passed the next gate. The
next step is protocol diagnosis and candidate redesign, not a patch-token
model.

## 5. How the files connect

```text
repaired Feather manifest + split plan
        |
        v
export_refav_subset.py
        |---- train/validation/test JSONL
        |---- subset manifest and hashes
        v
extract_pe_features.py ---- frozen pooled PE image/text cache
        |
        v
train_refav_baselines.py ---- four controls + ranking/calibration report
```

The external data and feature files stay outside Git. Git contains the code,
configuration, documentation, and hashes needed to reproduce or audit them.
