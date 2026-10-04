# Code Walkthrough: `refav_tracker.py`

## Why this file exists

The official RefAV annotation table tells us which object a prompt refers to. It is not, by itself, a tracker candidate pool with camera images. This module joins three kinds of input:

1. public tracker predictions;
2. RefAV prompt annotations;
3. Argoverse 2 poses, calibration, map, and camera files.

It writes rows that can be audited and later used by the ranking controls.

## Main path

`prepare_refav_tracker.py` calls `prepare_records()`. That function loads the annotations and tracker frames, selects one usable timestamp per prompt, converts tracker positions to the ego-vehicle frame, filters the official region and range, matches tracker objects to annotated objects, projects each object into a camera, and returns rows.

The command then writes a Feather manifest and a JSON summary. The large source files and the derived manifest stay outside Git.

## Important functions

### `load_tracker_pickle`

Reads the pinned Valeo4Cast pickle and groups frames by `log_id`. Pickle is trusted local input; the verifier does not download or execute a remote pickle.

### `global_to_ego` and `_ego_yaw`

Tracker positions are in the city frame. The camera and prompt geometry use the vehicle frame. These functions use the ego pose to transform positions and heading into that frame.

### `match_candidates`

For each object category, it performs one-to-one Hungarian matching using XY distance and keeps matches within 2 m. This is a deterministic bridge between a tracker row and a prompt annotation. It is not a claim that the official HOTA evaluator has been reproduced.

### `select_decision_timestamps`

Chooses the earliest frame for each prompt that has both a matched referred object and a matched negative after the range/ROI filter. This makes the pilot usable, but it also means the current selection uses labels. That target-informed selection must be removed or fixed before a final unbiased benchmark.

### `project_candidate`

Builds 3-D box corners, projects them into each ring camera, and keeps the camera with the largest visible box. It matches the nearest image within 100 ms. The current code does not perform motion compensation or an occlusion check, and “projected” does not guarantee that the object is visibly present in pixels.

### `prepare_records`

Creates one row per tracker candidate. A row can have `label=None` with `match_status=UNMATCHED_TRACK`; this is different from a negative label. The current pilot has many such rows, so controls must report labeled and unmatched strata separately.

## What this file proves

It gives us a repeatable way to build a small candidate manifest from pinned files, with geometry and camera associations. It does not prove that every candidate is visible, that the labels are complete, or that the selection procedure is free of label leakage.

## Read it with

Read [[RefAV Data Contract]] for the row meaning, then [[Code Walkthrough - refav_contract]] for the checks applied after this module writes the rows.
