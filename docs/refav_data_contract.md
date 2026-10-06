# RefAV data contract

## Scope

This repository initially evaluates a **custom RefAV referred-track ranking diagnostic**. It is not a yielding-action task, a policy, a planner, or a safety benchmark.

RefAV's official scenario-mining protocol is multi-object and multi-timestamp. Prompts describe scenarios; the official relevance labels are:

- `0`: `REFERRED_OBJECT`
- `1`: `RELATED_OBJECT`
- `2`: `OTHER_OBJECT`

A prompt can have multiple referred objects. Therefore, `Recall@1` is only a secondary diagnostic after defining how multiple positives are handled. The primary comparison should use per-frame/per-track average precision and, where the official evaluator can be reproduced, HOTA and balanced-accuracy metrics.

## Pinned provenance

Before any training run, record all of the following in the generated manifest:

- RefAV repository commit;
- exact scenario-mining dataset artifact/version;
- Argoverse 2 API and split-manifest version;
- source URLs and licenses/terms;
- SHA-256 for every local annotation, track, and manifest file.

The official repository provides scenario-mining files separately from the approximately 1 TB sensor download. The pilot must use the smallest official artifact that still supplies candidate tracks and camera/frame association.

## Normalized candidate record

The verifier normalizes each tracker candidate to:

```text
log_id
prompt
timestamp_ns
track_id
score
label                  # audit-only relevance label
name                   # audit-only relevance name
translation_m
size
yaw
camera/frame association when available
raw tracker category when explicitly available
distance to ego in meters and `is_drivable` after AV2 map projection
match status and matched RefAV track UUID
projection status and projected image box
```

The official spatio-temporal pickle format is a mapping from `(log_id, prompt)` to a list of frame dictionaries. Each frame contains N-length arrays for object fields. The verifier rejects unequal array lengths rather than silently truncating them.

The public Valeo4Cast artifact uses `{log_id: [frame, ...]}` and stores
predictions in the city frame because the published tracker was run with
`--ego_coord`. The adapter first applies the inverse `city_SE3_egovehicle`
pose to every candidate, then matches same-category candidates to annotation
cuboids by one-to-one Hungarian assignment with a 2 m ego-frame XY threshold.
Matching raw city coordinates to RefAV ego-frame annotations produces zero
matches and is therefore invalid.

The candidate label is transferred only when the matched ground-truth UUID is
present in the prompt-specific RefAV annotation rows. Candidates that match a
ground-truth object outside that prompt are `MATCHED_UNANNOTATED_GT`; candidates
with no match are `UNMATCHED_TRACK`. Both retain a null relevance label and are
reported separately. They are never relabeled as `OTHER_OBJECT`.

The pilot applies the AV2 maximum-range and RefAV ROI masks to tracker cuboids
before writing candidates. It chooses the ring camera with the largest visible
projected cuboid and the nearest image within 100 ms. Every retained candidate
has an image association; projected-box status is reported separately.

## Fixed pilot candidate pool

For each `(log_id, prompt, timestamp_ns)` group:

1. Apply the official 50 m and drivable-area filtering after AV2 pose/map projection; the verifier fails strict mode until `distance_m` and `is_drivable` are present.
2. Keep all eligible tracker-produced candidates at that timestamp.
3. Require at least one referred candidate and at least one negative for the primary ranking diagnostic.
4. Preserve the original candidate file and hash it.
5. Randomize candidate IDs before model input.
6. Exclude `label`, `name`, `score`, `timestamp_ns`, `future_timestamp`, `is_positive`, and `track_id` from learned candidate features.
7. Record oracle candidate coverage separately. A scorer cannot recover a referred object absent from the candidate pool.

`RELATED_OBJECT` and nearby `OTHER_OBJECT` candidates are hard negatives. They are not interchangeable with a yielding label.

## Split rules

Splits are by `log_id`, never by frame. Query-template holdout is based on a normalized prompt key and must be checked for near duplicates. No log, track, timestamp, or prompt family may cross train, validation, and test.

## Ambiguity rules

Frames where the official temporal target is unknown or filtered out are retained in the audit report but excluded from the primary ranking metric under a preregistered rule. They must not be silently relabeled as negative.

## Pass gate

The data gate passes only when the verifier confirms:

- complete required fields;
- valid per-frame array lengths;
- finite numeric geometry and scores;
- no duplicate `(log_id, prompt, timestamp_ns, track_id)` entries within a candidate group (the same track may legitimately occur under multiple prompts);
- camera/frame association for the selected visual input;
- at least two log IDs and two prompt-template groups;
- positive and negative candidates in the declared pilot groups;
- candidate features contain no audit labels or target-derived fields;
- oracle coverage and candidate-count distributions are reported.

## PE smoke-test note

The official PE configuration supports `PE-Core-L14-336`, whose vision tower
uses 336 px input, patch size 14, and returns 576 patch tokens of width 1024
after removing the class token (`[1, 576, 1024]` in the pinned checkpoint).
The pooled output is `[1, 1024]`. The smoke script loads the PE vision tower
only, so it does not test the official CLIP text path. The official repository
also exposes a CLIP configuration for this model with text context length 32.
The script's standalone `SimpleTokenizer` defaults to 77 and is not evidence
about the CLIP text interface. The fusion design must define and benchmark its
question encoder explicitly.

## Executed feasibility audit (2026-10-03)

The official RefAV repository was checked out at commit
`5c5be6439ce59b61a31d56431a79a8a04bba33fa`. The official validation
scenario-mining Feather artifact was downloaded from the Argoverse scenario-
mining S3 prefix and hashed as
`e461e51057fdf347a11bdd60609e7de0b0bd8d0eb199314b3fd73c50106d48c9`.
The official log index has 700 train, 150 validation, and 150 test logs with
zero pairwise log overlap; its local manifest hash is
`94150fb8c2a8d0dafc914e78f9f06f9b7e5887a3b5beb173bc7aa42d1092dc47`.

The observed annotation schema is `log_id`, `prompt`, `track_uuid`,
`mining_category`, `timestamp_ns`, `generation_type`, `category`, dimensions,
quaternion pose, ego-frame translation, and `num_interior_pts`. It does **not**
contain tracker confidence, a camera name, an image path, or a 2-D box. This
artifact is therefore ground-truth scenario annotation, not the fixed
tracker-produced candidate file required by the pilot protocol. On a two-log
sample the audit found referred/related/other labels and positive/negative
timestamp groups, but `score` and scalar `yaw` were absent, camera association
was 0%, and the official 50 m/drivable-area filter could not be recomputed
from the table alone. The gate is consequently **not passed**.

One official camera frame, intrinsics, camera extrinsics, and ego poses were
downloaded for one validation log. The Argoverse 2 camera model projected 51
of 210 annotated object centers into the selected front-center image frustum;
this demonstrates that projection is technically possible, but it does not
establish timestamp-aligned cuboid crops or a tracker candidate protocol.

The PE smoke script was run in the base environment and correctly stopped:
the official PE package requires Python >=3.11 and PyTorch/weights were not
installed. No PE tensor shape is treated as verified until that isolated
environment test completes.

## Public tracker pilot (2026-10-04)

The public Valeo4Cast repository was pinned to commit
`dce207347c140ed5cdaeac6c217eb00ee0634e71`. The published
`data/track/for_val_and_test/val_tracking.pkl` artifact was extracted from the
public `data.zip` archive and hashed as
`8cb35baf5c63ff673d9fe00a4b734d23e6b583e2150e51c4ca8d0d9351636b15`.

Using two RefAV validation logs and their public AV2 pose, ROI, calibration, and
camera assets, the repaired adapter kept all aligned prompt timestamps and one
shared `ring_front_center` image per group. It produced 446,810 candidates in
1,660 groups across two logs. Every candidate has a shared camera-image
association; 138,770 have a visible projected box and 308,040 are explicitly
`OUT_OF_VIEW`. There are 165 groups with both a referred object and a labeled
negative. The repaired Feather manifest hash is
`18658d04a723d4ab16799da8cf50ce7de1eabd3b7ff4dadb696cf74a287841ce`.

Only 14,670 candidates received prompt-specific labels (165 referred, 161
related, 14,344 other); 432,140 remain unmatched tracker candidates. This
limitation must be reported in every ranking result. The two logs are enough to
test the pipeline, but not enough for a generalization claim.

The pretrained PE CLIP smoke test completed on host GPU 2 with report hash
`362fa0ccb0c9d6bf30f75d5dc58eb872561b0e9803cc11fca5cfed6c2fc58472`.
`PE-Core-L14-336` returned `[1, 577, 1024]` including the class token,
`[1, 576, 1024]` patch tokens, `[1, 1024]` pooled image features, and
`[1, 1024]` text features with official context length 32. Patch, pooled-image,
and text latency were measured separately; the image input was synthetic.
