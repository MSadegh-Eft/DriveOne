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
```

The official spatio-temporal pickle format is a mapping from `(log_id, prompt)` to a list of frame dictionaries. Each frame contains N-length arrays for object fields. The verifier rejects unequal array lengths rather than silently truncating them.

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

The official PE configuration supports `PE-Core-L14-336`, whose vision tower uses 336 px input, patch size 14, 1024 hidden width, and 1024 output dimension; the text context is 32 tokens. The official downstream vision API exposes `forward_features`, but the repository's example of dense token output uses PE-Lang/Spatial. The pilot must verify the exact PE-Core tensor shape and whether the question path uses pooled text or token-level text before defining the fusion interface.

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
