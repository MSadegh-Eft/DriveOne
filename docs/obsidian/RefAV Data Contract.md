# RefAV Data Contract

## What one example means

The intended pilot unit is a fixed-frame group:

```text
(log_id, prompt, timestamp_ns) -> [candidate_1, candidate_2, ...]
```

Each candidate is a tracker-produced object track at that timestamp. The positive target is `REFERRED_OBJECT`; hard negatives are `RELATED_OBJECT` and nearby `OTHER_OBJECT` tracks. There may be multiple positives, so the task is multi-positive ranking rather than ordinary single-class classification.

## Canonical fields

The library expects:

| Field | Meaning | Model-input status |
| --- | --- | --- |
| `log_id` | Argoverse sequence/log identifier | Split key; not a learned feature by default |
| `prompt` | Natural-language scenario description | Question input |
| `timestamp_ns` | Decision timestamp | Audit-only; excluded from candidate features |
| `track_id` | Candidate identity | Audit-only; IDs are randomized before model input |
| `score` | Tracker confidence | Excluded from the main model; allowed only in an explicit shortcut control |
| `label` | RefAV relevance class 0/1/2 | Target/audit-only |
| `translation_m` | 3D position | Candidate geometry after projection or crop construction |
| `size` | Object dimensions | Candidate geometry/crop construction |
| `yaw` | Object heading | Candidate geometry/crop construction |
| camera fields | Image/camera association | Required for the visual pilot |

The library accepts aliases so official exports and tracker files can be normalized into one vocabulary. It does not silently invent missing score, camera, or geometry fields.

## Official annotation versus candidate artifact

The official validation Feather that was inspected has `track_uuid`, `mining_category`, quaternion pose, ego-frame position, object dimensions, and timestamps. It does not have tracker confidence, camera paths, 2D boxes, or a tracker candidate file. This is why the current audit finds relevance groups but still fails the strict pilot gate.

## Leakage rules

The candidate feature allow-list in the config is deliberately small: rendered candidate crop, projected box, and a pre-pinned raw tracker category if explicitly justified. The following are excluded from learned features:

- relevance label and label name;
- `is_positive`;
- tracker confidence;
- timestamps and future timestamps;
- track IDs;
- any metadata derived from the answer.

The verifier checks only the declared allow-list. A future dataset-construction program must enforce the feature extraction itself.

## Filters and coverage

The official protocol applies 50-meter and drivable-area constraints. The verifier considers the filter reproducible only when each record has finite distance and an explicit `is_drivable` value. It also reports candidate coverage: if the referred object is absent from the candidate pool, a scorer cannot recover it.

## Splits

Splits are by `log_id`, never by frame or candidate. The split manifest used in the local audit has 700 train, 150 validation, and 150 test logs with no overlap. Normalized prompt strings are only a rough holdout diagnostic; they are not proof of semantic template disjointness without external prompt-family IDs or a preregistered clustering method.

## Multi-positive metrics

Recall@1 is a secondary diagnostic because one timestamp can contain multiple referred tracks. The protocol calls for per-frame and per-track average precision, plus official RefAV metrics when the official evaluator can be reproduced. Calibration metrics such as NLL, Brier, and ECE require an explicit probability construction for a variable-size candidate set.
