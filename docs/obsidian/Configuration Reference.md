# Configuration Reference

Source: [`configs/refav_pilot.yaml`](../../configs/refav_pilot.yaml)

## Top-level identity

`protocol_version`, `name`, and `purpose` label this as a custom fixed-frame referred-track diagnostic. The purpose string prevents the pilot from being mistaken for the official RefAV leaderboard task or a driving-policy benchmark.

## `source`

This section records URLs, repository revision, dataset artifact hash, AV2 API version, tracker-artifact status, split-manifest hash, and license notes. The current `tracker_artifact` value explicitly says that the downloaded scenario-mining export does not contain the required tracker artifact. That is a recorded blocker, not a successful pin.

## `data`

- `root` is the ignored local data directory.
- `records_file` can override automatic discovery but is currently `null`.
- `split_manifest_file` points to the local official log manifest relative to `data/refav`.
- `expected_protocol` documents the intended RefAV spatio-temporal record family.

## `candidate_protocol`

This is the experimental contract: all aligned prompt timestamps, all eligible tracker tracks, one shared front-center image per group, label meanings, hard negatives, randomized IDs, allowed/excluded fields, official filtering, and the multi-positive target. The `top1_is_secondary_diagnostic` flag prevents a single winner metric from becoming the headline by accident.

## `split_protocol`

This declares log-level splitting and the conservative prompt normalization used for diagnostics. It does not claim that string normalization solves semantic template leakage.

## `metrics`

The official metric names are kept separate from custom fixed-frame diagnostics. This matters because a custom Recall@1 result is not automatically comparable to RefAV’s temporal/spatio-temporal challenge metrics.

## `verification`

These booleans control strict mode. In particular, strict mode requires camera association, a reproducible candidate pool, complete fields, no malformed arrays, a pinned provenance record, and an official split manifest. The strict verifier is intentionally expected to fail on the current annotation-only export.
