# RefAV candidate-pool decision

## What this audit tested

This audit asked whether RefAV can provide a fair candidate pool for a
language-conditioned external-object track ranker. It used the nine logs in
the pinned repeated-prompt plan and the native Le3DE2E tracker timestamps. No
model was trained.

The official RefAV source was pinned to commit
`5c5be6439ce59b61a31d56431a79a8a04bba33fa`. Its tracker conversion was
replayed without editing upstream code. The replay matched our independent
city-to-ego conversion with a maximum translation error of 0 m. The official
conversion's whole-log score filter and Le3DE2E height adjustment were
recorded separately because the score filter uses information from the whole
log and is not a causal online candidate generator.

The official tutorial uses Le3DE2E validation predictions:
<https://raw.githubusercontent.com/CainanD/RefAV/main/run/tutorial.ipynb>.
The source repository is <https://github.com/cainand/refav>.

## Camera inputs

The original local copy had only the front-center camera for eight logs. That
would have made a camera-coverage conclusion invalid. We downloaded only the
nearest camera frame for each of the 32 native tracker timestamps, for all
seven ring cameras and all nine logs:

- 1,536 JPEG files;
- approximately 0.55 GB;
- maximum camera timestamp offset: 32.7 ms;
- manifest: `/ehsan/m.sadegh/driveone_assets/refav/camera_download_manifest_20261008.json`.

The full audit now has all seven camera streams at every native timestamp used
in the comparison. Geometric projection means that a 3-D box intersects an
image; it does not prove that the object is unoccluded or visually clear.

## Candidate-source results

The machine-readable report is outside Git:

`/ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v3/candidate_source_audit.json`

Report SHA-256: `0ad1d654d85c0e66cd82311cbcab7ebc75256db31e1e64d79b76e9273ecd69e0`.

The audit contains 2,880 prompt/timestamp groups. Ego-vehicle referred
prompts are reported separately because an ego box is not an external camera
candidate. External-object results are:

| Source | External referred availability | Positive projection | Unknown candidate fraction |
| --- | ---: | ---: | ---: |
| Official Le3DE2E replay | 68.6% | 100.0% | 80.0% |
| Causal Le3DE2E pool | 66.9% | 100.0% | 92.3% |
| Historical Valeo pool | 66.8% | 100.0% | 92.3% |
| Ground-truth oracle pool | 100.0% | 100.0% | 0.0% |

Availability is conditional on a ground-truth external referred object being
eligible at that timestamp. The earlier 20.8% figure used all prompt
timestamps, including timestamps where the event was absent; that was the
wrong denominator for tracker recall.

Conservative unique-assignment recall for the official replay is:

| Match threshold | Recall on eligible external referred groups |
| ---: | ---: |
| 1 m | 71.7% |
| 2 m | 68.6% |
| 4 m | 59.2% |

The 1 m to 4 m change is 20.8 percentage points, above the 10-point
stability limit. The optimistic numbers that include ambiguous assignments
are 96.4%, 97.0%, and 97.3%. Those cannot be the primary result because a
single target may have multiple nearby tracker candidates and the label is not
unique.

The official replay controls use the same candidates for every control:

| Control | mAP | Recall@1 |
| --- | ---: | ---: |
| Tracker confidence | 0.533 | 0.469 |
| Candidate distance | 0.446 | 0.469 |
| Projected box area | 0.144 | 0.124 |
| Category frequency | 0.152 | 0.088 |
| Random | 0.061 | 0.019 |

These are diagnostics on the repaired native-timestamp pool. They are not
comparable to the previous label-selected 500-group baseline and do not prove
a model result.

## Label and ranking policy

Candidate membership is independent of prompts, mining labels, matching
success, and future timestamps. The candidate-pool hash is unchanged if
prompts and target labels are permuted.

Unmatched tracker rows remain marked `UNMATCHED_TRACK`. For ranking diagnostics
only, they are treated as explicit non-referred tracker false positives. They
are not relabeled as `OTHER_OBJECT`. This reflects the meaning of a prediction
that does not match any ground-truth object while preserving the original
status for auditing.

The official conversion also injects an `EGO_VEHICLE` row. Ego-vehicle prompts
are kept as a separate scenario-level category and are not counted as
camera-renderable external-object candidates.

## Decision

```text
REFAV_ORACLE_ONLY
```

The ground-truth oracle proves that the annotations, poses, calibration, and
camera data can support an external-object ranking experiment. The available
Le3DE2E tracker does not provide enough uniquely matched referred candidates,
and its confidence and distance metadata are strong ranking shortcuts.

Therefore:

- Do not train DriveOne on the official Le3DE2E pool.
- Do not present the oracle pool as deployment-like candidate generation.
- Keep RefAV as a documented oracle-candidate or negative finding.
- A new independent detector/tracker source is required before the RefAV
  branch can support the central candidate-ranking claim.
- If no such source is found, stop this RefAV branch and pivot to a task with
  an independently generated candidate pool.

The audit does not justify patch tokens, temporal input, Qwen, distillation,
trajectory scoring, or deployment optimization.

## Reproduction

```bash
TMPDIR=/data/sadegh/tmp conda run --no-capture-output -n refav \
  python scripts/audit_refav_candidate_sources.py \
  --output-dir /ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v3
```

Large candidate and control artifacts stay outside Git. The repository keeps
the audit code, tests, configuration, and this decision record.
