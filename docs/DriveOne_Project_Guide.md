# DriveOne Project Guide

> **Last updated:** 8 October 2026. The current RefAV candidate-source audit is complete. The project is paused at a data-validity gate; no new model architecture is being trained.

## How to use this guide

This is the main guide for the whole DriveOne project. Read it from the top the first time. It explains the project in simple English, but it keeps the technical meaning.

The project has two different levels:

1. The **final research idea**: a small model that scores bounded driving candidates using camera images and a natural-language question.
2. The **current feasibility study**: a much smaller experiment that asks whether a question and one camera image can rank the referred object above other tracked objects in RefAV.

The second level comes first. If this small test is invalid or does not beat simple controls, building the large model would waste time and compute.

The companion PDF is generated from this file. The Obsidian vault contains shorter notes and code walkthroughs, but this document is the best single starting point.

## 1. The project in one page

### The final goal

The proposal is called **DriveOne: Fast Language-Aligned Driving Decisions**. Its final goal is a fast multimodal model for autonomous-driving decisions.

The model receives:

- one or more camera images;
- a question written in natural language;
- a finite list of candidates.

It gives one score or probability to each candidate. The highest-scoring candidate is selected when the candidates are mutually exclusive. For multi-label tasks, each candidate receives an independent probability.

Examples of candidate types in the proposal are:

- `yes`, `no`, and `unclear` for a Boolean question;
- tracked people or vehicles for an entity-selection question;
- trajectories for a planning question.

The core mathematical idea is:

```text
p(candidate | image, question, candidate list)
```

This is a **bounded-candidate** model. It does not generate a free-form answer. It chooses from the candidates that someone else supplied.

### What the final goal does not automatically mean

A candidate scorer is not automatically:

- a driving policy;
- a closed-loop planner;
- a safe-driving system;
- a model that can find an object that is missing from the candidate list;
- evidence that a candidate generator is good;
- a real-time system.

These claims require separate measurements. DriveOne can only be useful if the candidate list is valid, independent of the answer, and available at deployment.

### The current research question

Before testing Boolean answers, trajectories, six cameras, four frames, Qwen, or distillation, we use one narrow task:

> Given a RefAV question, one front-center image, and all eligible tracker objects at one timestamp, can a model rank the referred object above the other candidates?

We call this task **RefAV referred-track ranking**. It is a custom diagnostic. It is not the official RefAV leaderboard task, and it is not a yielding or safety task.

### Current answer

The original 500-group baseline was blocked by metadata shortcuts, but the
follow-up **evaluation-correctness gate** has now corrected that diagnosis.
The corrected audit keeps unknown labels explicit, distinguishes global
assignment ambiguity from local nearby matches, excludes synthetic ego rows
from external controls, and confirms that the candidate pool itself did not
change.

The corrected causal Le3DE2E pool provides 99.0% external referred-track
availability at the 2 m threshold. Its 1 m, 2 m, and 4 m availability is
98.3%, 99.0%, and 99.1%, a 1.07-point range. The data gate therefore passes.
This does not mean the model has passed: the pooled-PE and task-ID models still
need to be rerun on the corrected external groups.

The corrected frozen pooled-PE baseline has now run with two seeds. On the test
split, pooled PE reaches 0.0668/0.0569 mAP for seeds 0/1. Task ID reaches
0.0787/0.0718, metadata-only reaches 0.0830/0.0810, and fixed front-center
projected area reaches 0.1964. Pooled PE also has worse seed-0 ECE than task
ID (0.1583 versus 0.0468). The model gate therefore fails:
**`POOLED_BASELINE_GATE_FAILED`**. Patch tokens remain paused.

The source candidate pool still passes its data checks. However, fixed
front-center projection covers only 46.0% of test positive rows, while the
seven-camera audit covers essentially all matched positives. The next decision
is whether to define a fair multi-camera input or stop the RefAV branch; it is
not to add a larger model.

## 2. Why we are moving slowly

The original proposal contains many large experiments:

- pooled PE versus patch tokens;
- PE-Spatial-L14-448;
- four-frame temporal input;
- six-camera input;
- shared scoring for entities and trajectories;
- DriveLM question conversion;
- NAVSIM and GTRS trajectory candidates;
- Waymo trajectory preference data;
- Qwen3-VL-Reranker-2B;
- distillation;
- latency and memory comparisons.

Each item changes several things at once. If a result improves, we would not know whether the improvement came from the representation, candidate construction, label conversion, data, training loss, or extra compute.

The project therefore uses **gates**. A gate is a small test that must pass before a more expensive stage begins.

The order is:

```text
understand task
    -> build candidate data
    -> audit leakage and splits
    -> check PE interface
    -> run simple controls
    -> run pooled baseline
    -> diagnose shortcuts
    -> only then test patch tokens
    -> only then add temporal/spatial/planning work
    -> distillation last
```

This order is a scientific protection. A negative result at an early gate is useful because it prevents an uninterpretable large experiment.

## 3. What the proposal claims

The due-diligence report separated the proposal into six claims. They must be tested separately.

| claim | what it would mean | current status |
| --- | --- | --- |
| Natural language beats task ID | Words help the model generalize to new questions better than a learned task label | unverified |
| Patch tokens beat pooled image features | Keeping spatial visual tokens improves grounded candidate selection | interface verified; performance unverified |
| One shared scorer handles all candidate types | One scoring function works for symbolic answers, entities, and trajectories without harmful negative transfer | unverified |
| Candidate scoring survives distribution shift | Ranking stays useful when candidate count, scenes, cities, datasets, or planners change | unverified |
| Qwen3-VL-Reranker-2B is a fair AV baseline/teacher | Qwen’s relevance score is an appropriate comparison for driving decisions | objective mismatch; must be tested carefully |
| A small scorer is faster with similar quality | End-to-end quality, latency, memory, and throughput are better than larger models | unverified |

The current repository only addresses the first two claims in a small RefAV setting. It does not yet test planning, safety, deployment, Qwen, or distillation.

## 4. Important words in plain English

### Candidate

A candidate is one possible answer the model is allowed to choose. For the current RefAV task, a candidate is one tracker-produced object track at one timestamp.

### Candidate pool

The candidate pool is the complete list of candidates for one question and timestamp. The scorer can rank only this list.

### Referred object

The object that matches the natural-language scenario description. In this project, the RefAV label is `REFERRED_OBJECT` and has numeric label `0`.

### Related and other objects

RefAV also marks objects that are related to the described interaction or other objects in the scene. They have labels `RELATED_OBJECT` (`1`) and `OTHER_OBJECT` (`2`). They are useful as labeled negatives for the custom ranking diagnostic.

### Unmatched track

The tracker can produce an object for which the RefAV annotation has no matching object. This row has no prompt-specific label. It remains in the candidate pool, but it is not silently converted into a negative.

### Log

A log is one continuous recorded driving sequence. A log is larger than one image. Splitting by log prevents frames from the same sequence appearing in both training and testing.

### Prompt

The natural-language description or question attached to a scenario. The current pilot uses exact prompt strings that repeat across selected logs. This is useful, but exact string overlap is not the same as a true semantic template holdout.

### Tracker score

The confidence value produced by the object tracker. It is not allowed as a learned DriveOne feature because it may correlate with the target label. It is used only as an explicit shortcut control.

### mAP and Recall@1

For each ranking group, average precision measures whether referred objects appear before negatives. mAP is the mean of average precision over groups. Recall@1 asks whether a referred object is ranked first. Because RefAV can have multiple referred objects, Recall@1 is secondary; mAP is more suitable.

### Calibration

Calibration asks whether a probability means what it says. If a model gives 0.8 to many predictions, roughly 80% of those predictions should be correct. We record NLL, Brier score, ECE, and reliability bins.

### Leakage

Leakage means that information related to the answer enters the input in an unfair way. Examples are using the relevance label, future timestamps, tracker confidence, or candidates that were selected because they were already known to be correct.

## 5. Full research roadmap

The roadmap is a sequence of research gates. Each gate answers one question and
produces an artifact that the next gate can use. This is important because a
large model can produce impressive numbers even when the candidate list, labels,
or train/test split already reveal the answer. We therefore validate the
experiment before increasing model size or adding more data.

### Stage 0 — Understand the contract

**Question:** What exactly is one prediction problem?

First define one **ranking group** as:

```text
(log_id, prompt, decision_timestamp)
    -> one shared camera frame
    -> all eligible tracker candidates at that time
    -> one or more REFERRED_OBJECT positives
    -> RELATED_OBJECT, OTHER_OBJECT, or unknown candidates
```

The group definition also fixes what the model may see. The question is an
input. The image is an input. A candidate representation may contain declared
geometry, but it may not contain the relevance label, future information, or a
candidate ID that has a hidden meaning. An unmatched track remains unknown; it
is not silently changed into `OTHER_OBJECT`.

This stage also decides which claim is being tested. The first claim is not
“the model drives safely.” It is narrower: can language and one image rank a
referred track above the other candidates? That makes RefAV useful as a cheap
diagnostic before testing trajectories or closed-loop planning.

**Output:** the data-contract document, label rules, candidate rules, leakage
rules, and a list of claims that this task cannot test.

**Do not proceed if:** the positive label, candidate pool, timestamp, or
unknown-row policy is ambiguous.

### Stage 1 — Prepare data

**Question:** Can the raw sources be converted into one reproducible table?

The preparation step pins the exact annotation artifact, tracker artifact,
Argoverse 2 sensor split, repository versions, and file hashes. Large files and
model weights stay outside Git, but their paths and hashes are recorded in the
manifest. This lets another person check whether they used the same inputs.

The adapter then performs the transformations that the raw sources do not
provide together:

1. Load prompt-specific RefAV annotations.
2. Load tracker frames and validate that every per-frame array has the same
   length as `track_id`.
3. Load the ego pose for each tracker timestamp.
4. Convert tracker cuboid centers from the city coordinate frame to the
   ego-vehicle frame.
5. Match tracker candidates to annotated objects using same-category,
   one-to-one matching within a fixed 2 m center-distance threshold.
6. Transfer the prompt label only when a valid match exists.
7. Find one camera image near the timestamp and project each candidate box into
   that image.
8. Apply the declared range and map-ROI filters.
9. Write one row per candidate, preserving unmatched candidates.

**Output:** a derived Feather manifest plus a preparation summary. The Feather
file is the fixed candidate table used by later controls; it is not an official
RefAV submission file.

**Do not proceed if:** the conversion depends on relevance labels to choose
timestamps, remove candidates, or select the camera frame.

### Stage 2 — Audit the data

**Question:** Does the derived table obey the written contract?

The verifier checks both structure and scientific validity. Structural checks
include required columns, numeric timestamps, array lengths, valid labels,
duplicate `(log_id, prompt, timestamp, track_id)` keys, and readable camera
paths. Scientific checks include:

- every ranking group has at least one referred track and a declared negative
  or unknown pool;
- every candidate in a group uses the same image path;
- log IDs do not cross train, validation, and test;
- prompt overlap is reported rather than assumed to be semantic holdout;
- label fields and future timestamps are excluded from learned features;
- camera projection status and missing-camera rows are visible;
- candidate counts and unmatched counts are recorded;
- source URLs, local paths, licenses, versions, and hashes are saved.

The audit is deliberately strict. A failed audit is not a minor warning if it
means the model would receive different candidate lists for different methods
or if test information entered the training features.

**Output:** a JSON audit report, manifest hash, coverage statistics, and an
explicit list of limitations.

**Do not proceed if:** log-disjoint evaluation is impossible, candidate
construction changes with the label, or the camera association cannot be
reproduced.

### Stage 3 — Check the PE interface

**Question:** Can the proposed frozen backbone actually expose the features
needed by DriveOne?

The smoke test loads the official `PE-Core-L14-336` checkpoint and runs the
released image and text paths. It records the checkpoint/configuration,
preprocessing, tensor shapes, token count, dtype, device, and latency. The
current real-image run found 577 sequence tokens including the class token,
576 patch tokens after removing it, token width 1024, pooled image width 1024,
text width 1024, and an official CLIP text context length of 32 tokens.

The test also checks the practical interface questions that shape the model:

- whether the returned sequence is before attention pooling;
- whether the image transform uses the expected resize and normalization;
- whether the prompt is truncated by the tokenizer;
- whether a projected candidate region can be represented consistently;
- whether the same checkpoint can produce cached features for every split.

**Output:** a shape/configuration report and a reproducible feature-extraction
command. This proves that the interface is available; it does not prove that
patch tokens improve ranking.

**Do not proceed if:** the checkpoint does not expose usable patch tokens, the
text is silently truncated, or the preprocessing differs between baselines.

### Stage 4 — Run deterministic controls

**Question:** Is the task already solvable from simple candidate properties?

Before training any neural scorer, rank the same candidate groups with simple
rules:

- random order, as a lower reference;
- tracker confidence, to expose tracker-quality shortcuts;
- nearest distance, to expose spatial priors;
- projected box area, to expose visibility and object-size priors;
- category frequency, to expose class imbalance;
- oracle order, as an upper bound using the target label.

The controls must use the exact same exported candidate rows as the learned
baselines. Report the full pool, not only candidates with known labels, because
unknown rows still push a referred object down in the real candidate list. The
main ranking measures are average precision per group and mean average
precision across groups. Recall@1 is also useful, but multiple referred
objects make it less complete than mAP.

**Output:** control metrics, candidate-count strata, per-log metrics,
score-matched hard-negative metrics, and the oracle ceiling.

**Do not proceed if:** a trivial control is already close to the proposed
model, or if the oracle ceiling is low because the candidate generator often
misses the referred object.

### Stage 5 — Audit shortcuts

**Question:** Does the apparent signal come from candidate construction rather
than image-language reasoning?

A candidate-only shortcut occurs when geometry, tracker metadata, or candidate
availability identifies the positive. To test this, create fixed comparisons:

1. Compare the full candidate pool with random negatives.
2. Match negatives to positives by tracker confidence.
3. Match again by tracker confidence and projected box size or log-area.
4. Keep those exact rows unchanged for every model and control.
5. Break metrics down by log, candidate count, projection status, and object
   category.

The matching operation is only an evaluation diagnostic. It must not look at
the target label while constructing the original candidate pool, and it must
not remove a candidate differently for one method than another. If the
projected-area control remains strong after score-and-size matching, then the
current protocol still carries a visibility or geometry signal. A model that
wins on the unpaired pool may only be learning that signal.

**Output:** a shortcut report that says whether the candidate protocol is
usable, repairable, or uninterpretable.

**Do not proceed if:** candidate-only or metadata-only controls remain within a
predeclared practical margin of the model, or if hard-negative construction
requires looking at relevance labels.

### Stage 6 — Frozen baseline gate

**Question:** Does language-conditioned image input add value after the simple
controls have been measured?

Run four small learned controls on fixed exported groups:

- **candidate-only:** declared candidate geometry and category features;
- **metadata-only:** explicit tracker or distance metadata used as a shortcut
  control;
- **task ID:** the same pooled image representation as the pooled-PE model,
  but with a learned ID for the prompt/task instead of natural-language text;
- **pooled PE:** the same candidate features and pooled PE image features,
  plus PE text features for the prompt.

The task-ID and pooled-PE models must use matched image inputs, candidate rows,
training data, optimizer budget, precision, batch size, and hardware. Otherwise
the comparison cannot answer whether language helps. The first run uses one
seed only to test reproducibility. After the command, data, and metrics are
stable, repeat with a second seed and report confidence intervals.

The gate reports mAP and Recall@1, but it also reports NLL, Brier score, ECE,
reliability bins, per-log results, candidate-count results, and hard-negative
results. Calibration matters because a ranking score may later be used for
abstention or thresholding. The learned baseline uses labeled rows for its
binary loss while retaining unknown rows in evaluation; this distinction is
recorded rather than hidden.

**Output:** a corrected baseline report with exact artifact hashes, seed,
feature-cache metadata, ranking metrics, calibration metrics, and timing
scope.

**Do not proceed if:** pooled PE fails to beat the strongest matched shortcut,
task ID is not a fair comparison, or the result cannot be reproduced from the
same cached features.

### Stage 7 — Minimal DriveOne model

**Question:** Do spatial visual tokens provide information that pooled features
discard?

Only after the pooled gate is meaningful, add the smallest proposed model:
frozen PE patch tokens, one question representation, one small
question-conditioned fusion block, one candidate projection, and one shared
scoring function. Use one image frame and the same candidate pool. Do not add
PE-Spatial, temporal frames, six cameras, Qwen, distillation, or trajectories.

The patch model is compared against pooled PE and task ID with matched
backbone, image resolution, candidate set, training data, precision, and
hardware. The previously selected practical threshold is at least +5 Recall@1
points or +0.03 mAP over the strongest matched baseline, with confidence
intervals excluding zero across two seeds. The threshold must be fixed before
looking at the final test result.

**Move on when:** the patch model improves held-out wording and log-disjoint
ranking without a material calibration regression and without relying on a
candidate-only shortcut.

### Stage 8 — Stress the result

**Question:** Does the improvement survive changes that matter at deployment?

Test candidate-count shifts, held-out wording and paraphrases, scene/log and
city disjointness, changed candidate generators, and independently generated
hard negatives. Keep the ranking task separate from candidate recall: a scorer
cannot receive credit for selecting an object that was never proposed.

Report per-log and per-count results, confidence intervals, calibration,
abstention coverage, and representative failures. A result that works only for
one prompt family or one tracker is a limited diagnostic, not generalization.

### Stage 9 — Expand the task

**Question:** Does the mechanism transfer to other bounded candidate types?

Add new tracks separately rather than combining them into one headline score:

- PE-Spatial-L14-448, with resolution, preprocessing, and compute accounted
  for;
- four-frame temporal input, with frame timestamps and motion assumptions
  stated;
- multiple cameras, with camera selection and missing-view rules stated;
- bounded DriveLM/nuScenes answers, only after defining candidate conversion;
- independently generated NAVSIM/GTRS trajectories, separating ranking from
  imitation, open-loop, and closed-loop metrics;
- Waymo preference ranking, only if the candidate and preference protocol is
  independently reproducible.

Every extension needs its own task-specific baseline and data-validity audit.
A strong entity result does not automatically validate trajectory scoring.

### Stage 10 — Distill last

**Question:** Can a larger model teach a smaller scorer without changing what
the task measures?

Only after the non-distilled student is understood should Qwen or another
teacher be evaluated. First compare the direct teacher score with ground truth,
simulator labels, or human/rater labels. Then compare training from labels
with training from teacher targets. Record teacher errors, confidence, prompt
truncation, and objective mismatch.

Distillation can reduce compute, but it cannot repair an invalid candidate
pool, missing positives, label leakage, or a shortcut. If the student copies a
teacher that ranks retrieval relevance rather than driving quality, the result
may look efficient while measuring the wrong thing.

### Why this order matters

The order separates four questions that are often mixed together:

1. **Data validity:** are the labels, candidates, images, and splits valid?
2. **Representation value:** do pooled or patch features add information over
   simple controls?
3. **Generalization:** does the gain survive new wording, logs, planners, and
   candidates?
4. **System value:** is the complete pipeline accurate, calibrated, and fast?

At the current point, data preparation, PE access, deterministic controls, and
the evaluator-correctness gate are complete. The corrected pooled baseline
model gate is the next step; patch tokens and all later stages remain paused.

## 6. The current data pipeline

The pipeline turns several incompatible raw sources into one fixed ranking
dataset. It has three boundaries:

1. **Preparation boundary:** raw annotations, tracker output, poses,
   calibration, images, and maps become candidate rows.
2. **Evaluation boundary:** the rows are checked, split, and exported into
   identical groups for every method.
3. **Model boundary:** images and cached PE features are passed to controls or
   a scorer. No model is allowed to rebuild the candidate pool.

Keeping these boundaries separate makes it possible to tell whether a failure
comes from missing data, bad labels, a shortcut, or the neural model.

```text
RefAV annotations
        + public tracker predictions
        + AV2 poses, calibration, camera images, map
                    |
                    v
        refav_tracker.py
        coordinate conversion, matching, filtering, projection
                    |
                    v
        derived Feather manifest
                    |
                    v
        verify_refav.py / refav_contract.py
        schema, leakage, split, coverage checks
                    |
                    v
        repeated-prompt log plan
                    |
                    v
        export_refav_subset.py
                    |
                    +--> deterministic controls
                    +--> PE feature cache
                              |
                              v
                    train_refav_baselines.py
```

### 6.1 Source annotations

The RefAV annotation file is the source of prompt-specific supervision. It
contains a log ID, a natural-language prompt, a timestamp, annotated object
geometry, and a mining category such as `REFERRED_OBJECT`,
`RELATED_OBJECT`, or `OTHER_OBJECT`. It does not contain everything needed by
DriveOne: in particular, it does not provide the public tracker confidence,
the derived camera image path, or a complete candidate list containing
unmatched tracker objects.

The annotation is therefore used for labels and matching targets. It is not
used to decide which timestamps are valid after preparation has started. That
distinction prevents the pipeline from selecting only timestamps where the
answer is easy to find.

### 6.2 Tracker predictions

The historical pilot used a Valeo4Cast-derived tracker file. The later source audit separately replayed the official RefAV Le3DE2E tracker; both source histories are kept so that old results are not silently rewritten. A frame
contains arrays such as:

```text
track_id, score, label, translation_m, size, name, yaw, timestamp_ns
```

All arrays in one frame must have the same length. The tracker output is not
ground truth. It is a proposal mechanism: it says which objects a deployed
system might ask the scorer to rank. This is why tracker confidence is treated
as a control and is excluded from the learned DriveOne input.

The loader sorts frames by their explicit timestamp and rejects duplicate
timestamps. Sorting makes the preparation deterministic when a source pickle
contains a small number of out-of-order frames; rejecting duplicates avoids an
ambiguous decision group.

### 6.3 Coordinate conversion

Tracker centers are stored in the city coordinate frame. A camera is mounted on
the ego vehicle, and RefAV matching is performed in the ego frame. For each
timestamp, the adapter loads the AV2 pose and applies the inverse pose transform:

```text
point_ego = inverse(city_from_ego_pose) * point_city
```

The converted position is used to calculate ego-relative distance and to build
the candidate box for camera projection. The vehicle yaw is also removed when
the tracker box yaw is written to the manifest. If this transform were wrong,
the same object could appear far from its true location, fail matching, or be
projected into the wrong image area.

### 6.4 Selecting decision timestamps without labels

For each prompt, the adapter starts from all timestamps present in the
annotation file. It keeps a timestamp only when infrastructure is available:

- the tracker has a frame at that timestamp;
- the AV2 pose table has a row at that timestamp;
- a `ring_front_center` image exists within the configured 100 ms tolerance.

The selection function does not inspect `REFERRED_OBJECT`, `RELATED_OBJECT`, or
`OTHER_OBJECT` when making this decision. The resulting timestamp list is the
observation window for that prompt. Later, candidate rows are created for every
selected prompt/timestamp pair.

This is a key anti-leakage rule. If timestamps were kept only when a referred
track was visible, the candidate pool would already encode the target.

### 6.5 Matching tracker candidates to annotated objects

At one prompt and timestamp, the adapter groups tracker candidates and
annotations by object category. It computes center distances in the ego frame
and solves a Hungarian assignment separately for each category. A pair becomes
a match only when its center distance is at most 2 m.

The one-to-one rule matters. Without it, several tracker boxes could inherit
the same annotation label, which would inflate the number of positives and make
the ranking problem artificial. A matched annotation transfers its mining
category to the tracker candidate. A candidate with no match receives a null
label and `UNMATCHED_TRACK` status; it stays in the group.

The adapter also records the match distance and annotated track UUID. These
fields are useful for audits, but they are not allowed as learned candidate
features because they are close to target construction.

### 6.6 Choosing one shared camera frame

The initial pilot uses one camera, `ring_front_center`, for every candidate in
a group. The nearest image timestamp is selected within 100 ms of the decision
timestamp. The selected path, image timestamp, and time difference are written
to every row in the group.

Sharing the image is intentional: a model must compare candidates under the
same visual evidence. It also makes the first experiment cheap. The limitation
is that objects outside this camera view remain difficult or impossible to
recognize. They are kept with `OUT_OF_VIEW` status instead of being deleted,
because deleting them would make candidate availability depend on visibility.

### 6.7 Projecting candidate boxes

For each tracker cuboid, the adapter creates its eight 3-D corners in the ego
frame. It uses the AV2 camera extrinsics and intrinsics to project those corners
into the image. The smallest valid rectangle around the visible corners is
stored as:

```text
(x_min, y_min, x_max, y_max)
```

Coordinates are clipped to the camera image dimensions. A candidate is marked
`PROJECTED` only when it has valid depth and a box area greater than one pixel.
Otherwise it receives `OUT_OF_VIEW` or a missing-camera status. The projection
is used for geometry controls and future crop experiments, but projected area
must be treated as a possible shortcut rather than as evidence of grounding.

### 6.8 Range and map-ROI filtering

The adapter removes tracker boxes beyond the configured 50 m distance and
boxes outside the available AV2 region-of-interest raster. These filters limit
the pilot to the part of the scene supported by the source evaluation setup.
They are infrastructure filters, not relevance labels. The code records the
ROI decision and keeps the filter rule fixed for all methods.

### 6.9 The derived candidate row and group

After these operations, one row contains source bookkeeping, prompt and time,
tracker identity, tracker metadata, ego-frame geometry, transferred label,
camera metadata, projection status, and match status. A ranking group is all
rows with the same `(log_id, prompt, timestamp_ns)`.

The model never receives the whole manifest as an unstructured table. Later
scripts load one group, construct the same candidate list for every baseline,
and pass the image and question plus one representation per candidate. This
prevents a model from benefiting simply because its method received more
candidate rows.

### 6.10 Contract verification

`verify_refav.py` loads the manifest through `refav_contract.py`. The contract
layer normalizes aliases, converts known label names to numeric IDs, checks
required fields, and preserves unknown fields for auditing. It then checks
duplicate keys, valid label values, shared camera association, group coverage,
log-disjoint splits, prompt overlap, and feature-redaction rules.

The verifier produces an audit JSON with counts and failures. In strict mode a
failed required check returns a nonzero exit code. This makes the audit part of
the experiment rather than a manual claim in a notebook.

### 6.11 Planning splits and exporting a fixed subset

The split planner reads only `log_id` and `prompt` to find exact prompt strings
that occur in at least three logs. It selects disjoint log triplets and assigns
one log to train, one to validation, and one to test. This gives repeated
wording across splits while preventing frames from the same log from crossing
the split boundary. It does not prove that paraphrases or semantic templates
are held out; that is a later experiment.

The subset exporter then selects a seeded list of rankable group keys from each
split. A rankable group has at least one referred row and at least one labeled
negative. It copies every candidate row for each selected group, including
unknown rows, into Feather and JSONL files. The exporter records the source
hash, selected group keys, output hashes, and random seed. Every baseline uses
these same exported groups.

### 6.12 PE feature caching and baseline input

The feature extractor loads PE once, applies the official image transform and
tokenizer, and writes pooled image features and question text features outside
Git. Caching makes baseline comparisons faster and ensures that candidate-only,
task-ID, and pooled-PE models see the same frozen image representation.

The cache is not an end-to-end latency result. It omits image decoding,
resizing, PE forward time, candidate construction, calibration, and ranking.
Those costs are measured later in the efficiency track.

### 6.13 Where the pipeline currently stops

The present pipeline ends after deterministic controls and the corrected
candidate-source audit. The next run is a frozen pooled-PE baseline on the
causal all-tracker pool. The patch-token fusion model is still deferred. That
pause is deliberate: the old pooled result used a different historical subset,
and we need a matched result under the corrected evaluator before comparing
representations.

This is why the pipeline is more than file conversion. It is the experiment’s
measurement instrument. If its candidate groups or labels are invalid, a larger
model would only make the invalid measurement more expensive.

## 7. What one candidate row means

A row in the derived manifest contains, among other fields:

| field | meaning | allowed in learned input? |
| --- | --- | --- |
| `log_id` | driving sequence ID | split bookkeeping only |
| `prompt` | natural-language description | yes, as question input |
| `timestamp_ns` | decision time | grouping only; not a candidate feature |
| `track_id` | tracker object ID | no |
| `score` | tracker confidence | no; control only |
| `label` | referred/related/other/null | target only; never input |
| `translation_m` | ego-frame position | allowed only in a declared candidate representation |
| `size` | object dimensions | allowed only in a declared candidate representation |
| `name` | tracker object class name | treated carefully; raw class controls are explicit |
| `raw_tracker_label` | tracker class code | allowed as a declared category control |
| `distance_m` | ego-relative distance | allowed in candidate geometry, but must be audited as a shortcut |
| `image_path` | shared front-center frame | used to load the image |
| `projected_box` | camera projection of the candidate | allowed for the current geometry control, but can be a shortcut |
| `projection_status` | visible/out-of-view/missing | audit information |

The main rule is simple: a field must be named before it is used. If a field is related to the answer, it belongs in a control or must be excluded.

## 8. The files and their jobs

### Configuration

**`configs/refav_pilot.yaml`** is the written experiment contract. It records source repositories, commits, artifact hashes, candidate rules, split rules, metrics, PE shape results, corrected baseline hashes, image size, and stop conditions. It is not a general YAML engine; the verifier reads the small subset used here.

### Data modules

**`src/driveone/data/refav_contract.py`** is the general data checker.

- Loads Feather, JSON, CSV, and official-style pickle records.
- Normalizes different field names into one schema.
- Converts known string labels into numeric labels.
- Checks required fields and array lengths.
- Detects duplicate candidates and invalid labels.
- Checks camera association, timestamps, positive/negative groups, prompt normalization, log disjointness, and feature redaction.
- Writes manifest hashes and audit summaries.

This file does not train a model or project a cuboid into an image. It checks the contract.

**`src/driveone/data/refav_tracker.py`** builds the derived candidate manifest.

- `load_tracker_pickle` loads and validates tracker frames.
- `global_to_ego` changes city-frame positions into the ego frame.
- `match_candidates` performs same-class Hungarian matching.
- `build_camera_file_index` and `_camera_index` find the nearest camera image.
- `build_camera_models` loads AV2 calibration.
- `roi_mask_for_frame` applies the map ROI check.
- `project_candidate` projects 3-D cuboid corners into a 2-D camera box.
- `select_decision_timestamps` keeps aligned prompt timestamps without looking at relevance labels.
- `prepare_records` combines all operations and writes candidate rows.

**`src/driveone/data/refav_repair.py`** is the stricter adapter used for the
official Le3DE2E replay. It builds the candidate pool first, then associates
tracker rows with ground truth. This order is important: matching success and
prompt labels cannot decide which candidates enter the pool. It also keeps the
four association states `MATCHED_ANNOTATED`, `MATCHED_UNANNOTATED_GT`,
`UNMATCHED_TRACK`, and `AMBIGUOUS_MATCH`, and contains the fixed class
compatibility map, Hungarian matching, seven-camera projection, and candidate
pool hashing rules.

**`src/driveone/data/refav_candidate_sources.py`** is the source-comparison
layer. It loads the official replay, the causal all-tracker pool, the older
historical pool, and a ground-truth oracle pool under one common schema. It
computes coverage, matching sensitivity, projection coverage, unknown-row
rates, per-log statistics, candidate-count strata, and deterministic control
inputs. It does not train a model.

**`src/driveone/data/refav_splits.py`** plans log-disjoint repeated-prompt splits. It finds exact prompt strings that occur in at least three logs, selects disjoint log triplets, and assigns one log in each triplet to train, validation, and test. It does not prove semantic template separation.

### Evaluation module

**`src/driveone/eval/refav_metrics.py`** implements deterministic controls.

- `_rank_order` defines random, tracker-score, distance, projected-area, category-frequency, and oracle orderings.
- `_group_metrics` computes candidate count, positive/negative counts, AP, and Recall@1.
- `_hard_negative_rows` keeps negatives close to positive tracker score and projected size.
- `run_control_suite` evaluates full pools, score-matched negatives, and score-plus-size-matched negatives.

Unknown candidates stay in the ranking order but are excluded from positive/negative label counts. This is important: an unknown row can still push the correct object down.

### Command-line scripts

**`scripts/prepare_refav_tracker.py`** is the command-line wrapper around the tracker adapter. It receives tracker, annotation, sensor, and log paths and writes a Feather manifest plus a summary.

**`scripts/verify_refav.py`** reads the pilot config and candidate manifest. It writes a JSON audit and returns a nonzero status in strict mode when required checks fail.

**`scripts/select_refav_log_plan.py`** reads the full annotation table and writes the repeated-prompt log plan.

**`scripts/run_refav_controls.py`** loads a manifest and writes the deterministic control report. It can also evaluate each planned split separately.

**`scripts/smoke_test_pe.py`** checks the official PE interface. It loads the checkpoint, runs `visual.forward_features` with and without the class token, runs pooled image encoding, runs text encoding, and records tensor shapes, text length, dtype, and timing. It does not train DriveOne.

**`scripts/export_refav_subset.py`** selects a seeded set of rankable groups and copies every candidate row in those groups into Feather and JSONL files. The manifest records selected group keys and hashes.

**`scripts/extract_pe_features.py`** loads PE once, applies the official image transform and tokenizer, extracts pooled image and text features, and saves them outside Git. The feature cache prevents repeated PE computation during baseline training.

**`scripts/train_refav_baselines.py`** runs the four first learned controls.

- `candidate_only`: candidate geometry and tracker category, no image or question.
- `metadata_only`: tracker score, raw tracker label, and distance; this is an explicit shortcut control.
- `task_id`: candidate geometry, pooled PE image, and a learned prompt ID.
- `pooled_pe`: candidate geometry, pooled PE image, and PE text features.

It also validates log-disjoint splits, duplicate candidates, shared images, labels, source image dimensions, feature counts, and pretrained PE metadata. It reports ranking, calibration, per-log, and candidate-count metrics. The baseline uses BCE on labeled rows. Unknown rows are left out of the loss but remain in evaluation.

**`scripts/repair_refav_protocol.py`** runs the official Le3DE2E repair. It
replays the pinned RefAV conversion, creates a label-independent candidate pool,
performs fixed-threshold association, projects every candidate into the seven
ring cameras, and writes a repaired manifest plus provenance summary.

**`scripts/audit_refav_candidate_sources.py`** is the latest diagnosis script.
It compares candidate sources without fitting DriveOne. It uses Arrow
column-selection and filtering so the 17-million-row annotation file can be
read without materializing the whole file as Python dictionaries. It also
replays the official conversion functions from the pinned RefAV repository and
records the source code hash. Its output is the external JSON report whose
decision is `POOLED_BASELINE_GATE_READY`; the learned-model gate is recorded
separately as pending.

### Tests

The `tests/` directory uses Python `unittest`.

- `test_refav_contract.py` checks schema normalization and leakage rules.
- `test_refav_tracker.py` checks matching and timestamp behavior.
- `test_refav_metrics.py` checks AP, unknown-row handling, and hard-negative controls.
- `test_refav_splits.py` checks disjoint prompt-overlap planning.
- `test_verify_script.py` checks the dependency-light verifier behavior.
- `test_refav_baselines.py` checks feature shape, unknown-row ranking, and that task ID and pooled PE accept the same image input.

The current suite has **35 passing tests** in the `refav` environment. Tests use
small synthetic records; they check code behavior and contract logic, not model
quality or dataset validity.

### Documentation and policy

**`docs/initial-experiment.md`** gives the experiment order and metric rules.

**`docs/refav_data_contract.md`** records the formal data contract and the real-data audit history.

**`docs/refav_baseline_smoke.md`** records the corrected baseline protocol, exact results, artifact hashes, and reproduction commands.

**`docs/obsidian/`** contains the learning path, roadmap, current status, glossary, code walkthroughs, and decision log. Those notes are useful for detail, but this guide is the single linear explanation.

**`CONTRIBUTING.md`** states the research rules: pin inputs, exclude target fields, do not claim safety or real time without measurements, and keep large artifacts outside Git.

**`.gitignore`** keeps data, weights, caches, runs, and generated reports out of commits. Large artifacts live under the external asset root, currently `/ehsan/m.sadegh/driveone_assets/refav`.

## 9. What has been done so far — a complete ledger

This section records the work in the order it happened. A result is marked as a
**measurement**, **code/infrastructure**, or **interpretation** so that a
future reader can see exactly what is known and what still needs testing.

### 9.1 Repository and project boundary — code/infrastructure

The standalone Git repository was created at `/data/sadegh/driveone`, separate
from sibling projects such as `fail2drive`. Large data, model weights, cached
features, run outputs, and generated reports are ignored by Git. The external
asset root is `/ehsan/m.sadegh/driveone_assets/refav`.

The repository now contains the data adapters, contract checks, split planner,
control metrics, PE smoke test, baseline scripts, tests, configuration, and
plain-language documentation. The due-diligence report and original proposal
are reference documents; they are not silently treated as training data.

### 9.2 Formal data contract — code/infrastructure

We defined a ranking group as `(log_id, prompt, decision_timestamp)` with one
shared visual observation and all eligible candidates at that time. Candidate
membership must be decided without looking at `REFERRED_OBJECT`,
`RELATED_OBJECT`, `OTHER_OBJECT`, matching success, or future timestamps.
Unknown tracker rows are kept and marked explicitly. This prevents a model from
being rewarded for a candidate list that was hand-built around the answer.

### 9.3 Raw source inspection — measurement

The official RefAV validation annotation artifact was inspected and pinned:

```text
scenario_mining_val_annotations.feather
SHA-256 e461e51057fdf347a11bdd60609e7de0b0bd8d0eb199314b3fd73c50106d48c9
17,254,820 rows; 150 logs; 403 prompt strings
```

The official RefAV repository is pinned to commit
`5c5be6439ce59b61a31d56431a79a8a04bba33fa`. The official Le3DE2E validation
tracker is pinned to:

```text
Le3DE2E_tracking_predictions_val.pkl
SHA-256 fd702ade8b640d325e5096e90f63cf1bf43f94a87a6aabfb52e71aa7b8470875
Hugging Face revision d983c7955b1c6a126542bea0a4dfb3a2c7327e6a
```

The repeated-prompt plan is also pinned:

```text
refav_repeated_prompt_log_plan.json
SHA-256 cba73f6f34a5373bf3a5f69765b78f5eeebf48ad2ebdab38032a204cd7c60f85
```

The sources have different jobs. The scenario file supplies prompt-specific
annotations. The tracker supplies possible deployed candidates. Argoverse 2
poses, calibration, maps, and images supply the coordinate and camera
information. No single source contains the complete experiment table.

### 9.4 First tracker manifest and old baseline — measurement and historical warning

The first pilot used a historical Valeo4Cast-derived tracker file and a
front-center image policy. We built a nine-log repeated-prompt subset, exported
fixed candidates, cached PE features, and ran deterministic and frozen pooled
baselines. That work was useful for finding problems, but it is not the final
benchmark because the source conversion and unknown-label handling were not yet
settled.

The corrected 500-group-per-split test numbers were:

| method | test mAP |
| --- | ---: |
| pooled PE image + question | 0.0476 |
| task ID + the same pooled image | 0.0576 |
| tracker-score ranking | 0.1813 |
| projected-box-area ranking | 0.1729 |

The pooled model therefore did not pass the predeclared gate against simple
controls. These numbers are retained as a historical warning. They are not
evidence about patch tokens, Qwen, planning, safety, or real-time performance.

### 9.5 PE smoke test — measurement

The official `PE-Core-L14-336` interface was loaded on a real Argoverse 2 image
using the dedicated PE environment. The observed output was:

- 577 sequence tokens including the class token;
- 576 patch tokens after removing the class token;
- token width 1024;
- pooled image width 1024;
- text feature width 1024;
- official CLIP text context length 32;
- image preprocessing to 336×336, while the source camera image is 1550×2048.

This proves that the proposed patch-token input is technically available at the
checkpoint tested. It does not prove that patch tokens contain useful task
signal, and it does not measure end-to-end latency.

### 9.6 Protocol diagnosis — measurement

The CPU-only diagnosis of the old 500-group artifacts found many unknown rows,
split-dependent visibility differences, and strong tracker/geometry controls.
It produced `PROTOCOL_REPAIR_REQUIRED`. That decision stopped patch-token
training and led to the evaluator-correctness gate. The old result remains a
historical warning; it is not silently mixed with the corrected source audit.

### 9.7 Official-source replay and camera completion — code/infrastructure and measurement

The official RefAV tutorial was checked and the Le3DE2E source was replayed on
the nine selected logs. Candidate construction occurred before target
association. The replay used native tracker timestamps, fixed 50 m/ROI filters,
all eligible tracker candidates, a fixed class map, one-to-one matching, and a
fixed seven-camera order.

The earlier local copy had only front-center images for most logs. To make the
camera-coverage conclusion testable, we downloaded only the nearest image to
each of the 32 native tracker timestamps for each of the seven ring cameras and
nine logs: 1,536 JPEG files, about 0.55 GB, with a maximum timestamp difference
of 32.7 ms. No full Argoverse sensor download was performed.

### 9.8 Evaluation-correctness correction — measurement

The first v3 evaluator marked a match ambiguous whenever either endpoint had
more than one valid geometric edge. That was too strict: a global one-to-one
assignment can be unique even when a candidate has several local alternatives.
The v4 evaluator now records both facts:

- `multiple_valid_edges`: a local crowded-scene diagnostic;
- `AMBIGUOUS_MATCH`: only an equal-cardinality, equal-cost global alternative.

The v4 report found no equal-cost global assignment ties. It did find many
local alternatives: 3,964 causal assignments and 3,528 official-replay
assignments at 2 m. Those numbers justify inspecting crowded examples, but they
do not justify NMS by themselves.

Unknown rows now retain `label=None` and their status. The controls report two
metric bounds. The operational bound keeps unknown rows in the ranking order;
the labeled-only bound excludes unknown rows from the metric denominator. This
makes it possible to see both deployment-style difficulty and label-supported
ranking quality.

Synthetic `EGO_VEHICLE` rows remain in the candidate pool and its hash, but are
excluded from external-object controls. Ego scenarios are reported separately.
The learned model gate is also separate: second-seed pooled PE is `PENDING`,
not a failed data check.

### 9.9 Corrected candidate-source results — measurement

The v4 audit evaluates 2,880 prompt/timestamp groups. External availability is
conditional on an eligible external referred ground-truth object; it is not the
fraction of all timestamps containing a scenario event.

| source | external availability at 2 m | positive projects into a camera | unknown candidate fraction |
| --- | ---: | ---: | ---: |
| causal Le3DE2E pool | 99.0% | 100.0% | 88.7% |
| official replay | 97.0% | 100.0% | 74.1% |
| historical pool | 99.0% | 100.0% | 88.7% |
| ground-truth oracle | 100.0% | 100.0% | 0.0% |

For the causal pool, availability is 98.3%, 99.0%, and 99.1% at 1 m, 2 m,
and 4 m. The maximum change is 1.07 percentage points. The official replay is
96.4%, 97.0%, and 97.3%, a 0.87-point range. The official replay still uses a
whole-log score filter and is therefore a comparison source, not the causal
online pool for the next model run.

The causal pool has 603,230 unmatched rows out of 680,040 (88.7%). This does
not mean that 88.7% of referred objects are missing. It means that many tracker
rows have no matching annotation. Candidate-list size and unknown-row placement
still matter for ranking, so the model evaluation must retain them.

### 9.10 Corrected external controls — measurement

External controls exclude synthetic ego rows and use the same candidate rows
for every rule. For the causal pool:

| control | operational mAP | labeled-only mAP | operational Recall@1 | labeled-only Recall@1 |
| --- | ---: | ---: | ---: | ---: |
| tracker confidence | 0.264 | 0.274 | 0.166 | 0.169 |
| projected box area | 0.143 | 0.289 | 0.110 | 0.164 |
| candidate distance | 0.078 | 0.242 | 0.056 | 0.123 |
| category frequency | 0.027 | 0.226 | 0.006 | 0.129 |
| random | 0.035 | 0.222 | 0.010 | 0.114 |

The controls are clearly stronger than random, but none is near-perfect after
correction. The strongest control varies by log and by metric bound. The audit
also writes per-log, split, candidate-count, and prompt-holdout results, so the
next model report must not use only one pooled headline number.

### 9.11 Current data decision — interpretation

The evaluator-correctness gate passes. Candidate membership is hash-stable and
label-independent; camera projection is complete for the selected assets;
external availability exceeds 80%; matching sensitivity is stable within 10
percentage points; unknown statuses are explicit; ego rows no longer inflate
external controls.

The machine-readable decision is **`POOLED_BASELINE_GATE_READY`**. This is a
data decision. It authorizes the frozen pooled-PE baseline only; it does not
claim that PE or language improves ranking.

## 10. What the current evidence means

### Statements supported by the work

- The repository can load and audit the relevant RefAV/Argoverse artifacts.
- The official PE checkpoint exposes usable pooled and patch-token features.
- A reproducible candidate table and log-disjoint split plan can be built.
- The evaluator can preserve unknown rows and report operational and
  labeled-only ranking bounds.
- The corrected causal pool has about 99% external referred availability at
  the 2 m threshold.
- The selected camera assets provide complete geometric projection coverage.
- Candidate hashes are unchanged by the evaluator correction.
- Simple tracker and geometry controls remain required baselines.

### Statements that remain unproven

- Natural language improves generalization over task ID.
- Pooled PE beats the strongest matched control on the corrected groups.
- Patch tokens improve ranking over pooled PE.
- One shared scorer works across answers, entities, and trajectories.
- The formulation survives candidate-count, scene, city, planner, or dataset
  shift.
- Qwen3-VL-Reranker-2B is a fair autonomous-driving baseline or teacher.
- A compact scorer has an end-to-end quality/latency/memory advantage.
- Any offline ranker is safe, grounded in the deployment sense, a planner, or
  real time.

### Why the next experiment is still small

The data gate now passes, but the old learned result used different historical
rows and cannot be used as the corrected model result. We must rerun candidate-
only, metadata-only, task-ID, and pooled-PE controls on the same corrected
external rows. This is cheaper and more informative than adding patch tokens.

## 11. The next step after this document

Run the **frozen pooled-baseline gate** on the causal all-tracker pool:

1. Use the v4 candidate rows and keep the candidate hash unchanged.
2. Exclude synthetic ego rows from the external-object model evaluation.
3. Train candidate-only, metadata-only, task-ID, and pooled-PE controls with
   identical rows, optimizer budget, precision, and hardware.
4. Leave unknown candidates out of the BCE training loss, but retain them in
   ranking order during evaluation. Report operational and labeled-only bounds.
5. Run one seed first. If it reproduces, run a second seed and compute
   confidence intervals.
6. Report mAP, Recall@1, per-log, split, candidate-count, hard-negative,
   calibration, and end-to-end timing results.

Proceed to patch-token fusion only if pooled PE beats the strongest matched
baseline by at least +5 Recall@1 points or +0.03 mAP, with two-seed confidence
intervals excluding zero and no material calibration regression.

Do not apply NMS merely because local matching alternatives exist. If crowded
examples suggest duplicate proposals, test a fixed label-independent NMS or
track-consistency variant as a separate hashed candidate-pool variant before
using it for a model comparison.

Do not start Qwen, distillation, PE-Spatial, temporal frames, trajectories, or
deployment optimization before this pooled gate passes.

## 12. Reproduction commands

The commands below assume the external data already exists. They are examples of the order, not a request to download the full Argoverse 2 dataset.

### Run the unit tests

```bash
cd /data/sadegh/driveone
conda run -n refav python -m unittest discover -s tests -v
conda run -n driveone-pe python -m py_compile scripts/*.py
```

### Check the PE interface

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n driveone-pe python scripts/smoke_test_pe.py \
  --image /path/to/real/ring_front_center.jpg \
  --config PE-Core-L14-336 --pretrained --device cuda \
  --output /ehsan/m.sadegh/driveone_assets/refav/pe_smoke.json --strict
```

Inside `driveone-pe`, `cuda` refers to the visible device after `CUDA_VISIBLE_DEVICES` remaps it. Do not use the old base environment for the official PE run if it lacks the required packages.

### Run corrected controls on an existing subset

```bash
conda run -n refav python scripts/run_refav_controls.py \
  --records /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/refav_test_subset.feather \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/controls_test.json \
  --seeds 0 1 --hard-negative-delta 0.05 --size-matched-log-area-delta 0.2
```

### Run the corrected learned baselines

```bash
conda run -n refav python scripts/train_refav_baselines.py \
  --train /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/refav_train_subset.jsonl \
  --validation /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/refav_validation_subset.jsonl \
  --test /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/refav_test_subset.jsonl \
  --features /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/pe_core_pooled_features.pt \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_baseline_subset_500/baseline_results_seed0_v2.json \
  --device cpu --epochs 12 --seed 0 --threads 2 \
  --image-width 1550 --image-height 2048
```

The learned baseline uses cached PE features. Therefore this command is a quality diagnostic, not an end-to-end latency measurement.

### Rebuild this PDF

The PDF is generated from this Markdown guide. The PDF itself is ignored by
Git so that it is not pushed with the source repository. Install the small
documentation dependency once, then run:

```bash
cd /data/sadegh/driveone
python -m pip install -r requirements-docs.txt
python scripts/render_project_guide_pdf.py \
  docs/DriveOne_Project_Guide.md reports/DriveOne_Project_Guide.pdf
```

The renderer includes the source guide, tables, code blocks, page numbers, and
the update date. It does not read the external datasets or model weights.

## 13. Evidence levels used in this project

**Verified fact** means it was observed in an official source, local file, or completed run. Example: the PE run returned 576 patch tokens.

**Mathematical consequence** means it follows directly from the model definition. Example: a bounded scorer cannot select a candidate that is absent from its candidate list.

**Inference** means a reasonable interpretation of several facts. Example: the strong projected-area control suggests the current candidate protocol may favor visible or large objects.

**Speculation** means a possible explanation that still needs an experiment. Example: RefAV label transfer may correlate with tracker geometry strongly enough to make image content redundant.

Keeping these levels separate prevents a polished architecture from being mistaken for evidence.

## 14. Final mental model

The project is not currently trying to build a full autonomous-driving system. It is testing whether a carefully defined candidate-scoring interface deserves more investment.

The correct chain of reasoning is:

```text
Can we define valid candidates?
    -> Can we build them without answer leakage?
    -> Do simple controls fail to solve the task?
    -> Does pooled image/question conditioning add value?
    -> Do patch tokens add value beyond pooled features?
    -> Does the result survive wording, scene, and candidate shifts?
    -> Does it transfer to trajectories or other tasks?
    -> Is the full system faster end to end?
    -> Only then: distill and expand.
```

At the current point, the evaluator-correctness gate has passed, while the corrected pooled model gate is still pending. The next decision is whether pooled image/question conditioning adds value over matched controls, not whether to add a larger model.

## 15. The final audit in technical detail

This section gives the details needed to reproduce or challenge the current
data/model gate decision. The code and JSON report are the primary record; the prose here
explains how to read them.

### 15.1 Pinned inputs

The final audit used:

| input | purpose | pin |
| --- | --- | --- |
| RefAV repository | official conversion and tutorial | commit `5c5be6439ce59b61a31d56431a79a8a04bba33fa` |
| `scenario_mining_val_annotations.feather` | prompt and ground-truth mining annotations | SHA-256 `e461e51057fdf347a11bdd60609e7de0b0bd8d0eb199314b3fd73c50106d48c9` |
| `Le3DE2E_tracking_predictions_val.pkl` | official tracker candidates | SHA-256 `fd702ade8b640d325e5096e90f63cf1bf43f94a87a6aabfb52e71aa7b8470875` |
| repeated-prompt plan | selected nine logs and split roles | SHA-256 `cba73f6f34a5373bf3a5f69765b78f5eeebf48ad2ebdab38032a204cd7c60f85` |
| AV2 validation sensor/calibration/pose assets | projection and nearest-image lookup | external asset root, paths in the audit manifest |

The nine logs are three log-disjoint train/validation/test triplets. The
selection came from repeated exact prompt strings, not from choosing logs with
successful matches. Exact prompt repetition is useful for this pilot, but it
is not a full semantic paraphrase holdout.

### 15.2 How the candidate pool is built

For every selected native tracker timestamp, the audit:

1. reads all tracker rows available at that timestamp;
2. applies only fixed infrastructure filters: finite values, supported
   geometry, valid timestamp, fixed 50 m range, and the fixed ROI;
3. keeps every eligible row before any ground-truth association;
4. sorts and hashes the candidate keys;
5. associates the already-built pool with ground truth using the fixed class map,
   one-to-one Hungarian matching, and a distance threshold;
6. transfers `REFERRED_OBJECT`, `RELATED_OBJECT`, or `OTHER_OBJECT` only after
   the association;
7. leaves unmatched rows as `UNMATCHED_TRACK` or `AMBIGUOUS_MATCH`;
8. projects every candidate into every available ring camera in a fixed order.

This ordering is the main anti-leakage property. A candidate cannot enter or
leave the pool because a RefAV prompt says that it is relevant.

The pool hash uses sorted `(log_id, timestamp_ns, track_id)` keys plus declared
candidate geometry fields. Prompts and target labels are not used to decide
membership. The audit also checks duplicate keys and log-split overlap.

### 15.3 Why matching is now separated into two measurements

The tracker and the annotation describe the same scene in different ways. A
tracker can create several nearby boxes for one annotated object, or miss an
annotated object. The v4 evaluator therefore separates:

- **local alternatives:** more than one candidate-to-ground-truth edge is under
  the distance and class threshold;
- **global ambiguity:** removing the chosen edge leaves another assignment with
  the same maximum number of matches and effectively the same total distance;
- **unmatched:** no valid assignment is made for that tracker row.

The causal pool has many local alternatives but no equal-cost global assignment
ties in the report. Its group availability is 98.3%, 99.0%, and 99.1% at 1 m,
2 m, and 4 m. The range is 1.07 percentage points. This supports running the
pooled baseline while keeping crowded-scene diagnostics visible.

A future NMS or track-consistency rule must be label-independent, deterministic,
and separately hashed. Nearby objects alone are not enough evidence to delete
one of them.

### 15.4 Why the camera result changed

The first repair used an incomplete local camera copy, so it reported many
missing projections. That was a storage/input problem, not evidence that
Argoverse camera geometry was unusable. The final audit downloaded the nearest
frame for all seven ring cameras at each selected native tracker time. With
those files present, the ground-truth oracle and the matched positives projected
into at least one camera for 100% of eligible groups.

This does not say every object is visible or unoccluded. A geometric cuboid
intersection only says that the projected box has valid image support. Visual
clarity and occlusion still require separate measurement.

### 15.5 How to read the control scores

Tracker confidence, distance, projected area, and category frequency are
metadata/geometry rules. Their role is to show whether the target label can be
predicted without image-language content. The corrected causal results are
tracker mAP 0.264 operational / 0.274 labeled-only, projected-area mAP 0.143 /
0.289, distance mAP 0.078 / 0.242, and random mAP 0.035 / 0.222.

The operational bound keeps unknown rows in the ranking order. The labeled-only
bound excludes unknown rows from the metric denominator. Neither bound is a
safety result. Both must be reported for the next learned comparison.

Score-matched and size-matched subsets remain evaluation diagnostics. Their
construction uses target-associated rows, so they must not be called
deployment-like candidate pools.

### 15.6 Unknown labels and ego scenarios

An unmatched tracker row is not the same thing as a human-labelled
`OTHER_OBJECT`. The manifest keeps that distinction. Synthetic ego rows remain
in the pool for hash and scenario accounting, but are excluded from external
controls. Ego scenarios are reported separately rather than treated as
camera-renderable external objects.

The v4 source audit does not claim calibrated probabilities. NLL, Brier, ECE,
abstention, and reliability diagrams belong to the upcoming corrected learned
baseline, after the label policy is fixed for that run.

## 16. What is in the repository and what is outside it

### Tracked in Git

- `configs/refav_pilot.yaml`: pinned sources, rules, thresholds, and artifact
  references.
- `src/driveone/data/refav_contract.py`: schema and leakage checks.
- `src/driveone/data/refav_tracker.py`: historical tracker-to-manifest adapter.
- `src/driveone/data/refav_repair.py`: official-replay repair and matching.
- `src/driveone/data/refav_candidate_sources.py`: source comparison and audit
  calculations.
- `src/driveone/eval/refav_metrics.py`: ranking controls and hard-negative
  diagnostics.
- `scripts/prepare_refav_tracker.py`: historical manifest command.
- `scripts/repair_refav_protocol.py`: official Le3DE2E repair command.
- `scripts/audit_refav_candidate_sources.py`: final source audit command.
- `scripts/verify_refav.py`, `scripts/run_refav_controls.py`,
  `scripts/train_refav_baselines.py`, and the PE scripts.
- `tests/`: 43 small contract, matching, metric, split, verifier, bootstrap,
  adapter, and baseline tests.
- `docs/`: data contract, baseline record, protocol diagnosis, candidate-source
  decision, and this guide.

### Kept outside Git

The large annotation and tracker files, AV2 sensor assets, camera JPEGs, PE
weights, cached PE tensors, candidate Feather/JSONL exports, controls, and
machine-readable audit reports are under
`/ehsan/m.sadegh/driveone_assets/refav`. This keeps Git reviewable and prevents
accidental commits of multi-gigabyte data or model weights.

The corrected evaluator-audit directory is:

```text
/ehsan/m.sadegh/driveone_assets/refav/candidate_source_audit_20261008_v6
```

Its JSON report SHA-256 is
`52aff5c738bd674969c302f0fde3572b2e21e134a99e843b1182f7290157e9e7`. The
manifest, per-log coverage table, prompt-holdout plan, source-comparison table,
controls, and candidate Feather files are in the same directory.

## 17. What has not been done yet

The following items are deliberately **not** complete:

- no patch-token DriveOne fusion model;
- no patch-token model on the corrected causal pool;
- no valid Qwen direct baseline or distillation teacher;
- no PE-Spatial or four-frame temporal experiment;
- no six-camera learned model;
- no DriveLM question conversion;
- no NAVSIM/GTRS or Waymo trajectory ranking;
- no closed-loop simulator experiment;
- no end-to-end real-time measurement;
- no safety claim;
- no publication claim based on the current data audit.

The absence of these experiments is intentional. They are downstream of the
corrected pooled-baseline gate, which has now run and failed its quality
criteria.

## 18. Decision record and reading order

The current decision is:

```text
POOLED_BASELINE_GATE_FAILED
```

The data construction gate passed, but the first learned-model gate did not.
The two-seed frozen pooled-PE test used the official Le3DE2E tracker pool. On
the test split, pooled PE reached 0.0668 mAP (seed 0) and 0.0569 mAP (seed 1).
Task ID reached 0.0787 and 0.0718; metadata-only reached 0.0830 and 0.0810;
fixed front-center projected area reached 0.1964. Pooled PE also had worse
seed-0 ECE than task ID (0.1583 versus 0.0468). These values are operational
metrics: unknown candidates remain in the ranking pool.

The source candidate pool is still reproducible and label-independent. The
main limitation is visual coverage: all seven cameras project essentially all
matched positives, but a fixed front-center image projects only 46.0% of test
positive rows. The main log split also reuses prompts across logs, so the small
`joint_holdout_eligible` subset is the only prompt-holdout diagnostic.

Read the project in this order if you want to understand it without jumping
between files:

1. This guide, Sections 1–4, for the idea and vocabulary.
2. Section 5, for the staged research gates.
3. Section 6, for how raw files become candidate rows.
4. Section 8, for the job of each source file and command.
5. Section 9, for the chronological record and measured numbers.
6. Section 15, for the final audit’s matching, projection, and control details.
7. `docs/refav_pooled_baseline_gate.md`, for the latest learned-model gate and
   its decision.
8. `docs/refav_candidate_pool_decision.md`, for the short machine-audited
   decision record.
9. `configs/refav_pilot.yaml`, for pinned inputs and thresholds.
10. `src/driveone/data/refav_candidate_sources.py` and
   `scripts/audit_refav_candidate_sources.py`, when you want to inspect the
   latest implementation.

The next implementation should not add patch tokens yet. First decide whether
the task should use a fixed multi-camera representation, or whether RefAV
should be recorded as a negative branch and replaced by a fairer candidate
interface. The machine-readable report is `baseline_gate_report.json` in the
external asset directory; it stays outside Git with the data and checkpoints.

## 19. Detailed record of the corrected pooled-PE gate

This section explains the most recent experiment in one place. It is included
so that the project guide remains understandable even when the external JSON
files are not open.

### 19.1 What question did this experiment ask?

The question was narrow:

> With the repaired RefAV candidate pool fixed, does a small scorer using one
> image and frozen pooled PE image/text features rank the referred track better
> than simple controls?

This is not a test of driving policy, yielding, safety, planning, or real-time
deployment. It tests only the first learned ranking step. A negative result is
useful because it prevents a larger model from hiding a weak candidate
interface or a weak representation.

### 19.2 What data entered the run?

The run used the official RefAV Le3DE2E tracker source and the same nine logs
used by the repeated-prompt plan. The adapter read the v6 causal candidate
pool before attaching prompt labels. It then joined the official tracker
fields, AV2 pose data, and the nearest fixed `ring_front_center` image.

The final export contains 2,880 prompt/timestamp groups:

| Split | Groups | Candidate rows | Positive rows | Unknown rows |
|---|---:|---:|---:|---:|
| Train | 960 | 270,190 | 454 | 236,600 |
| Validation | 960 | 209,640 | 335 | 184,960 |
| Test | 960 | 200,210 | 361 | 181,670 |

The same timestamp pool is repeated for several prompts. Therefore the
prompt-expanded JSONL contains repeated `(log_id, timestamp_ns, track_id)`
rows by design. The source pool before prompt expansion has no duplicate keys.
The source pool hash is checked before labels are attached, and the hash stays
unchanged in the exported manifest.

The candidate rows keep `None` for unknown labels. Unknown rows are not
silently changed to `OTHER_OBJECT`. They remain in the ranking list, where
they can push a known positive down, but they are omitted from the BCE loss.

### 19.3 What code performed the conversion?

The main files and their jobs are:

- `src/driveone/data/refav_baseline_export.py` reads the v6 source rows,
  checks the candidate and label array lengths, checks the pool hash, converts
  city-frame centers to ego-frame centers, attaches the fixed camera image,
  and shuffles candidates with a stable per-group seed.
- `scripts/export_refav_baseline.py` is the command-line entry point. It
  validates that every log belongs to the planned split, writes Feather and
  JSONL exports, and records the source hashes and adapter time.
- `scripts/train_refav_baselines.py` loads the exported rows and frozen PE
  tensors, trains four small scorers, and reports operational and
  labeled-only ranking metrics.
- `src/driveone/eval/bootstrap.py` computes paired bootstrap intervals over
  the same groups.
- `scripts/compare_refav_baseline_seeds.py` compares pooled PE against task ID,
  candidate-only, and metadata-only controls.
- `scripts/report_refav_baseline_gate.py` combines the data checks, control
  results, learned results, timing, hashes, and decision into the external
  `baseline_gate_report.json`.

### 19.4 What were the model inputs?

All methods received the same candidate rows and candidate order. Candidate
geometry contains normalized projected-box coordinates when the object is in
the front-center view, a visibility flag, distance, object size, and ego-frame
translation. A separate category embedding represents the raw tracker
category.

The four learned controls were:

1. **Candidate-only:** candidate geometry and category, with no image or
   question.
2. **Metadata-only:** tracker score, raw tracker label, distance, and category.
3. **Task ID:** pooled PE image features plus a learned embedding for the
   prompt/task ID.
4. **Pooled PE:** pooled PE image features plus PE text features for the full
   question.

The PE weights were frozen. The small head used linear projections to 64
dimensions, a category embedding, GELU, layer normalization, and a shared
two-layer scoring head. Training used binary cross-entropy only on candidates
with known positive or negative labels. The ranking metric still used the
complete candidate pool.

### 19.5 What did the deterministic controls show?

These controls do not learn an image representation. They test whether the
candidate pool already contains a useful shortcut.

| Test control | mAP | Recall@1 |
|---|---:|---:|
| Random | 0.0335 | 0.0090 |
| Distance | 0.0309 | 0.0000 |
| Tracker score | 0.1625 | 0.0315 |
| Category frequency | 0.0940 | 0.0360 |
| Fixed front-center projected area | 0.1964 | 0.1712 |
| Oracle label ranking | 1.0000 | 1.0000 |

Projected area is a strong control. This means that the referred-object label
is correlated with how large the projected box is in the selected camera. A
model that wins only because it learns this geometry would not support the
intended language-grounded claim.

### 19.6 What did the learned models show?

The main metrics below are **operational**: unknown candidates remain in the
pool. Labeled-only mAP is included only as a diagnostic.

| Method | Seed | mAP | Recall@1 | Labeled-only mAP | ECE |
|---|---:|---:|---:|---:|---:|
| Candidate-only | 0 | 0.0780 | 0.0405 | 0.2321 | 0.0559 |
| Metadata-only | 0 | 0.0830 | 0.0270 | 0.3242 | 0.0402 |
| Task ID + pooled image | 0 | 0.0787 | 0.0495 | 0.2737 | 0.0468 |
| Question + pooled PE | 0 | 0.0668 | 0.0360 | 0.2277 | 0.1583 |
| Candidate-only | 1 | 0.0710 | 0.0315 | 0.2500 | 0.0405 |
| Metadata-only | 1 | 0.0810 | 0.0405 | 0.2631 | 0.0282 |
| Task ID + pooled image | 1 | 0.0718 | 0.0315 | 0.2190 | 0.2182 |
| Question + pooled PE | 1 | 0.0569 | 0.0135 | 0.2197 | 0.1048 |

Pooled PE is below task ID on both seeds. It is also below metadata-only and
far below projected-area ranking. The seed-0 pooled-PE ECE is 0.1583, compared
with 0.0468 for task ID, so the accuracy result is not accompanied by better
calibration.

The required preregistered margin was at least +0.03 mAP or +0.05 Recall@1
over the strongest matched baseline. Pooled PE instead differs from task ID
by −0.0119 mAP and −0.0135 Recall@1 on seed 0, and by −0.0149 mAP and
−0.0180 Recall@1 on seed 1.

The paired test-set bootstrap intervals for pooled PE minus task ID were:

| Metric | Seed | Mean difference | 95% interval |
|---|---:|---:|---:|
| Operational mAP | 0 | −0.0119 | [−0.0339, 0.0089] |
| Operational mAP | 1 | −0.0149 | [−0.0245, −0.0056] |
| Operational Recall@1 | 0 | −0.0135 | [−0.0405, 0.0135] |
| Operational Recall@1 | 1 | −0.0180 | [−0.0405, 0.0000] |

These intervals do not support the required positive improvement.

### 19.7 What does camera coverage tell us?

The association audit and the fixed-view baseline measure different things.
Conditional on an eligible ground-truth event, the 2 m association gives
98.99% positive availability. Across every timestamp group, availability is
only 23.8%, because many sampled timestamps contain no referred event. This
is an event-sampling statistic, not a direct tracker-recall estimate.

The seven-camera audit projects 100% of matched positives into at least one
available camera. The baseline deliberately uses only `ring_front_center` so
that the comparison has one fixed image. In that view, only 46.0% of test
positive rows have a projected box. A positive can still be in the candidate
pool when it is outside the front-center view, but the image cannot provide
direct visual evidence for it.

This is why adding patch tokens to the same one-camera setup would be a weak
next step. The representation cannot recover visual evidence from a camera
that does not see the object.

### 19.8 What did the timing measure?

The adapter took 120.1 seconds to create the prompt-expanded export. The
missing PE cache extraction took 157.4 seconds on CPU for 86 images and 33
prompts. The baseline script measured JSONL loading, image-file validation,
cached feature loading, tensor construction, training, and cached-feature
scoring.

For seed 0, JSONL reading took 82.1 seconds, input construction 15.6 seconds,
and the pooled-PE scorer trained for 5.7 seconds and scored the three splits in
3.2 seconds. These are research-run timings, not deployment latency. They do
not include image decoding, PE image encoding for new frames, candidate
construction, calibration, or postprocessing.

### 19.9 Final interpretation and next action

The data protocol is usable for a controlled diagnostic: candidates are
created before labels are attached, unknowns are explicit, hashes are stable,
and log splits are disjoint. The first learned representation test is not
successful. Pooled PE does not add useful ranking quality over simpler
controls, and the fixed camera has limited positive visibility.

The correct next action is a design choice, not a larger model. Either define
one fixed multi-camera candidate representation and rerun the pooled baseline
with every control matched to it, or record RefAV as a negative branch and
choose a task with a fairer candidate interface. Patch tokens, Qwen,
distillation, temporal frames, trajectories, and deployment optimization stay
deferred until that choice is resolved.
