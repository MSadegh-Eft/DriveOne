# DriveOne Project Guide

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

The data pipeline and PE interface work. However, simple tracker and geometry controls are much stronger than the first corrected pooled-PE baseline.

On the corrected 500-group-per-split run:

| method | test mAP |
| --- | ---: |
| pooled PE image + question text | 0.0476 |
| task ID + the same pooled image | 0.0576 |
| tracker-score ranking | 0.1813 |
| projected-box-area ranking | 0.1729 |

This is an early stop for the current model path. It does not prove that visual or language features can never help. It says that the present candidate/label setup contains a strong shortcut, and the pooled model has not shown value beyond it.

The next step is to diagnose and, only if possible, repair the candidate protocol. Patch-token fusion is deliberately paused.

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
the corrected pooled baselines are complete. The result has not passed the
shortcut/frozen-baseline gate, so patch tokens and all later stages remain
paused.

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

The public Valeo4Cast tracker file is the source of candidate objects. A frame
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

The present pipeline ends after deterministic controls and corrected frozen-PE
baselines. It has not yet implemented the patch-token fusion model. That pause
is deliberate: projected box area and tracker-related controls are stronger
than the current pooled model on the tested subset. Before adding a new model,
we need to determine whether that gap comes from label transfer, tracker
quality, shared-camera visibility, candidate geometry, or a real lack of useful
image-language signal.

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

### Tests

The `tests/` directory uses Python `unittest`.

- `test_refav_contract.py` checks schema normalization and leakage rules.
- `test_refav_tracker.py` checks matching and timestamp behavior.
- `test_refav_metrics.py` checks AP, unknown-row handling, and hard-negative controls.
- `test_refav_splits.py` checks disjoint prompt-overlap planning.
- `test_verify_script.py` checks the dependency-light verifier behavior.
- `test_refav_baselines.py` checks feature shape, unknown-row ranking, and that task ID and pooled PE accept the same image input.

The current suite has 20 passing tests in the `refav` environment. Tests use small synthetic records; they do not prove model quality.

### Documentation and policy

**`docs/initial-experiment.md`** gives the experiment order and metric rules.

**`docs/refav_data_contract.md`** records the formal data contract and the real-data audit history.

**`docs/refav_baseline_smoke.md`** records the corrected baseline protocol, exact results, artifact hashes, and reproduction commands.

**`docs/obsidian/`** contains the learning path, roadmap, current status, glossary, code walkthroughs, and decision log. Those notes are useful for detail, but this guide is the single linear explanation.

**`CONTRIBUTING.md`** states the research rules: pin inputs, exclude target fields, do not claim safety or real time without measurements, and keep large artifacts outside Git.

**`.gitignore`** keeps data, weights, caches, runs, and generated reports out of commits. Large artifacts live under the external asset root, currently `/ehsan/m.sadegh/driveone_assets/refav`.

## 9. What we have done so far

### Repository and documentation

The standalone `driveone` repository was created under `/data/sadegh/driveone`. It is separate from other projects such as `fail2drive`. The first report and proposal remain outside the code pipeline; the report is available at `/data/sadegh/driveone/reports/DriveOne_Due_Diligence_Report.pdf`.

### Data inspection

The official RefAV annotation artifact and public tracker files were inspected. The annotation file is ground-truth scenario information, not a complete candidate file. It lacks tracker score and camera association, so it cannot be used alone for this model.

### Candidate manifest

The public tracker and AV2 sensor assets were used to construct a repaired six-log manifest and then a nine-log repeated-prompt manifest. The nine-log manifest contains 3,338,670 candidate rows in 14,120 groups. It has 3,242 rankable groups. Every row has a shared camera image association. Many tracker rows are unmatched; they remain explicit candidates instead of being labeled negative.

### Split plan

The nine logs are arranged as three train/validation/test triplets. Each triplet has two exact prompt strings shared across its three logs. The split is log-disjoint. It is not a full semantic template holdout because we do not have a trusted prompt-family taxonomy.

### Shortcut controls

On the full nine-log pool, tracker score and projected box area are already strong. Score-and-size matching reduces but does not remove the geometry shortcut. Control strength differs greatly by log, so pooled headline numbers would hide important variation.

### PE interface

The official `PE-Core-L14-336` checkpoint was run on a real AV2 image on host GPU 2. It returned:

- 577 tokens including the class token;
- 576 patch tokens after removing the class token;
- token width 1024;
- pooled image width 1024;
- text feature width 1024;
- official CLIP text context length 32.

The PE image input is resized to 336×336 by the official transform. The original camera image is 1550×2048. These are different dimensions and must not be confused.

### Corrected baseline smoke test

The first baseline run had two bugs. We kept its hashes for audit but marked them invalid. The corrected run fixed both problems and added checks for source image size, duplicate candidates, shared group images, feature alignment, split overlap, and pretrained PE metadata.

The corrected 200-group smoke test and 500-group replication both failed the pass criterion: pooled PE did not beat the deterministic tracker/geometry controls. The corrected artifacts are recorded in the config and remain outside Git.

### Protocol-diagnosis gate

The CPU-only diagnosis of the 500-group artifacts is complete. It confirms that
the selected groups all contain a positive and a labeled negative, but the
unknown-row rate is very high: 84.4% in train and about 89.8% in validation and
test. Visibility is also split-dependent. Referred objects are more visible
than labeled negatives in validation and test, while the direction is reversed
in train. Deterministic controls remain much stronger than pooled PE on the
held-out splits, including the score-and-size-matched diagnostic subset.

The diagnosis is recorded in `docs/refav_protocol_diagnosis.md` and in the
external JSON artifact. Its decision is `PROTOCOL_REPAIR_REQUIRED`. This does
not reject every possible RefAV study, but it blocks patch-token modeling until
we define a label-independent candidate-pool repair.

## 10. What the current result means

The result supports these statements:

- The candidate-building pipeline can produce a reproducible table.
- The PE checkpoint and patch-token interface can be loaded.
- The current RefAV candidate pool contains strong geometry and tracker-related shortcuts.
- The current small pooled-PE scorer does not beat those controls on the tested held-out logs.
- Task ID did not show an advantage over natural-language PE text in this smoke test.

The result does not support these statements:

- Patch tokens are useless.
- Natural language never helps.
- DriveOne cannot work on any AV task.
- The model is unsafe or safe.
- The model is real time.
- The candidate generator is good enough for deployment.

The present result is a reason to inspect the data protocol, not a reason to add more model capacity.

## 11. The next step

The protocol-diagnosis gate is complete. Its decision is
**`PROTOCOL_REPAIR_REQUIRED`**.

The immediate next step is to design one candidate-pool repair that:

1. never inspects the relevance label;
2. keeps the same timestamp, image, and candidate policy for every method;
3. keeps unknown candidates explicit;
4. is deterministic and hashable; and
5. does not turn the ranking task into hand-built positive/negative pairs.

After rebuilding or re-exporting that repaired pool, rerun the deterministic
controls. Only if the controls are no longer competitive should we run a
second-seed pooled baseline and confidence-interval check. Patch tokens remain
deferred until those checks pass.

Do not start Qwen, distillation, PE-Spatial, four-frame input, six-camera scaling, DriveLM conversion, NAVSIM/GTRS trajectories, Waymo ranking, or deployment optimization before this gate is resolved.

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

At the current point, the first three questions have exposed a serious shortcut and the pooled baseline has not passed. That is a useful research result. The next decision should be about the validity of the candidate protocol, not about adding a larger model.

## 15. Official Le3DE2E repair result

The next protocol repair used the official RefAV scenario-mining annotations and
the official Le3DE2E validation tracker.  The repair code is in
`src/driveone/data/refav_repair.py` and
`scripts/repair_refav_protocol.py`.  Candidate construction happens before
matching, uses a fixed semantic taxonomy map, keeps unknown rows, and projects
each candidate into all seven ring cameras.

The nine-log output is stored outside Git at
`/ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_repaired.feather`.
It contains 333,255 unique candidate rows and 14,110 prompt/timestamp groups.
Only 2,935 groups (20.8%) have a matched referred track.  Only 33.2% of
matched referred rows project into at least one camera, and 90.5% of the
prompt-labelled rows remain unknown.  Positive availability is 18.9%, 20.8%,
and 21.8% at 1 m, 2 m, and 4 m matching thresholds.  The machine-readable assessment is
`/ehsan/m.sadegh/driveone_assets/refav/official/refav_le3de2e_assessment.json`.

The decision remains **`PROTOCOL_REPAIR_REQUIRED`**.  This is a data-interface
stop, not a model result.  Do not add patch tokens, temporal frames, Qwen,
trajectories, or distillation until a candidate pool with adequate referred
coverage and camera coverage exists.  If a label-independent repair cannot
achieve that coverage, stop the RefAV branch and record the negative finding.

## 16. Candidate-source audit result

The next audit replayed the official RefAV Le3DE2E conversion on the nine
native tracker timestamp grids. It also downloaded only the nearest frame for
each of the seven ring cameras at those timestamps. This removed the earlier
missing-camera problem from the measurement.

For external-object prompts, the official replay provides 68.6% conservative
referred-track availability at 2 m. It provides 71.7% at 1 m and 59.2% at 4
m, so the conclusion changes by 20.8 percentage points. The ground-truth
oracle reaches 100% availability and 100% camera projection. The oracle proves
that the annotations and camera geometry are usable; it is not a deployable
candidate generator.

Tracker confidence and candidate distance are strong controls on the official
replay pool, reaching 0.533 and 0.446 mAP. These are data-source diagnostics,
not DriveOne model results. Ego-vehicle prompts are reported separately from
external object tracks. Unmatched rows remain `UNMATCHED_TRACK`; the ranking
diagnostic treats them as explicit non-referred tracker false positives but
does not relabel them `OTHER_OBJECT`.

The final decision for this branch is **`REFAV_ORACLE_ONLY`**. Do not add patch
tokens, temporal frames, Qwen, distillation, trajectories, or deployment
optimization. A new independent detector/tracker source is required before
RefAV can support the central candidate-ranking claim. The detailed record is
[`docs/refav_candidate_pool_decision.md`](refav_candidate_pool_decision.md).
