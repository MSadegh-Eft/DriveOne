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

### Stage 0 — Understand the contract

Read the proposal and define exactly what one example means. Decide whether the first task is entity ranking, question answering, or trajectory ranking. We chose RefAV referred-track ranking because it is the smallest test with public natural-language interaction descriptions and object tracks.

**Output:** a written definition of one group and its labels.

### Stage 1 — Prepare data

Use the official RefAV annotation file, public tracker output, Argoverse 2 poses, camera calibration, camera images, and map data. Convert tracker positions from the city frame into the ego-vehicle frame. Match tracker objects to RefAV annotated objects using same-category one-to-one assignment within a 2 m ego-frame threshold.

**Output:** a hashed Feather candidate manifest.

### Stage 2 — Audit the data

Check required columns, array lengths, duplicate candidate keys, labels, timestamps, camera association, projections, candidate coverage, and log-disjoint splits. Record all source URLs, repository commits, file hashes, licenses, and local paths.

**Output:** a machine-readable audit report and a list of limitations.

### Stage 3 — Check the PE interface

Load the official `PE-Core-L14-336` checkpoint. Run its image and text paths. Inspect the tensor shapes and timing. The current real-image test found 576 patch tokens of width 1024, pooled image features of width 1024, text features of width 1024, and an official CLIP text context of 32 tokens.

**Output:** a shape/configuration report. This does not prove that patch tokens improve ranking.

### Stage 4 — Run deterministic controls

Before training a multimodal model, rank candidates using random order, tracker score, distance, projected box area, category frequency, and an oracle. These controls reveal whether the dataset or candidate generator already makes the answer easy.

**Output:** control metrics, candidate counts, and hard-negative results.

### Stage 5 — Audit shortcuts

Match negatives to positives by tracker score and then by tracker score plus projected box size. Keep the exact candidate set fixed across methods. Inspect results by log and candidate count.

**Output:** a defensible shortcut test or a reason to redesign the candidate pool.

### Stage 6 — Frozen baseline gate

Run candidate-only, metadata-only, task-ID, and pooled-PE models on fixed subsets. Use log-disjoint splits and one seed for the first smoke test. Add another seed only after the first run is reproducible.

**Output:** a corrected baseline report with ranking and calibration metrics.

### Stage 7 — Minimal DriveOne model

Only if the pooled gate is meaningful, add frozen PE patch tokens, one question encoder, one small fusion block, one candidate projection, and one shared scorer. Use one frame. Do not add PE-Spatial, temporal input, Qwen, or trajectories at this point.

**Move on when:** the patch model beats pooled PE and task ID by a predeclared margin on held-out logs and wording, without a serious calibration regression.

### Stage 8 — Stress the result

Test candidate-count shift, held-out wording, paraphrases, log-disjoint scenes, city-disjoint scenes, and independently generated candidates. Report per-task results and failure cases.

### Stage 9 — Expand the task

Add PE-Spatial-L14-448, four frames over about 1.5 seconds, more cameras, DriveLM bounded answer tasks, NAVSIM/GTRS trajectories, and Waymo preference ranking as separate tracks. Each extension needs matched baselines.

### Stage 10 — Distill last

Only after the non-distilled student is understood should Qwen or another teacher be evaluated. A teacher cannot repair an invalid task or hide a candidate shortcut.

## 6. The current data pipeline

The pipeline has two parts: building the candidate table and evaluating models.

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

### Source annotations

The RefAV annotation file contains prompt-specific object relevance labels and object geometry. It does not contain the tracker confidence, camera image path, or complete candidate interface needed by DriveOne.

### Tracker predictions

The public Valeo4Cast tracker file supplies candidate objects, timestamps, positions, sizes, classes, and tracker scores. These objects are potential candidates, not ground truth.

### Coordinate conversion

Tracker positions are in the city frame. Camera projection and RefAV matching use the ego-vehicle frame. The adapter uses the AV2 pose at the timestamp to transform positions and compute ego-relative distance and yaw.

### Matching

For each object class, the adapter computes distances between tracker centers and annotation centers. Hungarian assignment makes the matching one-to-one. A match is accepted only within 2 m. This transfers the prompt label to the tracker candidate without allowing one candidate to match several annotations.

### Camera association

Each group uses one shared `ring_front_center` image chosen near the decision timestamp. Every candidate in that group points to the same image. Candidates outside the image view are retained with `OUT_OF_VIEW` status, because removing them would make the pool depend on visual visibility.

### ROI and range filtering

The adapter applies the official-style maximum range and map ROI checks available from the downloaded AV2 assets. These are infrastructure filters. They are not relevance labels.

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

The next step is a **protocol-diagnosis gate**.

1. Read the per-log and candidate-count metrics stored in the corrected 500-group result.
2. Check whether tracker geometry is acting as a proxy for how RefAV labels were transferred.
3. Check whether visual appearance is redundant because the candidate pool already identifies the referred object through location, size, or visibility.
4. Design a label-independent candidate-pool repair only if one is technically credible. Examples could include independently generated hard negatives, score-and-size matching, or a candidate protocol that equalizes visibility and geometry without inspecting labels.
5. Rerun deterministic controls after the repair.
6. Add patch tokens only if the repaired pooled baseline is reproducible and beats the strongest deterministic control by a predeclared margin.

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

