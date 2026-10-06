# Code Walkthrough: repeated-prompt log selection

## Why this exists

The first six logs were useful for checking the pipeline, but they do not give
enough repeated prompt coverage for a language-generalization test. This tool
chooses a small, explicit log plan before we download more sensor data.

## What the script reads

`select_refav_log_plan.py` reads only `log_id` and `prompt` from the full RefAV
validation annotation Feather file. It records the source SHA-256 and does not
read tracker scores or labels when choosing logs.

## How selection works

`build_prompt_log_sets` creates a map from each exact prompt string to the logs
where it appears. It does not normalize wording, replace numbers, or call
paraphrases equivalent. This conservative rule avoids claiming a template
holdout that has not been measured.

`select_disjoint_log_triplets` counts how many exact prompts are shared by each
three-log combination. It greedily takes the highest-scoring combinations that
do not reuse a log. `make_three_way_plan` assigns the sorted member of each
triplet to train, validation, and test. Therefore the three splits are log
disjoint, while each selected prompt is observed in all three logs of its
triplet.

## Current plan

The external plan selects nine logs as three triplets. Each triplet has two
exact prompt strings shared across its three logs. This is a split plan, not a
benchmark result: camera coverage, candidate coverage, and shortcut controls
still have to be checked after the assets are downloaded.

## Run it

```bash
conda run -n refav python scripts/select_refav_log_plan.py \
  --annotations /data/sadegh/driveone/data/refav/source/scenario_mining_val_annotations.feather \
  --output /ehsan/m.sadegh/driveone_assets/refav/refav_repeated_prompt_log_plan.json \
  --triplets 3 --min-shared-prompts 2
```

Read this after [[Code Walkthrough - refav_tracker]] and before downloading
the next sensor subset.
