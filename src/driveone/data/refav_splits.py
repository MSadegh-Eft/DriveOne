"""Prompt-overlap based log-selection helpers for RefAV audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable, Mapping


def build_prompt_log_sets(rows: Iterable[Mapping[str, object]]) -> dict[str, tuple[str, ...]]:
    """Return the exact prompt-to-log relation, with duplicate rows removed."""

    prompt_logs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        log_id = str(row.get("log_id") or "").strip()
        if prompt and log_id:
            prompt_logs[prompt].add(log_id)
    return {prompt: tuple(sorted(logs)) for prompt, logs in sorted(prompt_logs.items())}


def select_disjoint_log_triplets(
    prompt_logs: Mapping[str, Iterable[str]],
    triplet_count: int = 3,
    min_shared_prompts: int = 2,
) -> list[dict[str, object]]:
    """Select disjoint log triplets sharing exact prompt strings.

    Each returned triplet can supply one log to train, validation, and test.
    The selection is greedy after sorting by the number of prompts common to
    all three logs, then by the stable log IDs. This is a split-planning aid,
    not a claim that the resulting logs are otherwise matched.
    """

    if triplet_count < 1:
        raise ValueError("triplet_count must be positive")
    if min_shared_prompts < 1:
        raise ValueError("min_shared_prompts must be positive")

    triple_prompts: Counter[tuple[str, str, str]] = Counter()
    for logs_iter in prompt_logs.values():
        logs = sorted(set(str(log) for log in logs_iter if str(log)))
        for triple in combinations(logs, 3):
            triple_prompts[triple] += 1

    ranked = sorted(triple_prompts.items(), key=lambda item: (-item[1], item[0]))
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for logs, shared_count in ranked:
        if shared_count < min_shared_prompts or any(log in used for log in logs):
            continue
        common_prompts = sorted(
            prompt
            for prompt, prompt_logs_for_prompt in prompt_logs.items()
            if set(logs).issubset(set(prompt_logs_for_prompt))
        )
        selected.append({"logs": list(logs), "shared_prompt_count": shared_count, "shared_prompts": common_prompts})
        used.update(logs)
        if len(selected) == triplet_count:
            break
    return selected


def make_three_way_plan(triplets: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Assign each selected triplet's sorted logs to train/validation/test."""

    split_logs: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    normalized_triplets = []
    for item in triplets:
        logs = sorted(str(log) for log in item["logs"])
        if len(logs) != 3 or len(set(logs)) != 3:
            raise ValueError("each triplet must contain three distinct logs")
        split_logs["train"].append(logs[0])
        split_logs["validation"].append(logs[1])
        split_logs["test"].append(logs[2])
        normalized_triplets.append(dict(item, logs=logs))
    all_logs = [log for split in split_logs.values() for log in split]
    if len(all_logs) != len(set(all_logs)):
        raise ValueError("split logs must be disjoint")
    return {"triplets": normalized_triplets, "splits": split_logs, "all_selected_logs": sorted(all_logs)}
