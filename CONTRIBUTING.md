# Development rules

1. Keep the first milestone non-distilled and frozen-backbone.
2. Treat RefAV as multi-object, multi-timestamp scenario mining; do not relabel it as yielding or single-winner classification.
3. Pin the RefAV/Argoverse artifact, repository commit, tracker prediction artifact, split manifest, and hashes for every run.
4. Record dataset, candidate-generation, split, seed, precision, hardware, and commit metadata for every run.
5. Remove relevance labels, names, tracker scores, timestamps, future values, and track IDs from learned candidate features unless a control explicitly studies them.
6. Do not claim grounding, safety, planner transfer, or real-time performance without the corresponding measurement protocol.
7. Keep large datasets, checkpoints, and generated outputs outside Git.
