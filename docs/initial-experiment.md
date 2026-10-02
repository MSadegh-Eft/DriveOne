# Initial experiment protocol

The initial experiment evaluates language-grounded ranking of independently supplied RefAV tracks. It does not claim yielding behavior, trajectory planning, safety, or closed-loop performance.

Required comparisons:

1. Candidate-only baseline
2. Metadata-only baseline
3. Learned task-ID baseline
4. Question-conditioned pooled PE baseline
5. Question-conditioned patch-token scorer
6. Random/frequency ranking
7. Oracle ranking

Required controls include log-disjoint splits, held-out query templates, matched candidate sets, hard negatives, two random seeds, and end-to-end latency measurement.
