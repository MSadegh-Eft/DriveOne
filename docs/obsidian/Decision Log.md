# Decision Log

## 2026-10-03 — Start with a feasibility gate

**Decision:** Build a RefAV referred-track ranking gate before implementing the full DriveOne scorer, temporal variants, trajectory work, or distillation.

**Reason:** A compact experiment can invalidate the central candidate interface before substantial GPU spending.

## 2026-10-03 — Keep official annotations separate from candidates

**Decision:** Treat the downloaded scenario-mining Feather as ground-truth annotation only.

**Reason:** It contains relevance labels and geometry but no tracker confidence, camera association, image path, or 2D crop. Calling it a tracker candidate file would make the result uninterpretable.

## 2026-10-03 — Preserve multiple positives

**Decision:** Use a multi-positive ranking contract and keep Recall@1 secondary.

**Reason:** RefAV prompts can refer to multiple objects and multiple timestamps. A single winner label would discard the official task semantics.

## 2026-10-03 — Defer PE and distillation claims

**Decision:** Add a PE interface smoke test but do not claim patch-token usability until the official environment and checkpoint are run. Defer Qwen and distillation.

**Reason:** Interface shape, text length, and patch-token exposure must be measured, and the non-distilled baseline must exist before teacher comparisons are meaningful.

## 2026-10-07 — Supersede the first learned-baseline numbers

**Decision:** Use only the corrected frozen-PE baseline artifacts for decisions.

**Reason:** The first run normalized projected boxes with the wrong source image
dimensions and compared pooled PE against a task-ID model that did not receive
the same pooled image input. The corrected 200- and 500-group runs use the
1550×2048 front-center image size and give task ID and pooled PE the same image
input. The corrected pooled-PE result remains below tracker-score and
projected-area controls, so patch tokens stay deferred.

## How to use this log

Add future entries with date, decision, evidence, and what would cause the decision to be reversed. This keeps the research process auditable as the project grows.
