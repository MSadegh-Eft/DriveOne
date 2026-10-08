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

## 2026-10-08 — Keep RefAV as oracle-only

**Decision:** Do not train DriveOne on the available Le3DE2E candidate pool.
Keep RefAV as an oracle-candidate or negative diagnostic unless an independent
detector/tracker source is found.

**Evidence:** The official conversion was replayed on nine logs with all seven
camera views at the native tracker timestamps. Conservative external-object
availability was 68.6% at 2 m, with 71.7% at 1 m and 59.2% at 4 m. The
ground-truth oracle reached 100% availability and projection. Tracker-score
and distance controls reached 0.533 and 0.446 mAP on the official replay pool.

**What would reverse it:** An independently generated, label-independent pool
must reach at least 80% external referred availability, at least 90% projection,
matching-threshold change no larger than 10 points, and no dominant metadata
shortcut on the same candidates.
