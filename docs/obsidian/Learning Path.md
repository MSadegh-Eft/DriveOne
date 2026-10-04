# Learning Path

This is a short study plan. You do not need to read every line before running a command.

## Session 1 — the question (20–30 minutes)

Read [[Project Overview]] and [[Glossary]]. Goal: explain why the first task is *referred-track ranking*, not “driving” or “grounding” in the broad sense.

## Session 2 — the data contract (30–45 minutes)

Read [[RefAV Data Contract]] and [[Code Walkthrough - refav_tracker]]. Goal: explain the difference between a tracker candidate, a ground-truth label, a hard negative, and an unmatched track.

## Session 3 — follow the audit (30–45 minutes)

Read [[Architecture and Data Flow]], then [[Code Walkthrough - refav_contract]] and [[Code Walkthrough - verify_refav]]. Goal: trace one file from disk to the JSON audit. Run the unit tests while reading.

## Session 4 — check the visual backbone (20–30 minutes)

Read [[Code Walkthrough - smoke_test_pe]]. Goal: know the tensor shapes the script measured, and know why those shapes do not prove a useful model.

## Session 5 — reproduce before extending (45–60 minutes)

Read [[Tests and Validation]] and [[Current Status and Next Steps]]. Re-run the manifest audit and the controls when they are implemented. Goal: decide whether the candidate interface is trustworthy.

## Session 6 — only after the gate (later)

Read the future experiment notes added for candidate-only controls, pooled PE, patch PE, and the small fusion scorer. Do not study Qwen, temporal input, trajectories, or distillation until the gate in [[Development Roadmap]] passes.
