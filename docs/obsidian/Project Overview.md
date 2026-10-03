# Project Overview

## What this repository is for

The original DriveOne proposal describes a compact, language-conditioned scorer that ranks bounded candidates. The first implementation step intentionally narrows that ambition to a diagnostic:

> Given one RefAV prompt, one timestamp, a fixed pool of tracker-produced object candidates, and the corresponding camera input, can a language-conditioned model rank the referred tracks above matched negatives?

This is a data and identifiability question. It is not yet a claim about driving actions, yielding, planning, safety, or real-time deployment.

## What is deliberately deferred

The current repository does not implement:

- a DriveOne fusion transformer;
- a learned candidate scorer;
- temporal four-frame input;
- PE-Spatial;
- six-camera fusion;
- trajectory ranking;
- DriveLM conversion;
- Qwen direct comparison or distillation;
- closed-loop planning;
- deployment optimization.

Those features are deferred because a model trained on an invalid or leaky candidate interface would produce uninterpretable results.

## Research gate

The first gate requires all of the following:

1. Candidate records are tracker outputs, not only ground-truth annotations.
2. Every candidate can be connected reproducibly to a timestamp and camera frame.
3. Each selected group has at least one referred candidate and a reproducible negative pool.
4. Labels, tracker confidence, timestamps, and IDs are kept out of learned candidate features.
5. Log-disjoint splits and a defensible prompt holdout can be constructed.
6. The frozen PE interface is measured before a fusion model is written.

The strict verifier encodes these as a stop gate. A nonzero exit is useful evidence: it tells us which prerequisite is missing.

## Why the code is small

The code is intentionally infrastructure-first. A small validator can expose a fatal data problem much more cheaply than training a multimodal model. The repository currently has a data-contract library, a verifier CLI, an optional PE smoke test, configuration, tests, and explanatory docs.

## Source of truth

Claims about the proposal come from the original proposal PDF and due-diligence report. Claims about the RefAV file format and metrics come from the [official RefAV repository](https://github.com/cainand/refav). Claims about PE configuration and APIs come from the [official Perception Models repository](https://github.com/facebookresearch/perception_models). The local code should be treated as an implementation of these contracts, not as evidence that the contracts are already satisfied.
