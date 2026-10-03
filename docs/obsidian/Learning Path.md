# Learning Path

## First session: understand the goal

Read [[Project Overview]], then [[RefAV Data Contract]]. Explain in your own words why the same row can contain a target label for evaluation but must not expose that label to the model.

## Second session: trace a command

Read [[Architecture and Data Flow]] and [[Code Walkthrough - verify_refav]]. Run the verifier on an empty directory and inspect the JSON. Then run it on a small trusted export.

## Third session: understand normalization

Read [[Code Walkthrough - refav_contract]]. Focus on `_flatten_refav_pickle`, the array-length check, `_group_key`, and `inspect_records`. Use the unit tests as executable examples.

## Fourth session: understand the model boundary

Read [[Code Walkthrough - smoke_test_pe]]. Identify exactly what it measures and what it leaves untested. This prevents “the backbone loaded” from being confused with “the proposed model works.”

## Fifth session: understand the stop decision

Read [[Current Status and Next Steps]] and [[Decision Log]]. The project is currently waiting on a reproducible tracker candidate artifact. This is the correct place to solve the next problem.

## Later expansion

When the gate passes, add notes for candidate construction, crop projection, baseline implementations, metric definitions, and model training. Keep each note linked to the code and record decisions in [[Decision Log]].
