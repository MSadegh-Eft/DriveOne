# Glossary

| Term | Meaning in this project |
| --- | --- |
| AV2 | Argoverse 2 dataset and API family used by RefAV |
| Candidate | A tracker-produced object track eligible for ranking at one decision timestamp |
| Candidate pool | All candidates presented to the scorer for one `(log, prompt, timestamp)` group |
| Camera association | Reproducible mapping from a candidate/timestamp to an image, camera, or projected crop |
| ECE | Expected Calibration Error; a summary of confidence/reliability mismatch |
| Feather | Apache Arrow columnar file format used by the official RefAV annotation artifact |
| Hard negative | A plausible non-target candidate; here `RELATED_OBJECT` or nearby `OTHER_OBJECT` |
| HOTA | Higher Order Tracking Accuracy, an official RefAV-style tracking metric |
| Log-disjoint | No log ID appears in more than one split |
| Multi-positive ranking | A ranking group can contain more than one correct candidate |
| PE | Perception Encoder, the proposed frozen visual backbone |
| PE-Core-L14-336 | PE-Core large/14 configuration at 336-pixel input resolution |
| Prompt/template holdout | Testing wording or prompt families not used for training |
| RefAV | Referring autonomous-vehicle scenario-mining dataset/benchmark |
| Relevance label | RefAV’s `REFERRED_OBJECT`, `RELATED_OBJECT`, or `OTHER_OBJECT` annotation |
| Strict mode | Verifier mode that returns exit `2` until every configured gate passes |
| Tracker artifact | Reproducible output of an object tracker, including candidate IDs/scores/geometry |
| Oracle coverage | Fraction of target instances that are present in the candidate pool before scoring |
| Calibration | Whether a reported confidence matches the observed frequency of being correct |
| Data contract | The exact rules for what one row means, which fields are required, and which fields may enter a model |
| Ego frame | Coordinates measured relative to the vehicle; city-frame tracker positions are converted into this frame |
| Leakage | Information reaches the model that would not be available at the intended decision time, or directly reveals the answer |
| Manifest | A derived data table plus hashes and settings that identify exactly how it was made |
| Pooled feature | One vector summarizing an image; it removes the spatial grid used by patch tokens |
| Patch token | One vector for a local image patch, usually arranged as a spatial sequence |
| Provenance | Where an artifact came from, including source URL, revision, version, and hash |
| ROI | Region of interest; here the official map-based area/filter used by the preparation code |
