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
