import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.data.refav_candidate_sources import (  # noqa: E402
    associate,
    choose_decision,
    count_bin,
    coverage,
    pool_hash,
    validate_pool,
    validate_splits,
)


class CandidateSourceAuditTests(unittest.TestCase):
    def _row(self, track_id, x, name="REGULAR_VEHICLE"):
        return {
            "log_id": "log-a", "timestamp_ns": 10, "track_id": str(track_id), "name": name,
            "translation_m": [x, 0.0, 0.0], "size": [4.0, 2.0, 1.5],
            "rotation": np.eye(3).tolist(), "score": 0.5,
        }

    def test_pool_hash_ignores_prompt_and_target_fields(self):
        rows = [self._row(1, 0.0), self._row(2, 2.0)]
        changed = [dict(row, prompt="query", label="REFERRED_OBJECT", match_status="MATCHED_ANNOTATED") for row in rows[::-1]]
        self.assertEqual(pool_hash(rows), pool_hash(changed))
        self.assertEqual(validate_pool(rows)["duplicate_keys"], [])

    def test_matching_maximizes_valid_edges_and_records_ambiguity(self):
        candidates = [self._row(1, 0.0), self._row(2, 0.1)]
        gt = [dict(self._row("gt", 0.05), track_id="gt")]
        matches = associate(candidates, gt, threshold_m=2.0)
        self.assertEqual(len(matches), 1)
        self.assertTrue(next(iter(matches.values()))["ambiguous"])

    def test_coverage_separates_ego_event_frequency_from_external_availability(self):
        rows = [
            {"gt_positive_count": 1, "gt_external_positive_count": 0, "eligible_positive_count": 0,
             "eligible_external_positive_count": 0, "matched_positive_count": 0, "matched_external_positive_count": 0,
             "projected_positive_count": 0, "projected_external_positive_count": 0,
             "projected_external_positive_count_all_seven_assets": 0, "matched_external_positive_count_all_seven_assets": 0,
             "positive_count_all_seven_assets": 0, "ego_positive_count": 1, "unknown_count": 0, "candidate_count": 1},
            {"gt_positive_count": 1, "gt_external_positive_count": 1, "eligible_positive_count": 1,
             "eligible_external_positive_count": 1, "matched_positive_count": 1, "matched_external_positive_count": 1,
             "projected_positive_count": 1, "projected_external_positive_count": 1,
             "projected_external_positive_count_all_seven_assets": 1, "matched_external_positive_count_all_seven_assets": 1,
             "positive_count_all_seven_assets": 1, "ego_positive_count": 0, "unknown_count": 0, "candidate_count": 1},
        ]
        result = coverage(rows)
        self.assertEqual(result["gt_positive_group_count"], 2)
        self.assertEqual(result["gt_external_positive_group_count"], 1)
        self.assertEqual(result["positive_availability_given_eligible_gt_positive"], 1.0)
        self.assertEqual(result["ego_positive_group_count"], 1)

    def test_fixed_split_and_decision_rules(self):
        with self.assertRaises(ValueError):
            validate_splits({"train": ["a"], "test": ["a"]})
        self.assertEqual(count_bin(0), "0")
        self.assertEqual(count_bin(51), "51-100")
        self.assertEqual(choose_decision(infrastructure_complete=True, deployable_pass=True, oracle_pass=True)[0], "OFFICIAL_PROTOCOL_REPAIRED")
        self.assertEqual(choose_decision(infrastructure_complete=False, deployable_pass=False, oracle_pass=None)[0], "ALTERNATE_CANDIDATE_SOURCE_REQUIRED")


if __name__ == "__main__":
    unittest.main()
