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
    matching_range,
    partition_external_control_rows,
    pool_hash,
    validate_pool,
    validate_splits,
)
from driveone.eval.refav_metrics import run_control_suite  # noqa: E402


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

    def test_global_assignment_can_be_unique_with_local_alternatives(self):
        candidates = [self._row(1, 0.0), self._row(2, 1.9)]
        gt = [dict(self._row("gt-a", 0.1), track_id="gt-a"), dict(self._row("gt-b", 3.0), track_id="gt-b")]
        matches = associate(candidates, gt, threshold_m=2.0)
        self.assertEqual({item["gt_index"] for item in matches.values()}, {0, 1})
        self.assertFalse(any(item["ambiguous"] for item in matches.values()))
        self.assertTrue(any(item["multiple_valid_edges"] for item in matches.values()))

    def test_unknown_labels_are_preserved_and_reported_as_two_metric_bounds(self):
        rows = [
            dict(self._row("positive", 1.0), prompt="q", label=0, score=0.5),
            dict(self._row("negative", 2.0), prompt="q", label=1, score=0.4),
            dict(self._row("unknown", 3.0), prompt="q", label=None, score=0.9),
        ]
        result = run_control_suite(rows, seeds=(0,))
        self.assertEqual(result["dataset"]["unmatched_record_count"], 1)
        tracker = result["results"]["tracker_score:seed_0"]
        self.assertLess(tracker["mean_average_precision"], tracker["labeled_only_mean_average_precision"])
        self.assertEqual(tracker["labeled_only_mean_average_precision"], 1.0)

    def test_ego_rows_are_partitioned_before_external_controls(self):
        external, ego = partition_external_control_rows([
            dict(self._row("external", 1.0)),
            dict(self._row("ego", 0.0), name="EGO_VEHICLE", synthetic_ego=True),
        ])
        self.assertEqual([r["track_id"] for r in external], ["external"])
        self.assertEqual([r["track_id"] for r in ego], ["ego"])

    def test_threshold_range_is_reported_in_fraction_units(self):
        self.assertAlmostEqual(matching_range([0.7167630058, 0.6864161850, 0.5924855491]), 0.1242774567, places=9)

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
        self.assertEqual(choose_decision(infrastructure_complete=True, data_gate_pass=True, oracle_pass=True)[0], "POOLED_BASELINE_GATE_READY")
        self.assertEqual(choose_decision(infrastructure_complete=True, data_gate_pass=False, oracle_pass=True, association_uncertain=True)[0], "ASSOCIATION_REVIEW_REQUIRED")
        self.assertEqual(choose_decision(infrastructure_complete=False, data_gate_pass=False, oracle_pass=None)[0], "ALTERNATE_CANDIDATE_SOURCE_REQUIRED")


if __name__ == "__main__":
    unittest.main()
