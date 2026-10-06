import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.eval.refav_metrics import run_control_suite


class RefAVMetricTests(unittest.TestCase):
    def test_unknown_candidates_are_retained_but_not_counted_as_labels(self):
        rows = [
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 1, "label": None, "score": 1.0, "distance_m": 3.0, "name": "CAR"},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 2, "label": 0, "score": 0.5, "distance_m": 5.0, "name": "CAR"},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 3, "label": 2, "score": 0.1, "distance_m": 7.0, "name": "PEDESTRIAN"},
        ]
        result = run_control_suite(rows, seeds=[0])
        self.assertEqual(result["dataset"]["record_count"], 3)
        self.assertEqual(result["dataset"]["unmatched_record_count"], 1)
        self.assertEqual(result["dataset"]["rankable_group_count"], 1)
        tracker = result["results"]["tracker_score:seed_0"]
        self.assertEqual(tracker["group_count"], 1)
        self.assertEqual(tracker["rankable_group_count"], 1)
        self.assertEqual(tracker["labeled_only_mean_average_precision"], 1.0)

    def test_oracle_is_perfect_on_rankable_group(self):
        rows = [
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 1, "label": 2, "score": 1.0, "distance_m": 3.0, "name": "CAR"},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 2, "label": 0, "score": 0.5, "distance_m": 5.0, "name": "CAR"},
        ]
        oracle = run_control_suite(rows, seeds=[0])["results"]["oracle:seed_0"]
        self.assertEqual(oracle["mean_average_precision"], 1.0)
        self.assertEqual(oracle["recall_at_1"], 1.0)

    def test_hard_negative_subset_is_reported(self):
        rows = [
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 1, "label": 0, "score": 0.50, "distance_m": 3.0, "name": "CAR", "projected_box": [0, 0, 10, 10]},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 2, "label": 2, "score": 0.52, "distance_m": 5.0, "name": "CAR", "projected_box": [0, 0, 11, 11]},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "track_id": 3, "label": 2, "score": 0.10, "distance_m": 7.0, "name": "PEDESTRIAN", "projected_box": [0, 0, 100, 100]},
        ]
        result = run_control_suite(rows, seeds=[0], hard_negative_delta=0.05, size_matched_log_area_delta=0.2)
        self.assertEqual(result["hard_negative_results"]["group_count"], 1)
        self.assertEqual(result["hard_negative_results"]["results"]["oracle:seed_0"]["group_count"], 1)
        self.assertEqual(result["score_and_size_matched_results"]["group_count"], 1)
        self.assertEqual(result["score_and_size_matched_results"]["results"]["oracle:seed_0"]["group_count"], 1)


if __name__ == "__main__":
    unittest.main()
