import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.data.refav_contract import (
    inspect_records,
    load_records,
    template_key,
    validate_candidate_feature_schema,
    validate_log_disjoint,
)


class RefAVContractTests(unittest.TestCase):
    def make_payload(self):
        return {
            ("log-a", "pedestrian crossing at stop sign"): [
                {
                    "timestamp_ns": 100,
                    "track_id": [10, 11],
                    "score": [0.9, 0.8],
                    "label": [0, 2],
                    "name": ["REFERRED_OBJECT", "OTHER_OBJECT"],
                    "translation_m": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "size": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    "yaw": [0.0, 0.1],
                },
                {
                    "timestamp_ns": 200,
                    "track_id": [10, 12],
                    "score": [0.88, 0.7],
                    "label": [0, 1],
                    "name": ["REFERRED_OBJECT", "RELATED_OBJECT"],
                    "translation_m": [[1.1, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    "size": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    "yaw": [0.0, 0.2],
                },
            ],
            ("log-b", "vehicle turning left"): [
                {
                    "timestamp_ns": 300,
                    "track_id": [20, 21],
                    "score": [0.6, 0.5],
                    "label": [0, 2],
                    "name": ["REFERRED_OBJECT", "OTHER_OBJECT"],
                    "translation_m": [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                    "size": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    "yaw": [0.0, 0.4],
                }
            ],
        }

    def test_official_pickle_is_flattened_with_variable_candidate_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.pkl"
            with path.open("wb") as handle:
                pickle.dump(self.make_payload(), handle)
            rows = load_records(path)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["log_id"] for row in rows}, {"log-a", "log-b"})
        self.assertEqual(rows[0]["label"], 0)
        self.assertEqual(rows[1]["label"], 2)

    def test_malformed_array_lengths_are_rejected(self):
        payload = self.make_payload()
        payload[("log-a", "pedestrian crossing at stop sign")][0]["yaw"] = [0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            with self.assertRaises(ValueError):
                load_records(path)

    def test_inspection_reports_positive_negative_groups_and_holdout(self):
        rows = []
        for log_id, prompt, timestamp in [
            ("log-a", "pedestrian crossing at stop sign", 100),
            ("log-b", "vehicle turning left", 200),
        ]:
            rows.extend([
                {
                    "log_id": log_id,
                    "prompt": prompt,
                    "timestamp_ns": timestamp,
                    "track_id": 1,
                    "score": 0.8,
                    "label": 0,
                    "translation_m": [1, 0, 0],
                    "size": [1, 1, 1],
                    "yaw": 0.0,
                    "camera_frame": f"{log_id}-{timestamp}.jpg",
                },
                {
                    "log_id": log_id,
                    "prompt": prompt,
                    "timestamp_ns": timestamp,
                    "track_id": 2,
                    "score": 0.7,
                    "label": 2,
                    "translation_m": [2, 0, 0],
                    "size": [1, 1, 1],
                    "yaw": 0.0,
                    "camera_frame": f"{log_id}-{timestamp}.jpg",
                },
            ])
        report = inspect_records(rows)
        self.assertTrue(report["fixed_frame_ranking_feasible"])
        self.assertTrue(report["log_disjoint_feasible"])
        self.assertTrue(report["normalized_prompt_holdout_feasible"])
        self.assertEqual(report["camera_association_fraction"], 1.0)
        self.assertFalse(report["missing_required_fields"])

    def test_duplicate_candidates_and_invalid_labels_are_reported(self):
        row = {
            "log_id": "log-a",
            "prompt": "x",
            "timestamp_ns": 1,
            "track_id": 3,
            "score": 0.5,
            "label": 9,
            "translation_m": [1, 0, 0],
            "size": [1, 1, 1],
            "yaw": 0.0,
        }
        report = inspect_records([row, dict(row)])
        self.assertEqual(report["duplicate_candidate_key_count"], 1)
        self.assertEqual(report["invalid_label_values"], {"9": 2})

    def test_feature_schema_redacts_target_fields(self):
        self.assertEqual(
            validate_candidate_feature_schema(["projected_box", "label", "track_id", "raw_category"]),
            ["label", "track_id"],
        )
        self.assertEqual(validate_candidate_feature_schema(["projected_box", "raw_category"]), [])

    def test_split_overlap_is_rejected(self):
        errors = validate_log_disjoint({"train": ["a", "b"], "val": ["b"], "test": ["c"]})
        self.assertEqual(len(errors), 1)
        self.assertIn("train/val", errors[0])

    def test_template_key_is_conservative(self):
        self.assertEqual(template_key("Pedestrian 17 at 12.5m"), "pedestrian <num> at <num>m")

    def test_json_records_are_supported(self):
        row = {
            "log_id": "log-a", "prompt": "x", "timestamp_ns": 1,
            "track_id": 1, "score": 0.2, "label": "REFERRED_OBJECT",
            "translation_m": [1, 0, 0], "size": [1, 1, 1], "yaw": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            self.assertEqual(load_records(path)[0]["label"], 0)


if __name__ == "__main__":
    unittest.main()
