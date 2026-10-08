import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.data import refav_baseline_export as export
from driveone.data.refav_candidate_sources import pool_hash


def source_row(track_id="1", timestamp=10, image_path="/data/sadegh/tmp/driveone_test_front.jpg"):
    return {
        "log_id": "log",
        "timestamp_ns": timestamp,
        "track_id": track_id,
        "name": "REGULAR_VEHICLE",
        "translation_m": [1.0, 2.0, 3.0],
        "size": [4.0, 2.0, 1.5],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "score": 0.8,
        "distance_m": 2.2,
        "synthetic_ego": False,
        "camera_projections": json.dumps([
            {"camera_name": "ring_front_left", "image_path": "/bad.jpg", "image_timestamp_ns": 9, "timestamp_delta_ns": 1, "box": [1, 1, 2, 2]},
            {"camera_name": "ring_front_center", "image_path": image_path, "image_timestamp_ns": 9, "timestamp_delta_ns": 1, "box": [3, 4, 20, 30]},
        ]),
    }


class RefAVBaselineExportTests(unittest.TestCase):
    def test_camera_selection_is_fixed_and_does_not_choose_largest(self):
        row = source_row()
        selected = export._camera_entry(row)
        self.assertEqual(selected["camera_name"], "ring_front_center")
        self.assertEqual(selected["box"], [3, 4, 20, 30])

    def test_camera_selection_rejects_frame_outside_100ms(self):
        row = source_row()
        row["camera_projections"] = json.dumps([
            {"camera_name": "ring_front_center", "image_path": "/bad.jpg", "image_timestamp_ns": 1, "timestamp_delta_ns": 100_000_001, "box": [1, 1, 2, 2]},
        ])
        with self.assertRaises(export.BaselineExportError):
            export._camera_entry(row)

    def test_source_group_hash_and_unknown_label_are_preserved(self):
        row = source_row()
        group = {
            "log_id": "log", "timestamp_ns": 10, "prompt": "a vehicle",
            "candidate_count": 1, "labels": [None], "pool_hash": pool_hash([row]),
        }
        export._validate_source_group([row], group)
        self.assertIsNone(group["labels"][0])

    def test_hash_mismatch_fails_before_label_attachment(self):
        row = source_row()
        group = {
            "log_id": "log", "timestamp_ns": 10, "prompt": "a vehicle",
            "candidate_count": 1, "labels": [0], "pool_hash": "wrong",
        }
        with self.assertRaises(export.BaselineExportError):
            export._validate_source_group([row], group)

    def test_build_rows_keeps_unknown_and_uses_ego_geometry_field(self):
        image = Path("/data/sadegh/tmp/driveone_test_front.jpg")
        image.write_bytes(b"placeholder")
        original_pose = export._pose_table
        original_transform = export._ego_translation
        try:
            export._pose_table = lambda _: type("PoseTable", (), {"index": [10], "loc": {10: {}}})()
            export._ego_translation = lambda city, pose: [9.0, 8.0, 7.0]
            row = source_row()
            group = {
                "source": "le3de2e_causal", "log_id": "log", "timestamp_ns": 10,
                "prompt": "a vehicle", "split": "train", "prompt_split": "train",
                "joint_holdout_eligible": True, "candidate_count": 1, "labels": [None],
                "pool_hash": pool_hash([row]),
            }
            output, details = export.build_baseline_rows(
                groups=[group], source_rows={"log": [row]}, sensor_root=Path("/unused"),
                tracker_index={("log", 10, "1"): {"raw_tracker_label": 0, "tracker_name": "REGULAR_VEHICLE", "tracker_score": 0.8}},
            )
            value = output["train"][0]
            self.assertIsNone(value["label"])
            self.assertEqual(value["match_status"], "UNKNOWN")
            self.assertEqual(value["translation_m"], [9.0, 8.0, 7.0])
            self.assertEqual(value["projected_box"], [3, 4, 20, 30])
            self.assertTrue(details["source_pool_hashes_unchanged"])
        finally:
            export._pose_table = original_pose
            export._ego_translation = original_transform
            image.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
