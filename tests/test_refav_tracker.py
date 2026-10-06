import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import numpy as np
    import pandas as pd
    from driveone.data.refav_tracker import match_candidates, select_decision_timestamps
    OPTIONAL_TRACKER_DEPS = True
except ImportError:  # pragma: no cover - minimal CI environments
    OPTIONAL_TRACKER_DEPS = False


@unittest.skipUnless(OPTIONAL_TRACKER_DEPS, "optional tracker preparation dependencies are not installed")
class RefAVTrackerTests(unittest.TestCase):
    def test_timestamp_selection_does_not_require_positive_and_negative(self):
        tracker_frames = [
            {"timestamp_ns": 100, "track_id": [1]},
            {"timestamp_ns": 200, "track_id": [1]},
        ]
        pose_table = pd.DataFrame(index=[100, 200])
        camera_files = {"ring_front_center": ([100, 200], [Path("100.jpg"), Path("200.jpg")])}
        annotations = pd.DataFrame([
            {"prompt": "one target", "timestamp_ns": 100, "mining_category": "REFERRED_OBJECT"},
            {"prompt": "one target", "timestamp_ns": 200, "mining_category": "REFERRED_OBJECT"},
        ])
        selected = select_decision_timestamps(
            tracker_frames,
            annotations,
            pose_table,
            camera_files=camera_files,
        )
        self.assertEqual(selected, {"one target": [100, 200]})

    def test_same_class_assignment_uses_two_meter_threshold(self):
        candidates = np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.0]])
        names = np.array(["PEDESTRIAN", "PEDESTRIAN", "BICYCLE"])
        annotations = pd.DataFrame([
            {"track_uuid": "p0", "category": "PEDESTRIAN", "tx_m": 0.05, "ty_m": 0.0},
            {"track_uuid": "p1", "category": "PEDESTRIAN", "tx_m": 10.1, "ty_m": 0.0},
            {"track_uuid": "b0", "category": "BICYCLE", "tx_m": 3.0, "ty_m": 0.0},
        ])
        matches = match_candidates(candidates, names, annotations)
        self.assertEqual(set(matches), {0, 1})
        self.assertEqual(matches[0].gt_track_uuid, "p0")
        self.assertEqual(matches[1].gt_track_uuid, "p1")

    def test_matching_is_one_to_one(self):
        candidates = np.array([[0.0, 0.0], [0.1, 0.0]])
        names = np.array(["PEDESTRIAN", "PEDESTRIAN"])
        annotations = pd.DataFrame([
            {"track_uuid": "p0", "category": "PEDESTRIAN", "tx_m": 0.05, "ty_m": 0.0},
        ])
        matches = match_candidates(candidates, names, annotations)
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
