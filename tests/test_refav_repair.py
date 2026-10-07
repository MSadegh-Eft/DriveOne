import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import numpy as np
    from driveone.data.refav_repair import (
        build_candidate_pool,
        candidate_pool_hash,
        match_candidates_compatible,
        taxonomy_group,
        validate_candidate_pool,
    )
    OPTIONAL_DEPS = True
except ImportError:  # pragma: no cover - minimal CI environments
    OPTIONAL_DEPS = False


@unittest.skipUnless(OPTIONAL_DEPS, "RefAV repair dependencies are not installed")
class RefAVRepairTests(unittest.TestCase):
    def _frame(self):
        return {
            "timestamp_ns": 100,
            "track_id": np.asarray([5, 6, 7]),
            "score": np.asarray([0.2, 0.8, 0.4]),
            "label": np.asarray([0, 0, 0]),
            "name": np.asarray(["BOX_TRUCK", "PEDESTRIAN", "REGULAR_VEHICLE"]),
            "translation_m": np.asarray([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
            "size": np.asarray([[4.0, 2.0, 1.5], [0.5, 0.5, 1.7], [4.0, 2.0, 1.5]]),
            "yaw": np.asarray([0.0, 0.0, 0.0]),
        }

    def test_pool_is_prompt_and_label_independent(self):
        rows = build_candidate_pool(
            log_id="log-a",
            frames=[self._frame()],
            decision_timestamps=[100],
            ego_positions_by_timestamp={100: self._frame()["translation_m"]},
            roi_masks_by_timestamp={100: [True, True, True]},
        )
        self.assertEqual([row["track_id"] for row in rows], [5, 6])
        self.assertTrue(all("prompt" not in row for row in rows))
        self.assertTrue(all(row["label"] is None for row in rows))
        self.assertEqual(validate_candidate_pool(rows)["duplicate_candidate_keys"], [])

    def test_hash_ignores_prompt_and_target_label(self):
        rows = build_candidate_pool(
            log_id="log-a",
            frames=[self._frame()],
            decision_timestamps=[100],
            ego_positions_by_timestamp={100: self._frame()["translation_m"]},
            roi_masks_by_timestamp={100: [True, True, False]},
        )
        changed = [dict(row, prompt="query", label=0, match_status="MATCHED_ANNOTATED") for row in rows]
        self.assertEqual(candidate_pool_hash(rows), candidate_pool_hash(changed))

    def test_taxonomy_compatibility_prevents_exact_name_drop(self):
        self.assertEqual(taxonomy_group("BOX_TRUCK"), "VEHICLE")
        self.assertEqual(taxonomy_group("LARGE_VEHICLE"), "VEHICLE")
        candidates = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        matches = match_candidates_compatible(
            candidates,
            ["BOX_TRUCK", "PEDESTRIAN"],
            [
                {"track_uuid": "v0", "category": "LARGE_VEHICLE", "tx_m": 0.1, "ty_m": 0.0},
                {"track_uuid": "p0", "category": "PEDESTRIAN", "tx_m": 10.1, "ty_m": 0.0},
            ],
        )
        self.assertEqual(matches[0]["gt_track_uuid"], "v0")
        self.assertEqual(matches[1]["gt_track_uuid"], "p0")

    def test_matching_is_one_to_one(self):
        matches = match_candidates_compatible(
            np.asarray([[0.0, 0.0], [0.1, 0.0]]),
            ["PEDESTRIAN", "PEDESTRIAN"],
            [{"track_uuid": "p0", "category": "PEDESTRIAN", "tx_m": 0.05, "ty_m": 0.0}],
        )
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()

