import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.eval.refav_diagnosis import diagnose_split, decide_report, group_rows


def row(track_id, label, area, score, projection_status="PROJECTED", log_id="log-a", timestamp=1):
    return {
        "log_id": log_id,
        "prompt": "p",
        "timestamp_ns": timestamp,
        "track_id": track_id,
        "label": label,
        "projected_box": [0, 0, area, area] if area else None,
        "projection_status": projection_status,
        "score": score,
        "distance_m": 5.0,
        "match_distance_m": 0.5 if label is not None else None,
        "size": [4.0, 2.0, 1.5],
        "translation_m": [5.0, 0.0, 0.0],
        "name": "REGULAR_VEHICLE",
        "match_status": "MATCHED_ANNOTATED" if label is not None else "UNMATCHED_TRACK",
    }


class RefAVDiagnosisTests(unittest.TestCase):
    def test_group_rows_uses_fixed_ranking_key(self):
        groups = group_rows([row(1, 0, 10, 0.5), row(2, 2, 8, 0.4)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))), 2)

    def test_feature_summary_reports_visibility_and_unknowns(self):
        rows = [
            row(1, 0, 20, 0.9),
            row(2, 2, 5, 0.2, projection_status="OUT_OF_VIEW"),
            row(3, None, 0, 0.1, projection_status="OUT_OF_VIEW"),
        ]
        report = diagnose_split(rows, {"models": {}}, {"results": {}, "hard_negative_results": {}, "score_and_size_matched_results": {}}, "test")
        features = report["label_and_feature_distributions"]
        self.assertEqual(features["UNKNOWN"]["row_count"], 1)
        self.assertEqual(features["REFERRED_OBJECT"]["projection_status"]["PROJECTED"], 1)
        self.assertEqual(features["comparison"]["referred_visible_rate"], 1.0)
        self.assertEqual(report["dataset"]["oracle_candidate_coverage"], 1.0)

    def test_decision_requires_repair_when_controls_or_visibility_are_strong(self):
        rows = [
            row(1, 0, 20, 0.9),
            row(2, 2, 5, 0.2, projection_status="OUT_OF_VIEW"),
        ]
        controls = {
            "results": {
                "projected_box_area:seed_0": {"mean_average_precision": 0.8},
                "random:seed_0": {"mean_average_precision": 0.5},
                "oracle:seed_0": {"mean_average_precision": 1.0},
            },
            "hard_negative_results": {"group_count": 1, "results": {}},
            "score_and_size_matched_results": {"group_count": 1, "results": {}},
            "dataset": {},
        }
        report = diagnose_split(
            rows,
            {"models": {"pooled_pe": {"metrics": {"test": {"mAP": 0.5}}}}},
            controls,
            "test",
        )
        self.assertNotIn("oracle", report["shortcut_flags"]["full_pool_control_dominance"]["control_mAP"])
        decision = decide_report({"test": report}, minimum_rankable_groups=1)
        self.assertEqual(decision["code"], "PROTOCOL_REPAIR_REQUIRED")


if __name__ == "__main__":
    unittest.main()
