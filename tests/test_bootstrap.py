import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driveone.eval.bootstrap import paired_bootstrap, paired_metric_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], samples=100, seed=4)
        second = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], samples=100, seed=4)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_difference"], 1.0)

    def test_metric_bootstrap_joins_by_group_key_and_skips_missing(self):
        left = [
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "mAP": 0.8},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 2, "mAP": None},
        ]
        right = [
            {"log_id": "a", "prompt": "p", "timestamp_ns": 1, "mAP": 0.5},
            {"log_id": "a", "prompt": "p", "timestamp_ns": 2, "mAP": 0.2},
        ]
        result = paired_metric_bootstrap(left, right, metric="mAP", samples=10, seed=0)
        self.assertEqual(result["group_count"], 1)
        self.assertEqual(result["shared_group_count"], 2)
        self.assertAlmostEqual(result["mean_difference"], 0.3)


if __name__ == "__main__":
    unittest.main()
