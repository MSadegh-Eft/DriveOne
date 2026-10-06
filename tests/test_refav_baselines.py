import unittest

from scripts.train_refav_baselines import average_precision, box_features


class RefAVBaselineHelperTests(unittest.TestCase):
    def test_box_features_have_fixed_shape_and_normalized_coordinates(self):
        row = {
            "projected_box": [0.0, 0.0, 960.0, 600.0],
            "translation_m": [10.0, 0.0, 5.0],
            "size": [2.0, 4.0, 1.0],
            "distance_m": 25.0,
        }
        values = box_features(row)
        self.assertEqual(len(values), 12)
        self.assertEqual(values[:4], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(values[4], 1.0)

    def test_average_precision_keeps_unknown_candidates_as_ranked_distractors(self):
        # The unknown row is ranked first. It is not a negative label, but it
        # still pushes the referred candidate down in the full candidate pool.
        labels = [None, 0, 1]
        scores = [0.99, 0.80, 0.10]
        self.assertAlmostEqual(average_precision(labels, scores), 0.5)


if __name__ == "__main__":
    unittest.main()
