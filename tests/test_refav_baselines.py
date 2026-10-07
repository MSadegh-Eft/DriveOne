import unittest

import torch

from scripts.train_refav_baselines import CandidateScorer, average_precision, box_features


class RefAVBaselineHelperTests(unittest.TestCase):
    def test_box_features_have_fixed_shape_and_normalized_coordinates(self):
        row = {
            "projected_box": [0.0, 0.0, 960.0, 600.0],
            "translation_m": [10.0, 0.0, 5.0],
            "size": [2.0, 4.0, 1.0],
            "distance_m": 25.0,
        }
        values = box_features(row, image_width=1920.0, image_height=1200.0)
        self.assertEqual(len(values), 12)
        self.assertEqual(values[:4], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(values[4], 1.0)

    def test_average_precision_keeps_unknown_candidates_as_ranked_distractors(self):
        # The unknown row is ranked first. It is not a negative label, but it
        # still pushes the referred candidate down in the full candidate pool.
        labels = [None, 0, 1]
        scores = [0.99, 0.80, 0.10]
        self.assertAlmostEqual(average_precision(labels, scores), 0.5)

    def test_task_id_and_text_models_accept_the_same_pooled_image_input(self):
        candidate = torch.zeros(2, 12)
        category = torch.zeros(2, dtype=torch.long)
        prompt_id = torch.zeros(2, dtype=torch.long)
        image = torch.zeros(2, 1024)
        text = torch.zeros(2, 1024)
        task_id = CandidateScorer(12, 2, "task_id", 2)
        pooled_pe = CandidateScorer(12, 2, "pooled_pe", 2)
        self.assertEqual(task_id(candidate, category, prompt_id, image, text).shape, (2,))
        self.assertEqual(pooled_pe(candidate, category, prompt_id, image, text).shape, (2,))


if __name__ == "__main__":
    unittest.main()
