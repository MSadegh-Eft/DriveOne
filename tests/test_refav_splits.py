import unittest

from driveone.data.refav_splits import build_prompt_log_sets, make_three_way_plan, select_disjoint_log_triplets


class RefAVSplitTests(unittest.TestCase):
    def test_selects_disjoint_prompt_overlapping_triplets(self):
        rows = []
        for prompt, logs in {
            "p1": ["a", "b", "c"],
            "p2": ["a", "b", "c"],
            "p3": ["d", "e", "f"],
            "p4": ["d", "e", "f"],
        }.items():
            rows.extend({"prompt": prompt, "log_id": log} for log in logs)
        prompt_logs = build_prompt_log_sets(rows)
        triplets = select_disjoint_log_triplets(prompt_logs, triplet_count=2, min_shared_prompts=2)
        plan = make_three_way_plan(triplets)
        self.assertEqual(len(plan["triplets"]), 2)
        self.assertEqual(set(plan["all_selected_logs"]), {"a", "b", "c", "d", "e", "f"})
        self.assertEqual(set(plan["splits"]["train"]), {"a", "d"})

    def test_rejects_malformed_triplet(self):
        with self.assertRaises(ValueError):
            make_three_way_plan([{"logs": ["a", "b"]}])


if __name__ == "__main__":
    unittest.main()
