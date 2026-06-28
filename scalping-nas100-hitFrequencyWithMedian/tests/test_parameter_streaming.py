import unittest

from hfmed_core import parameters


class ParameterStreamingTests(unittest.TestCase):
    def test_streaming_stage1_count_matches_current_grid(self):
        grid = parameters.load_grid("parameter_grid.ini")

        self.assertEqual(parameters.stage1_candidate_count(grid, 0), 746_496)
        self.assertEqual(parameters.stage1_candidate_count(grid, 262_144), 262_144)
        self.assertEqual(parameters.stage1_candidate_count(grid, 8_388_608), 746_496)

    def test_streaming_stage1_batches_are_valid_and_unique_for_sample(self):
        grid = parameters.load_grid("parameter_grid.ini")
        seen = set()
        count = 0

        for batch in parameters.iter_stage1_candidate_batches(grid, 10_000, seed=12345, batch_size=777):
            self.assertEqual(batch.start_index, count)
            for values in batch.candidates:
                self.assertTrue(parameters.is_valid(values))
                digest = parameters.parameter_hash(values)
                self.assertNotIn(digest, seen)
                seen.add(digest)
                count += 1

        self.assertEqual(count, 10_000)


if __name__ == "__main__":
    unittest.main()
