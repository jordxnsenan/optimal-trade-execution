import unittest

from execai.evaluation.statistics import (
    paired_bootstrap_confidence_interval,
)


class PairedBootstrapTests(unittest.TestCase):

    def test_constant_paired_difference(self):
        first_costs = [10.0, 20.0, 30.0, 40.0]
        second_costs = [8.0, 18.0, 28.0, 38.0]

        lower, upper = paired_bootstrap_confidence_interval(
            first_costs,
            second_costs,
        )

        self.assertAlmostEqual(lower, 2.0)
        self.assertAlmostEqual(upper, 2.0)

    def test_identical_policies_have_zero_difference(self):
        costs = [5.0, 8.0, 12.0, 7.0]

        lower, upper = paired_bootstrap_confidence_interval(
            costs,
            costs,
        )

        self.assertAlmostEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.0)

    def test_bootstrap_is_reproducible(self):
        first_costs = [5.0, 9.0, 12.0, 7.0]
        second_costs = [6.0, 8.0, 10.0, 8.0]

        first_interval = paired_bootstrap_confidence_interval(
            first_costs,
            second_costs,
            seed=123,
        )

        second_interval = paired_bootstrap_confidence_interval(
            first_costs,
            second_costs,
            seed=123,
        )

        self.assertEqual(first_interval, second_interval)

    def test_interval_uses_paired_observations(self):
        first_costs = [101.0, 201.0, 301.0]
        second_costs = [100.0, 200.0, 300.0]

        lower, upper = paired_bootstrap_confidence_interval(
            first_costs,
            second_costs,
        )

        # every matched pair differs by exactly one basis point
        self.assertAlmostEqual(lower, 1.0)
        self.assertAlmostEqual(upper, 1.0)

    def test_rejects_unmatched_observations(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_confidence_interval(
                [5.0, 6.0],
                [5.0],
            )

        with self.assertRaises(ValueError):
            paired_bootstrap_confidence_interval(
                [5.0],
                [4.0],
            )

    def test_rejects_invalid_confidence_level(self):
        for confidence_level in (0.0, 1.0, float("nan"), True):
            with self.subTest(confidence_level=confidence_level):
                with self.assertRaises(ValueError):
                    paired_bootstrap_confidence_interval(
                        [5.0, 6.0],
                        [4.0, 5.0],
                        confidence_level=confidence_level,
                    )

    def test_rejects_invalid_sample_count(self):
        for num_samples in (0, 1, True):
            with self.subTest(num_samples=num_samples):
                with self.assertRaises(ValueError):
                    paired_bootstrap_confidence_interval(
                        [5.0, 6.0],
                        [4.0, 5.0],
                        num_samples=num_samples,
                    )

    def test_rejects_nonfinite_costs(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_confidence_interval(
                [5.0, float("inf")],
                [4.0, 5.0],
            )


if __name__ == "__main__":
    unittest.main()
