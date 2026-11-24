import unittest

from scripts.compare_historical import compare_baselines


class HistoricalComparisonTests(unittest.TestCase):

    def test_baselines_complete_order_on_flat_market(self):
        bars = [{"c": 100.0} for _ in range(78)]

        results = compare_baselines(bars)

        immediate = results["Immediate Execution"]
        twap = results["TWAP"]

        self.assertEqual(
            immediate["execution_schedule"],
            [10_000],
        )

        self.assertEqual(
            twap["execution_schedule"],
            [1_000] * 8 + [500] * 4,
        )

        self.assertAlmostEqual(
            immediate["implementation_shortfall_bps"],
            26.0,
        )

        self.assertAlmostEqual(
            twap["implementation_shortfall_bps"],
            7.8,
        )

    def test_future_price_change_does_not_affect_immediate_execution(self):
        bars = [{"c": 100.0} for _ in range(78)]

        original_results = compare_baselines(bars)

        # changes the price of the bar beginning at 10:00 AM
        bars[6]["c"] = 110.0

        updated_results = compare_baselines(bars)

        self.assertAlmostEqual(
            original_results["Immediate Execution"]["implementation_shortfall_bps"],
            updated_results["Immediate Execution"]["implementation_shortfall_bps"],
        )

        self.assertNotAlmostEqual(
            original_results["TWAP"]["implementation_shortfall_bps"],
            updated_results["TWAP"]["implementation_shortfall_bps"],
        )


if __name__ == "__main__":
    unittest.main()
