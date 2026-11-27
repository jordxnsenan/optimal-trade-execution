import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.historical import build_historical_price_path


class HistoricalPricePathTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        self.bars = [{"c": 100.0 + index} for index in range(78)]

    def test_builds_correct_number_of_prices(self):
        prices = build_historical_price_path(
            self.bars,
            self.config,
        )

        self.assertEqual(len(prices), 12)

    def test_uses_previously_completed_bars(self):
        prices = build_historical_price_path(
            self.bars,
            self.config,
        )

        self.assertEqual(prices[0], 105.0)
        self.assertEqual(prices[1], 106.0)
        self.assertEqual(prices[-1], 116.0)

    def test_future_close_does_not_change_current_price(self):
        original_prices = build_historical_price_path(
            self.bars,
            self.config,
        )

        # changes the close of the interval beginning at 10:00
        self.bars[6]["c"] = 999.0

        updated_prices = build_historical_price_path(
            self.bars,
            self.config,
        )

        self.assertEqual(updated_prices[0], original_prices[0])
        self.assertEqual(updated_prices[1], 999.0)

    def test_rejects_insufficient_bars(self):
        with self.assertRaisesRegex(ValueError, "Not enough historical bars"):
            build_historical_price_path(
                self.bars[:10],
                self.config,
            )

    def test_rejects_invalid_starting_index(self):
        with self.assertRaises(ValueError):
            build_historical_price_path(
                self.bars,
                self.config,
                first_decision_bar_index=0,
            )


if __name__ == "__main__":
    unittest.main()
