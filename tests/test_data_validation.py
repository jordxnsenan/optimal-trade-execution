import unittest

from execai.data.validation import validate_bars


class DataValidationTests(unittest.TestCase):

    def setUp(self):
        self.bars = [
            {
                "t": "2025-10-15T13:30:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 150000,
            },
            {
                "t": "2025-10-15T13:35:00Z",
                "o": 100.5,
                "h": 102.0,
                "l": 100.0,
                "c": 101.5,
                "v": 175000,
            },
        ]

    def test_valid_market_bars(self):
        validate_bars(self.bars)

    def test_empty_market_data(self):
        with self.assertRaises(ValueError):
            validate_bars([])

    def test_missing_required_field(self):
        del self.bars[0]["c"]

        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_bars(self.bars)

    def test_invalid_prices(self):
        invalid_prices = [
            0.0,
            -100.0,
            float("nan"),
            float("inf"),
            "100.0",
        ]

        for invalid_price in invalid_prices:
            with self.subTest(price=invalid_price):
                self.bars[0]["c"] = invalid_price

                with self.assertRaisesRegex(ValueError, "invalid price"):
                    validate_bars(self.bars)

    def test_inconsistent_high_price(self):
        self.bars[0]["h"] = 99.0

        with self.assertRaisesRegex(ValueError, "inconsistent OHLC"):
            validate_bars(self.bars)

    def test_inconsistent_low_price(self):
        self.bars[0]["l"] = 101.0

        with self.assertRaisesRegex(ValueError, "inconsistent OHLC"):
            validate_bars(self.bars)

    def test_invalid_volume(self):
        invalid_volumes = [-1, 1.5, True, "150000"]

        for invalid_volume in invalid_volumes:
            with self.subTest(volume=invalid_volume):
                self.bars[0]["v"] = invalid_volume

                with self.assertRaisesRegex(ValueError, "invalid trading volume"):
                    validate_bars(self.bars)

    def test_zero_volume_is_allowed(self):
        self.bars[0]["v"] = 0

        validate_bars(self.bars)

    def test_duplicate_timestamps(self):
        self.bars[1]["t"] = self.bars[0]["t"]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_bars(self.bars)

    def test_out_of_order_timestamps(self):
        self.bars.reverse()

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_bars(self.bars)

    def test_timestamp_requires_timezone(self):
        self.bars[0]["t"] = "2025-10-15T13:30:00"

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            validate_bars(self.bars)

    def test_invalid_timestamp(self):
        self.bars[0]["t"] = "not-a-date"

        with self.assertRaisesRegex(ValueError, "invalid timestamp"):
            validate_bars(self.bars)


if __name__ == "__main__":
    unittest.main()