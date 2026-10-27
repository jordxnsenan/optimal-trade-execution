import unittest

from execai.environment.market import generate_price_path


class MarketTests(unittest.TestCase):

    def test_correct_number_of_prices(self):
        prices = generate_price_path(100.0, 0.002, 12, seed=42)

        self.assertEqual(len(prices), 12)

    def test_first_price_matches_starting_price(self):
        prices = generate_price_path(100.0, 0.002, 12, seed=42)

        self.assertEqual(prices[0], 100.0)

    def test_same_seed_produces_same_prices(self):
        first_path = generate_price_path(100.0, 0.002, 12, seed=42)
        second_path = generate_price_path(100.0, 0.002, 12, seed=42)

        self.assertEqual(first_path, second_path)

    def test_different_seeds_produce_different_prices(self):
        first_path = generate_price_path(100.0, 0.002, 12, seed=42)
        second_path = generate_price_path(100.0, 0.002, 12, seed=99)

        self.assertNotEqual(first_path, second_path)

    def test_zero_volatility_produces_constant_prices(self):
        prices = generate_price_path(100.0, 0.0, 12, seed=42)

        self.assertEqual(prices, [100.0] * 12)

    def test_prices_remain_positive(self):
        prices = generate_price_path(100.0, 0.002, 12, seed=42)

        for price in prices:
            self.assertGreater(price, 0)

    def test_invalid_starting_price(self):
        with self.assertRaises(ValueError):
            generate_price_path(0.0, 0.002, 12)

    def test_invalid_volatility(self):
        with self.assertRaises(ValueError):
            generate_price_path(100.0, -0.002, 12)

    def test_invalid_time_steps(self):
        with self.assertRaises(ValueError):
            generate_price_path(100.0, 0.002, 0)


if __name__ == "__main__":
    unittest.main()