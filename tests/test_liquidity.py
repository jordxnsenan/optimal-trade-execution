import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.liquidity import (
    build_relative_liquidity_profile,
)


class RelativeLiquidityTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        self.volume_fractions = [1.0] * 78

    def test_uniform_volume_produces_reference_liquidity(self):
        liquidity = build_relative_liquidity_profile(
            self.volume_fractions,
            self.config,
        )

        self.assertEqual(liquidity, [1.0] * 12)

    def test_relative_liquidity_has_mean_one(self):
        for index in range(6, 18):
            self.volume_fractions[index] = index + 1

        liquidity = build_relative_liquidity_profile(
            self.volume_fractions,
            self.config,
        )

        self.assertAlmostEqual(
            sum(liquidity) / len(liquidity),
            1.0,
        )

    def test_higher_volume_produces_higher_liquidity(self):
        self.volume_fractions[6] = 2.0

        liquidity = build_relative_liquidity_profile(
            self.volume_fractions,
            self.config,
        )

        self.assertGreater(liquidity[0], 1.0)
        self.assertLess(liquidity[1], 1.0)

    def test_volume_outside_window_does_not_affect_liquidity(self):
        original_liquidity = build_relative_liquidity_profile(
            self.volume_fractions,
            self.config,
        )

        self.volume_fractions[77] = 100.0

        updated_liquidity = build_relative_liquidity_profile(
            self.volume_fractions,
            self.config,
        )

        self.assertEqual(original_liquidity, updated_liquidity)

    def test_rejects_missing_execution_window(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            build_relative_liquidity_profile(
                self.volume_fractions[:10],
                self.config,
            )

    def test_rejects_zero_volume_in_execution_window(self):
        self.volume_fractions[6] = 0.0

        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_relative_liquidity_profile(
                self.volume_fractions,
                self.config,
            )

    def test_rejects_invalid_forecast_volume(self):
        invalid_volumes = [
            -1.0,
            float("nan"),
            float("inf"),
            True,
        ]

        for invalid_volume in invalid_volumes:
            with self.subTest(volume=invalid_volume):
                self.volume_fractions[6] = invalid_volume

                with self.assertRaises(ValueError):
                    build_relative_liquidity_profile(
                        self.volume_fractions,
                        self.config,
                    )
