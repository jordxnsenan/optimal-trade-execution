import math
import random

from execai.environment.config import ExecutionConfig
from execai.environment.market import generate_price_path

DEFAULT_STARTING_PRICE = 100.0
DEFAULT_LIQUIDITY_NOISE_STD = 0.30


def generate_training_paths(
    config: ExecutionConfig,
    volatility_per_step: float,
    num_episodes: int,
    starting_seed: int,
) -> list[list[float]]:
    """generates reproducible market paths for DQN experiments"""

    if type(num_episodes) is not int or num_episodes <= 0:
        raise ValueError("Number of episodes must be positive.")

    price_paths = []

    for episode_index in range(num_episodes):
        prices = generate_price_path(
            starting_price=DEFAULT_STARTING_PRICE,
            volatility_per_step=volatility_per_step,
            num_time_steps=config.num_time_steps,
            seed=starting_seed + episode_index,
        )

        price_paths.append(prices)

    return price_paths


def generate_liquidity_profiles(
    config: ExecutionConfig,
    expected_profile: list[float],
    num_episodes: int,
    starting_seed: int,
    noise_std: float = DEFAULT_LIQUIDITY_NOISE_STD,
) -> list[list[float]]:
    """generates reproducible liquidity profiles for training episodes"""

    if type(num_episodes) is not int or num_episodes <= 0:
        raise ValueError("Number of episodes must be positive.")

    if type(starting_seed) is not int or starting_seed < 0:
        raise ValueError("Starting seed must be non-negative.")

    if len(expected_profile) != config.num_time_steps:
        raise ValueError("Liquidity profile length must match time steps.")

    if any(
        isinstance(liquidity, bool)
        or not isinstance(liquidity, (int, float))
        or not math.isfinite(liquidity)
        or liquidity <= 0
        for liquidity in expected_profile
    ):
        raise ValueError("Expected liquidity must be positive and finite.")

    if (
        isinstance(noise_std, bool)
        or not isinstance(noise_std, (int, float))
        or not math.isfinite(noise_std)
        or not 0 <= noise_std <= 1
    ):
        raise ValueError("Liquidity noise must be between 0 and 1.")

    profiles = []

    for episode_index in range(num_episodes):
        random_generator = random.Random(starting_seed + episode_index)
        liquidity_profile = []

        for expected_liquidity in expected_profile:
            shock = random_generator.gauss(
                -0.5 * noise_std**2,
                noise_std,
            )

            relative_liquidity = expected_liquidity * math.exp(shock)
            liquidity_profile.append(relative_liquidity)

        profiles.append(liquidity_profile)

    return profiles
