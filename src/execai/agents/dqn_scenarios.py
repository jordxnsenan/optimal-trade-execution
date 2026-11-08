from execai.environment.config import ExecutionConfig
from execai.environment.market import generate_price_path

DEFAULT_STARTING_PRICE = 100.0


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
