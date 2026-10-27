import math
import random


def generate_price_path(
    starting_price: float,
    volatility_per_step: float,
    num_time_steps: int,
    seed: int | None = None,
) -> list[float]:
    """generates a reproducible sequence of simulated market prices"""

    if not math.isfinite(starting_price) or starting_price <= 0:
        raise ValueError("Starting price must be positive and finite.")

    if not math.isfinite(volatility_per_step) or volatility_per_step < 0:
        raise ValueError("Volatility must be non-negative and finite.")

    if num_time_steps <= 0:
        raise ValueError("Number of time steps must be positive.")

    random_generator = random.Random(seed)
    prices = [float(starting_price)]

    for _ in range(num_time_steps - 1):
        random_change = random_generator.gauss(0.0, volatility_per_step)

        # adjusts the log return so expected price stays constant
        adjusted_change = random_change - 0.5 * volatility_per_step**2

        next_price = prices[-1] * math.exp(adjusted_change)
        prices.append(next_price)

    return prices
