from collections.abc import Callable
from execai.environment.simulator import ExecutionEnvironment


def run_policy(
    environment: ExecutionEnvironment,
    prices: list[float],
    choose_action: Callable[[dict], int],
) -> float:
    """runs a policy through one complete execution episode"""

    observation = environment.reset(prices)

    while observation is not None:
        action = choose_action(observation)
        observation, _, _ = environment.step(action)

    return environment.implementation_shortfall_bps