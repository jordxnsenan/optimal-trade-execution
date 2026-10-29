from execai.environment.config import ExecutionConfig


def immediate_action(observation: dict) -> int:
    """sells all remaining inventory immediately"""

    return observation["remaining_lots"]


def twap_action(observation: dict, config: ExecutionConfig) -> int:
    """distributes the remaining inventory across the remaining time steps"""

    remaining_lots = observation["remaining_lots"]
    remaining_steps = config.num_time_steps - observation["time_step"]

    lots_to_sell = (
        remaining_lots + remaining_steps - 1
    ) // remaining_steps

    return lots_to_sell