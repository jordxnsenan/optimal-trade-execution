import math

from collections.abc import Callable
from execai.agents.dqn import DQNAgent
from execai.agents.dqn_training import DQNTrainer
from execai.environment.simulator import ExecutionEnvironment

DEFAULT_VALIDATION_INTERVAL = 200


def train_dqn(
    agent: DQNAgent,
    trainer: DQNTrainer,
    environment: ExecutionEnvironment,
    price_paths: list[list[float]],
    initial_exploration_rate: float = 1.0,
    final_exploration_rate: float = 0.1,
    liquidity_profiles: list[list[float]] | None = None,
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL,
    validation_callback: Callable[[DQNAgent, int], None] | None = None,
) -> list[float]:
    """trains DQN through complete execution episodes"""

    if not price_paths:
        raise ValueError("At least one price path is required.")

    if liquidity_profiles is not None and len(liquidity_profiles) != len(price_paths):
        raise ValueError(
            "Each training price path must have a matching liquidity profile."
        )

    if agent.config != environment.config:
        raise ValueError("Agent and environment configurations must match.")

    if trainer.agent is not agent:
        raise ValueError("Trainer must belong to the supplied agent.")

    exploration_rates = (
        initial_exploration_rate,
        final_exploration_rate,
    )

    if any(
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        for rate in exploration_rates
    ):
        raise ValueError("Exploration rates must be finite numbers.")

    if not 0 <= final_exploration_rate <= initial_exploration_rate <= 1:
        raise ValueError("Exploration rates must satisfy 0 <= final <= initial <= 1.")

    if type(validation_interval) is not int or validation_interval <= 0:
        raise ValueError("Validation interval must be positive.")

    episode_costs = []
    num_episodes = len(price_paths)

    for episode_index, prices in enumerate(price_paths):

        if num_episodes == 1:
            exploration_rate = initial_exploration_rate
        else:
            training_progress = episode_index / (num_episodes - 1)

            exploration_rate = (
                initial_exploration_rate
                + (final_exploration_rate - initial_exploration_rate)
                * training_progress
            )

        episode_liquidity = None

        if liquidity_profiles is not None:
            episode_liquidity = liquidity_profiles[episode_index]

        observation = environment.reset(
            prices,
            liquidity_profile=episode_liquidity,
        )

        while observation is not None:
            action = agent.select_action(
                observation,
                exploration_rate=exploration_rate,
            )

            next_observation, reward, is_done = environment.step(action)

            trainer.replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=is_done,
            )

            trainer.train_step()

            observation = next_observation

        episode_costs.append(environment.implementation_shortfall_bps)

        completed_episodes = episode_index + 1

        if validation_callback is not None and (
            completed_episodes % validation_interval == 0
            or completed_episodes == num_episodes
        ):
            validation_callback(agent, completed_episodes)

    return episode_costs
