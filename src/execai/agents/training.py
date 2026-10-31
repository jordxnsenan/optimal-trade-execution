from execai.agents.q_learning import QLearningAgent
from execai.environment.simulator import ExecutionEnvironment


def train_q_learning(
    agent: QLearningAgent,
    environment: ExecutionEnvironment,
    price_paths: list[list[float]],
    initial_exploration_rate: float = 1.0,
    final_exploration_rate: float = 0.1,
    random_tie_breaking: bool = False,
) -> list[float]:
    """trains Q-learning agent and return execution costs per episode"""

    if not price_paths:
        raise ValueError("At least one price path is required.")

    if agent.config != environment.config:
        raise ValueError("Agent and environment configurations must match.")

    if not 0 <= final_exploration_rate <= initial_exploration_rate <= 1:
        raise ValueError("Exploration rates must satisfy 0 <= final <= initial <= 1.")

    episode_costs = []
    num_episodes = len(price_paths)

    for episode_index, prices in enumerate(price_paths):

        # gradually reduces exploration during training
        if num_episodes == 1:
            
            exploration_rate = initial_exploration_rate
            
        else:
            
            training_progress = episode_index / (num_episodes - 1)

            exploration_rate = initial_exploration_rate + (
                final_exploration_rate - initial_exploration_rate
            ) * training_progress

        observation = environment.reset(prices)

        while observation is not None:
            
            action = agent.select_action(
                observation,
                exploration_rate,
                random_tie_breaking=random_tie_breaking,
            )

            next_observation, reward, _ = environment.step(action)

            agent.update(
                observation,
                action,
                reward,
                next_observation,
            )

            observation = next_observation

        episode_costs.append(environment.implementation_shortfall_bps)

    return episode_costs