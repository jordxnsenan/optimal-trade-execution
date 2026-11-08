import argparse
import torch

from functools import partial
from statistics import mean

from execai.agents.dqn import DQNAgent
from execai.agents.dqn_episodes import train_dqn
from execai.agents.dqn_training import DQNTrainer
from execai.agents.replay_buffer import ReplayBuffer
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.policies.baselines import immediate_action, twap_action

DEFAULT_EPISODES = 2000
DEFAULT_SEED = 42
REPLAY_CAPACITY = 5000
TRAINING_BATCH_SIZE = 32
TARGET_SYNC_INTERVAL = 100
RECENT_EPISODES = 100

BENCHMARK_PRICE = 100.0
KNOWN_OPTIMAL_COST_BPS = 13.5


def run_execution_episode(environment, prices, policy):
    """evaluates a policy without training or exploration"""

    observation = environment.reset(prices)
    actions = []

    while observation is not None:
        action = policy(observation)
        actions.append(action)

        observation, _, _ = environment.step(action)

    return actions, environment.implementation_shortfall_bps


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN on a small deterministic execution problem."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    arguments = parser.parse_args()

    if arguments.episodes <= 0:
        parser.error("Number of episodes must be positive.")

    torch.set_num_threads(1)
    torch.manual_seed(arguments.seed)

    config = ExecutionConfig(
        total_shares=400,
        num_time_steps=3,
        num_lots=4,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=5.0,
        temporary_impact_bps=20.0,
        fee_per_share=0.01,
    )

    prices = [BENCHMARK_PRICE] * config.num_time_steps

    agent = DQNAgent(config, seed=arguments.seed)

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        seed=arguments.seed,
    )

    trainer = DQNTrainer(
        agent,
        replay_buffer,
        learning_rate=0.001,
        discount_factor=1.0,
        batch_size=TRAINING_BATCH_SIZE,
        target_sync_interval=TARGET_SYNC_INTERVAL,
    )

    training_environment = ExecutionEnvironment(config, cost_config)

    training_paths = [prices.copy() for _ in range(arguments.episodes)]

    episode_costs = train_dqn(
        agent,
        trainer,
        training_environment,
        training_paths,
        initial_exploration_rate=1.0,
        final_exploration_rate=0.05,
    )

    recent_count = min(RECENT_EPISODES, len(episode_costs))

    print("DQN Deterministic Benchmark")
    print("---------------------------")
    print(f"Training episodes: {arguments.episodes}")
    print(f"Random seed: {arguments.seed}")
    print(f"Gradient updates: {trainer.update_count}")

    print(
        f"First {recent_count} episodes average: "
        f"{mean(episode_costs[:recent_count]):.4f} bps"
    )

    print(
        f"Last {recent_count} episodes average: "
        f"{mean(episode_costs[-recent_count:]):.4f} bps"
    )

    policies = {
        "Immediate": immediate_action,
        "TWAP": partial(twap_action, config=config),
        "DQN Greedy": partial(
            agent.select_action,
            exploration_rate=0.0,
        ),
    }

    print("\nEvaluation without exploration:")

    evaluation_environment = ExecutionEnvironment(config, cost_config)

    for policy_name, policy in policies.items():
        schedule, cost = run_execution_episode(
            evaluation_environment,
            prices,
            policy,
        )

        print(f"{policy_name}: {cost:.4f} bps " f"| Actions: {schedule}")

        if policy_name == "DQN Greedy":
            print(f"Gap to known optimum: " f"{cost - KNOWN_OPTIMAL_COST_BPS:+.4f} bps")


if __name__ == "__main__":
    main()
