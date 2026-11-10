import argparse
import json
import torch

from functools import partial
from pathlib import Path
from statistics import mean
from typing import Callable

from execai.agents.dqn import DQNAgent
from execai.agents.dqn_episodes import train_dqn
from execai.agents.dqn_scenarios import (
    generate_liquidity_profiles,
    generate_training_paths,
)
from execai.agents.dqn_training import DQNTrainer
from execai.agents.replay_buffer import ReplayBuffer
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.liquidity import build_relative_liquidity_profile
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.statistics import paired_bootstrap_confidence_interval
from execai.policies.almgren_chriss import optimize_almgren_chriss_schedule
from execai.policies.baselines import immediate_action, twap_action
from execai.policies.schedules import scheduled_action
from execai.policies.vwap import build_vwap_schedule, vwap_action

from scripts.compare_dqn_synthetic import evaluate_policies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = PROJECT_ROOT / "configs" / "spy_training_calibration.json"

DEFAULT_EPISODES = 2000
DEFAULT_EVALUATION_EPISODES = 250
DEFAULT_SEED = 42

REPLAY_CAPACITY = 5000
TRAINING_BATCH_SIZE = 32
TARGET_SYNC_INTERVAL = 100
RECENT_EPISODES = 100

STARTING_PRICE = 100.0
HALF_SPREAD_BPS = 5.0
TEMPORARY_IMPACT_BPS = 20.0
FEE_PER_SHARE = 0.01
AC_RISK_SENSITIVE = 0.05

TRAINING_LIQUIDITY_SEED_OFFSET = 50_000_000
EVALUATION_PRICE_SEED = 10_000_000
EVALUATION_LIQUIDITY_SEED = 20_000_000


def generate_market_scenarios(
    config: ExecutionConfig,
    volatility: float,
    expected_liquidity: list[float],
    num_episodes: int,
    price_seed: int,
    liquidity_seed: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """generates matching price paths and liquidity profiles"""

    price_paths = generate_training_paths(
        config,
        volatility,
        num_episodes,
        price_seed,
    )

    liquidity_profiles = generate_liquidity_profiles(
        config,
        expected_liquidity,
        num_episodes,
        liquidity_seed,
    )

    return price_paths, liquidity_profiles


def average_actions_by_step(
    environment: ExecutionEnvironment,
    price_paths: list[list[float]],
    liquidity_profiles: list[list[float]],
    policy: Callable[[dict], int],
) -> list[float]:
    """calculates the average number of lots sold at each decision"""

    if not price_paths or len(price_paths) != len(liquidity_profiles):
        raise ValueError("Price paths and liquidity profiles must match.")

    action_totals = [0] * environment.config.num_time_steps

    for prices, liquidity_profile in zip(price_paths, liquidity_profiles):
        observation = environment.reset(
            prices,
            liquidity_profile=liquidity_profile,
        )

        while observation is not None:
            time_step = observation["time_step"]
            action = policy(observation)

            action_totals[time_step] += action

            observation, _, _ = environment.step(action)

    return [total / len(price_paths) for total in action_totals]


def train_agent(
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    price_paths: list[list[float]],
    liquidity_profiles: list[list[float]],
    seed: int,
    use_liquidity: bool,
) -> tuple[DQNAgent, list[float], int]:
    """trains one DQN configuration on the supplied market scenarios"""

    torch.manual_seed(seed)

    agent = DQNAgent(
        config,
        seed=seed,
        use_liquidity=use_liquidity,
    )

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        seed=seed,
    )

    trainer = DQNTrainer(
        agent,
        replay_buffer,
        batch_size=TRAINING_BATCH_SIZE,
        target_sync_interval=TARGET_SYNC_INTERVAL,
    )

    environment = ExecutionEnvironment(config, cost_config)

    episode_costs = train_dqn(
        agent,
        trainer,
        environment,
        price_paths,
        initial_exploration_rate=1.0,
        final_exploration_rate=0.05,
        liquidity_profiles=liquidity_profiles,
    )

    return agent, episode_costs, trainer.update_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare basic and liquidity-aware DQN execution."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
    )

    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=DEFAULT_EVALUATION_EPISODES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    arguments = parser.parse_args()

    if arguments.episodes <= 0 or arguments.evaluation_episodes < 2:
        parser.error(
            "Training episodes must be positive and evaluation needs at least two."
        )

    if arguments.seed < 0:
        parser.error("Random seed must be non-negative.")

    torch.set_num_threads(1)

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

    volatility = calibration["volatility_per_step"]

    config = ExecutionConfig(
        total_shares=10_000,
        num_time_steps=12,
        num_lots=20,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=HALF_SPREAD_BPS,
        temporary_impact_bps=TEMPORARY_IMPACT_BPS,
        fee_per_share=FEE_PER_SHARE,
    )

    expected_liquidity = build_relative_liquidity_profile(
        calibration["volume_fractions"],
        config,
    )

    seed_block_size = arguments.episodes + arguments.evaluation_episodes
    training_price_seed = arguments.seed * seed_block_size
    training_liquidity_seed = TRAINING_LIQUIDITY_SEED_OFFSET + training_price_seed

    training_paths, training_liquidity = generate_market_scenarios(
        config,
        volatility,
        expected_liquidity,
        arguments.episodes,
        training_price_seed,
        training_liquidity_seed,
    )

    evaluation_paths, evaluation_liquidity = generate_market_scenarios(
        config,
        volatility,
        expected_liquidity,
        arguments.evaluation_episodes,
        EVALUATION_PRICE_SEED,
        EVALUATION_LIQUIDITY_SEED,
    )

    agents = {}

    for use_liquidity in (False, True):
        name = "Liquidity-Aware DQN" if use_liquidity else "Basic DQN"

        agent, episode_costs, update_count = train_agent(
            config,
            cost_config,
            training_paths,
            training_liquidity,
            arguments.seed,
            use_liquidity,
        )

        agents[name] = agent

        recent_count = min(RECENT_EPISODES, len(episode_costs))

        print(f"\n{name} Training")
        print(f"Gradient updates: {update_count}")
        print(
            f"First {recent_count} episodes: "
            f"{mean(episode_costs[:recent_count]):.4f} bps"
        )
        print(
            f"Last {recent_count} episodes: "
            f"{mean(episode_costs[-recent_count:]):.4f} bps"
        )

    vwap_schedule = build_vwap_schedule(
        config,
        calibration["volume_fractions"],
    )

    risk_neutral_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=STARTING_PRICE,
        volatility_per_step=volatility,
        risk_aversion=0.0,
        liquidity_profile=expected_liquidity,
    )

    risk_sensitive_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=STARTING_PRICE,
        volatility_per_step=volatility,
        risk_aversion=AC_RISK_SENSITIVE,
        liquidity_profile=expected_liquidity,
    )

    policies = {
        "Immediate": immediate_action,
        "TWAP": partial(twap_action, config=config),
        "VWAP": partial(vwap_action, schedule=vwap_schedule),
        "AC Risk-Neutral": partial(
            scheduled_action,
            schedule=risk_neutral_schedule,
        ),
        "AC Risk-Sensitive": partial(
            scheduled_action,
            schedule=risk_sensitive_schedule,
        ),
        "Basic DQN": partial(
            agents["Basic DQN"].select_action,
            exploration_rate=0.0,
        ),
        "Liquidity-Aware DQN": partial(
            agents["Liquidity-Aware DQN"].select_action,
            exploration_rate=0.0,
        ),
    }

    results = evaluate_policies(
        ExecutionEnvironment(config, cost_config),
        evaluation_paths,
        policies,
        liquidity_profiles=evaluation_liquidity,
    )

    print("\nLiquidity-Aware DQN Comparison")
    print("------------------------------")
    print(f"Training episodes per agent: {arguments.episodes}")
    print(f"Evaluation episodes: {arguments.evaluation_episodes}")
    print(f"Training seed: {arguments.seed}")
    print(f"Evaluation price seed: {EVALUATION_PRICE_SEED}")
    print(f"Evaluation liquidity seed: {EVALUATION_LIQUIDITY_SEED}")
    print(f"Calibrated volatility: {volatility:.8f}")

    print("\nMean implementation shortfall (lower is better):")

    for policy_name, costs in results.items():
        print(f"{policy_name}: {mean(costs):.4f} bps")

    comparisons = [
        ("Basic DQN", "TWAP"),
        ("Liquidity-Aware DQN", "TWAP"),
        ("Liquidity-Aware DQN", "Basic DQN"),
    ]

    print("\nPaired mean differences:")

    for first_policy, second_policy in comparisons:
        differences = [
            first_cost - second_cost
            for first_cost, second_cost in zip(
                results[first_policy],
                results[second_policy],
            )
        ]

        lower_bound, upper_bound = paired_bootstrap_confidence_interval(
            results[first_policy],
            results[second_policy],
        )

        print(
            f"{first_policy} minus {second_policy}: "
            f"{mean(differences):+.4f} bps "
            f"| 95% CI [{lower_bound:+.4f}, {upper_bound:+.4f}]"
        )

    print("\nAverage lots sold per time step:")

    diagnostic_environment = ExecutionEnvironment(config, cost_config)

    diagnostic_policies = (
        "TWAP",
        "Basic DQN",
        "Liquidity-Aware DQN",
    )

    for policy_name in diagnostic_policies:
        average_actions = average_actions_by_step(
            diagnostic_environment,
            evaluation_paths,
            evaluation_liquidity,
            policies[policy_name],
        )

        formatted_actions = ", ".join(f"{action:.2f}" for action in average_actions)

        print(f"{policy_name}: [{formatted_actions}]")


if __name__ == "__main__":
    main()
