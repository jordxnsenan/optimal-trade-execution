import argparse
import json
import torch

from functools import partial
from pathlib import Path
from statistics import mean
from typing import Callable

from execai.agents.dqn import DQNAgent
from execai.agents.dqn_episodes import train_dqn
from execai.agents.dqn_scenarios import generate_training_paths
from execai.agents.dqn_training import DQNTrainer
from execai.agents.replay_buffer import ReplayBuffer
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.policies.almgren_chriss import optimize_almgren_chriss_schedule
from execai.policies.baselines import immediate_action, twap_action
from execai.policies.schedules import scheduled_action
from execai.policies.vwap import build_vwap_schedule, vwap_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = PROJECT_ROOT / "configs" / "spy_training_calibration.json"

DEFAULT_EPISODES = 2000
DEFAULT_EVALUATION_EPISODES = 250
DEFAULT_SEED = 42
COMMON_EVALUATION_SEED = 10_000_000

REPLAY_CAPACITY = 5000
TRAINING_BATCH_SIZE = 32
TARGET_SYNC_INTERVAL = 100
RECENT_EPISODES = 100

REFERENCE_PRICE = 100.0
AC_RISK_SENSITIVE = 0.05


def evaluate_policies(
    environment: ExecutionEnvironment,
    price_paths: list[list[float]],
    policies: dict[str, Callable[[dict], int]],
) -> dict[str, list[float]]:
    """evaluates every policy on the same price paths"""

    if not price_paths:
        raise ValueError("At least one evaluation path is required.")

    results = {policy_name: [] for policy_name in policies}

    for prices in price_paths:
        for policy_name, policy in policies.items():
            observation = environment.reset(prices)

            while observation is not None:
                action = policy(observation)
                observation, _, _ = environment.step(action)

            results[policy_name].append(environment.implementation_shortfall_bps)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DQN with execution baselines on synthetic markets."
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

    if arguments.episodes <= 0 or arguments.evaluation_episodes <= 0:
        parser.error("Episode counts must be positive.")

    if arguments.seed < 0:
        parser.error("Random seed must be non-negative.")

    torch.set_num_threads(1)
    torch.manual_seed(arguments.seed)

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

    volatility = calibration["volatility_per_step"]

    config = ExecutionConfig(
        total_shares=10_000,
        num_time_steps=12,
        num_lots=20,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=5.0,
        temporary_impact_bps=20.0,
        fee_per_share=0.01,
    )

    # gives each experiment seed its own range of market scenarios
    seed_block_size = arguments.episodes + arguments.evaluation_episodes

    training_seed = arguments.seed * seed_block_size
    evaluation_seed = COMMON_EVALUATION_SEED

    # keeps training and evaluation scenario seed ranges separate
    training_end = training_seed + arguments.episodes
    evaluation_end = evaluation_seed + arguments.evaluation_episodes

    if training_seed < evaluation_end and evaluation_seed < training_end:
        parser.error("Training and evaluation seed ranges must not overlap.")

    training_paths = generate_training_paths(
        config,
        volatility,
        arguments.episodes,
        training_seed,
    )

    evaluation_paths = generate_training_paths(
        config,
        volatility,
        arguments.evaluation_episodes,
        evaluation_seed,
    )

    agent = DQNAgent(config, seed=arguments.seed)

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        seed=arguments.seed,
    )

    trainer = DQNTrainer(
        agent,
        replay_buffer,
        batch_size=TRAINING_BATCH_SIZE,
        target_sync_interval=TARGET_SYNC_INTERVAL,
    )

    training_environment = ExecutionEnvironment(config, cost_config)

    training_costs = train_dqn(
        agent,
        trainer,
        training_environment,
        training_paths,
        initial_exploration_rate=1.0,
        final_exploration_rate=0.05,
    )

    vwap_schedule = build_vwap_schedule(
        config,
        calibration["volume_fractions"],
    )

    risk_neutral_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=REFERENCE_PRICE,
        volatility_per_step=volatility,
        risk_aversion=0.0,
    )

    risk_sensitive_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=REFERENCE_PRICE,
        volatility_per_step=volatility,
        risk_aversion=AC_RISK_SENSITIVE,
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
        "DQN": partial(agent.select_action, exploration_rate=0.0),
    }

    evaluation_environment = ExecutionEnvironment(config, cost_config)

    results = evaluate_policies(
        evaluation_environment,
        evaluation_paths,
        policies,
    )

    recent_count = min(RECENT_EPISODES, len(training_costs))

    print("DQN Stochastic Market Evaluation")
    print("--------------------------------")
    print(f"Training episodes: {arguments.episodes}")
    print(f"Evaluation episodes: {arguments.evaluation_episodes}")
    print(f"Random seed: {arguments.seed}")
    print(f"Evaluation scenario seed: {evaluation_seed}")
    print(f"Calibrated volatility: {volatility:.8f}")
    print(f"Gradient updates: {trainer.update_count}")

    print(
        f"\nFirst {recent_count} training episodes: "
        f"{mean(training_costs[:recent_count]):.4f} bps"
    )

    print(
        f"Last {recent_count} training episodes: "
        f"{mean(training_costs[-recent_count:]):.4f} bps"
    )

    print("\nMean implementation shortfall (lower is better):")

    for policy_name, costs in results.items():
        print(f"{policy_name}: {mean(costs):.4f} bps")

    paired_differences = [
        dqn_cost - twap_cost
        for dqn_cost, twap_cost in zip(results["DQN"], results["TWAP"])
    ]

    print(f"\nDQN minus TWAP: " f"{mean(paired_differences):+.4f} bps")


if __name__ == "__main__":
    main()
