import argparse

from functools import partial
from pathlib import Path
from statistics import mean

from execai.agents.checkpoints import load_dqn_checkpoint
from execai.evaluation.statistics import paired_bootstrap_confidence_interval

from scripts.evaluate_historical import (
    CHECKPOINT_DIRECTORY,
    build_evaluation_settings,
    evaluate_validation_baselines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "results" / "models"

DEFAULT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved DQN policies on historical SPY data."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    arguments = parser.parse_args()

    config, _ = build_evaluation_settings()

    basic_path = MODEL_DIRECTORY / f"dqn_basic_seed_{arguments.seed}.pt"

    liquidity_path = MODEL_DIRECTORY / f"dqn_liquidity_aware_seed_{arguments.seed}.pt"

    basic_agent, basic_metadata = load_dqn_checkpoint(basic_path)
    liquidity_agent, liquidity_metadata = load_dqn_checkpoint(liquidity_path)

    if basic_agent.config != config or liquidity_agent.config != config:
        raise ValueError("Saved DQN execution settings do not match.")

    if basic_agent.use_liquidity or not liquidity_agent.use_liquidity:
        raise ValueError("Saved DQN feature configurations do not match.")

    additional_policies = {
        "Basic DQN": partial(
            basic_agent.select_action,
            exploration_rate=0.0,
        ),
        "Liquidity-Aware DQN": partial(
            liquidity_agent.select_action,
            exploration_rate=0.0,
        ),
    }

    results = evaluate_validation_baselines(
        CHECKPOINT_DIRECTORY,
        liquidity_aware=True,
        additional_policies=additional_policies,
    )

    print("Historical DQN Validation")
    print("-------------------------")
    print("Dataset: 2025 SPY validation sessions")
    print("Liquidity: Training-derived forecast")
    print("Impact: Linear, 20 bps coefficient")
    print(f"Training seed: {arguments.seed}")
    print(f"Trading sessions: {len(results['TWAP'])}")

    print(f"Basic DQN checkpoint: " f"episode {basic_metadata['best_episode']}")

    print(
        f"Liquidity-aware DQN checkpoint: "
        f"episode {liquidity_metadata['best_episode']}"
    )

    print("\nMean implementation shortfall:")

    for policy_name, daily_costs in results.items():
        print(f"{policy_name}: {mean(daily_costs.values()):.4f} bps")

    trading_dates = sorted(results["TWAP"])

    comparisons = [
        ("Basic DQN", "TWAP"),
        ("Liquidity-Aware DQN", "TWAP"),
        ("Liquidity-Aware DQN", "Basic DQN"),
    ]

    print("\nPaired mean differences:")

    for first_policy, second_policy in comparisons:
        if set(results[first_policy]) != set(trading_dates) or set(
            results[second_policy]
        ) != set(trading_dates):
            raise ValueError("Policy evaluation dates must match.")

        first_costs = [
            results[first_policy][trading_date] for trading_date in trading_dates
        ]

        second_costs = [
            results[second_policy][trading_date] for trading_date in trading_dates
        ]

        differences = [
            first_cost - second_cost
            for first_cost, second_cost in zip(
                first_costs,
                second_costs,
            )
        ]

        lower_bound, upper_bound = paired_bootstrap_confidence_interval(
            first_costs,
            second_costs,
        )

        print(
            f"{first_policy} minus {second_policy}: "
            f"{mean(differences):+.4f} bps "
            f"| 95% CI [{lower_bound:+.4f}, {upper_bound:+.4f}]"
        )


if __name__ == "__main__":
    main()
