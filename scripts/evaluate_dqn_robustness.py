import argparse
import csv
import torch

from functools import partial
from pathlib import Path
from statistics import mean

from execai.agents.checkpoints import load_dqn_checkpoint
from execai.data.dataset import load_dataset_split
from execai.environment.liquidity import build_relative_liquidity_profile
from execai.evaluation.robustness import evaluate_frozen_scenario
from execai.evaluation.statistics import paired_bootstrap_confidence_interval
from execai.policies.almgren_chriss import optimize_almgren_chriss_schedule
from execai.policies.baselines import immediate_action, twap_action
from execai.policies.schedules import scheduled_action
from execai.policies.vwap import build_vwap_schedule, vwap_action

from scripts.evaluate_historical import (
    AC_RISK_NEUTRAL,
    AC_RISK_SENSITIVE,
    CALIBRATION_PATH,
    CHECKPOINT_DIRECTORY,
    OPTIMIZATION_REFERENCE_PRICE,
    PROJECT_ROOT,
    TEMPORARY_IMPACT_BPS,
    build_evaluation_settings,
    load_training_calibration,
)

MODEL_DIRECTORY = PROJECT_ROOT / "results" / "models"
OUTPUT_PATH = PROJECT_ROOT / "results" / "robustness" / "2025_frozen_dqn.csv"

EXISTING_TRAINING_SEEDS = (7, 42, 123, 2026)

LOW_IMPACT_BPS = 10.0
HIGH_IMPACT_BPS = 40.0
REFERENCE_LIQUIDITY_SCALE = 1.0
REDUCED_LIQUIDITY_SCALE = 0.5

ROBUSTNESS_SCENARIOS = {
    "nominal": ("linear", TEMPORARY_IMPACT_BPS, REFERENCE_LIQUIDITY_SCALE),
    "low_impact": ("linear", LOW_IMPACT_BPS, REFERENCE_LIQUIDITY_SCALE),
    "high_impact": ("linear", HIGH_IMPACT_BPS, REFERENCE_LIQUIDITY_SCALE),
    "square_root": ("square_root", TEMPORARY_IMPACT_BPS, REFERENCE_LIQUIDITY_SCALE),
    "reduced_liquidity": (
        "linear",
        TEMPORARY_IMPACT_BPS,
        REDUCED_LIQUIDITY_SCALE,
    ),
}


def build_frozen_policies(
    config,
    nominal_cost_config,
    calibration,
    forecast_liquidity,
    training_seeds,
):
    """builds nominal baselines and load selected DQN checkpoints"""

    vwap_schedule = build_vwap_schedule(
        config,
        calibration["volume_fractions"],
    )

    risk_neutral_schedule = optimize_almgren_chriss_schedule(
        config,
        nominal_cost_config,
        reference_price=OPTIMIZATION_REFERENCE_PRICE,
        volatility_per_step=calibration["volatility_per_step"],
        risk_aversion=AC_RISK_NEUTRAL,
        liquidity_profile=forecast_liquidity,
    )

    risk_sensitive_schedule = optimize_almgren_chriss_schedule(
        config,
        nominal_cost_config,
        reference_price=OPTIMIZATION_REFERENCE_PRICE,
        volatility_per_step=calibration["volatility_per_step"],
        risk_aversion=AC_RISK_SENSITIVE,
        liquidity_profile=forecast_liquidity,
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
    }

    checkpoint_metadata = {}

    for seed in training_seeds:
        model_settings = [
            ("Basic DQN", "basic", False),
            ("Liquidity-Aware DQN", "liquidity_aware", True),
        ]

        for model_name, filename, use_liquidity in model_settings:
            checkpoint_path = MODEL_DIRECTORY / f"dqn_{filename}_seed_{seed}.pt"

            agent, metadata = load_dqn_checkpoint(checkpoint_path)

            if agent.config != config:
                raise ValueError(
                    f"Checkpoint execution configuration does not match: "
                    f"{checkpoint_path}"
                )

            if agent.use_liquidity != use_liquidity:
                raise ValueError(
                    f"Checkpoint feature configuration does not match: "
                    f"{checkpoint_path}"
                )

            agent.network.eval()

            policy_name = f"{model_name} (seed {seed})"

            policies[policy_name] = partial(
                agent.select_action,
                exploration_rate=0.0,
            )

            checkpoint_metadata[policy_name] = (
                seed,
                metadata["best_episode"],
            )

    return policies, checkpoint_metadata


def run_robustness_scenarios(
    sessions,
    config,
    nominal_cost_config,
    policies,
    forecast_liquidity,
    scenarios,
):
    """evaluates the same frozen policies under each market scenario"""

    results = {}

    for scenario_name, settings in scenarios.items():
        impact_model, impact_strength, liquidity_scale = settings

        results[scenario_name] = evaluate_frozen_scenario(
            sessions,
            config,
            nominal_cost_config,
            policies,
            forecast_liquidity,
            impact_model=impact_model,
            impact_strength_bps=impact_strength,
            liquidity_scale=liquidity_scale,
        )

    return results


def save_robustness_results(
    results,
    scenarios,
    checkpoint_metadata,
    output_path: Path,
):
    """saves dated execution costs and experiment settings to CSV"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "scenario",
                "impact_model",
                "impact_strength_bps",
                "liquidity_scale",
                "trading_date",
                "policy",
                "training_seed",
                "best_episode",
                "shortfall_bps",
            ]
        )

        for scenario_name, policy_results in results.items():
            impact_model, impact_strength, liquidity_scale = scenarios[scenario_name]

            for policy_name, daily_costs in policy_results.items():
                training_seed, best_episode = checkpoint_metadata.get(
                    policy_name,
                    (None, None),
                )

                for trading_date in sorted(daily_costs):
                    writer.writerow(
                        [
                            scenario_name,
                            impact_model,
                            impact_strength,
                            liquidity_scale,
                            trading_date.isoformat(),
                            policy_name,
                            training_seed,
                            best_episode,
                            daily_costs[trading_date],
                        ]
                    )


def print_robustness_summary(results, training_seeds):
    """summarizes frozen-policy performance across robustness scenarios"""

    print("\nFrozen-Policy Historical Robustness")
    print("-----------------------------------")
    print("Dataset: 2025 SPY validation period")

    for scenario_name, policy_results in results.items():
        print(f"\nScenario: {scenario_name}")

        for baseline_name in ("TWAP", "AC Risk-Neutral"):
            average_cost = mean(policy_results[baseline_name].values())

            print(f"{baseline_name}: {average_cost:.4f} bps")

        reference_costs = policy_results["AC Risk-Neutral"]
        trading_dates = sorted(reference_costs)

        for seed in training_seeds:
            for model_name in ("Basic DQN", "Liquidity-Aware DQN"):
                policy_name = f"{model_name} (seed {seed})"
                daily_costs = policy_results[policy_name]

                if set(daily_costs) != set(reference_costs):
                    raise ValueError("Policy evaluation dates must match.")

                dqn_costs = [
                    daily_costs[trading_date] for trading_date in trading_dates
                ]

                ac_costs = [
                    reference_costs[trading_date] for trading_date in trading_dates
                ]

                differences = [
                    dqn_cost - ac_cost for dqn_cost, ac_cost in zip(dqn_costs, ac_costs)
                ]

                lower, upper = paired_bootstrap_confidence_interval(
                    dqn_costs,
                    ac_costs,
                )

                print(
                    f"{policy_name}: {mean(dqn_costs):.4f} bps "
                    f"| vs AC: {mean(differences):+.4f} "
                    f"| 95% CI [{lower:+.4f}, {upper:+.4f}]"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DQN policies under market stress."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(EXISTING_TRAINING_SEEDS),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(ROBUSTNESS_SCENARIOS),
        default=list(ROBUSTNESS_SCENARIOS),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    arguments = parser.parse_args()

    if any(seed < 0 for seed in arguments.seeds):
        parser.error("Training seeds must be non-negative.")

    if len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error("Training seeds must be unique.")

    torch.set_num_threads(1)

    config, nominal_cost_config = build_evaluation_settings()

    calibration = load_training_calibration(CALIBRATION_PATH)

    forecast_liquidity = build_relative_liquidity_profile(
        calibration["volume_fractions"],
        config,
    )

    sessions = load_dataset_split(
        CHECKPOINT_DIRECTORY,
        "validation",
    )

    policies, checkpoint_metadata = build_frozen_policies(
        config,
        nominal_cost_config,
        calibration,
        forecast_liquidity,
        arguments.seeds,
    )

    selected_scenarios = {
        name: settings
        for name, settings in ROBUSTNESS_SCENARIOS.items()
        if name in arguments.scenarios
    }

    results = run_robustness_scenarios(
        sessions,
        config,
        nominal_cost_config,
        policies,
        forecast_liquidity,
        selected_scenarios,
    )

    save_robustness_results(
        results,
        selected_scenarios,
        checkpoint_metadata,
        arguments.output,
    )

    print_robustness_summary(results, arguments.seeds)
    print(f"\nSaved results to: {arguments.output}")


if __name__ == "__main__":
    main()
