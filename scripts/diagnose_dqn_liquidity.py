"""inspects DQN execution schedules under nominal and reduced liquidity"""

import csv

from collections.abc import Callable
from functools import partial
from statistics import mean

from execai.data.dataset import load_dataset_split
from execai.environment.liquidity import build_relative_liquidity_profile
from execai.evaluation.robustness import evaluate_frozen_scenario

from scripts.evaluate_historical import (
    CALIBRATION_PATH,
    CHECKPOINT_DIRECTORY,
    PROJECT_ROOT,
    build_evaluation_settings,
    load_training_calibration,
)
from scripts.evaluate_dqn_robustness import build_frozen_policies

DIAGNOSTIC_SEEDS = (1001, 4096)

DQN_MODELS = (
    "Basic DQN",
    "Liquidity-Aware DQN",
)

LIQUIDITY_SCENARIOS = {
    "nominal": 1.0,
    "reduced_liquidity": 0.5,
}

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "analysis" / "liquidity_execution_schedules.csv"
)


def track_policy_actions(
    policy: Callable[[dict], int],
    num_time_steps: int,
) -> tuple[Callable[[dict], int], list[int]]:
    """records actions without changing the underlying policy"""

    action_totals = [0] * num_time_steps

    def tracked_policy(observation: dict) -> int:
        action = policy(observation)
        time_step = observation["time_step"]

        action_totals[time_step] += action

        return action

    return tracked_policy, action_totals


def main() -> None:
    config, cost_config = build_evaluation_settings()

    calibration = load_training_calibration(CALIBRATION_PATH)

    forecast_liquidity = build_relative_liquidity_profile(
        calibration["volume_fractions"],
        config,
    )

    sessions = load_dataset_split(
        CHECKPOINT_DIRECTORY,
        "validation",
    )

    policies, _ = build_frozen_policies(
        config,
        cost_config,
        calibration,
        forecast_liquidity,
        DIAGNOSTIC_SEEDS,
    )

    output_rows = []

    print("DQN Liquidity Failure Analysis")
    print("------------------------------")
    print(f"Historical sessions: {len(sessions)}")

    for training_seed in DIAGNOSTIC_SEEDS:
        for model_name in DQN_MODELS:
            policy_name = f"{model_name} (seed {training_seed})"

            for scenario_name, liquidity_scale in LIQUIDITY_SCENARIOS.items():
                tracked_policy, action_totals = track_policy_actions(
                    policies[policy_name],
                    config.num_time_steps,
                )

                results = evaluate_frozen_scenario(
                    sessions,
                    config,
                    cost_config,
                    {policy_name: tracked_policy},
                    forecast_liquidity,
                    impact_model="linear",
                    impact_strength_bps=cost_config.temporary_impact_bps,
                    liquidity_scale=liquidity_scale,
                )

                daily_costs = results[policy_name]
                num_sessions = len(daily_costs)

                average_actions = [total / num_sessions for total in action_totals]

                if abs(sum(average_actions) - config.num_lots) > 1e-9:
                    raise RuntimeError("Execution schedule did not complete the order.")

                average_cost = mean(daily_costs.values())

                final_step_percentage = average_actions[-1] / config.num_lots * 100

                print(f"\n{policy_name} — {scenario_name}")
                print(f"Mean shortfall: {average_cost:.4f} bps")
                print(f"Inventory sold at final step: " f"{final_step_percentage:.1f}%")
                print(
                    "Average lots by step: "
                    f"{[round(action, 2) for action in average_actions]}"
                )

                for time_step, average_lots in enumerate(average_actions):
                    output_rows.append(
                        {
                            "scenario": scenario_name,
                            "model": model_name,
                            "training_seed": training_seed,
                            "time_step": time_step,
                            "average_lots_sold": average_lots,
                            "mean_shortfall_bps": average_cost,
                        }
                    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(output_rows[0]),
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved execution schedules to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
