"""evaluates frozen DQN models on historical validation or held-out data"""

import argparse

from pathlib import Path
from statistics import mean

import torch

from execai.data.dataset import load_dataset_split
from execai.environment.liquidity import build_relative_liquidity_profile

from scripts.analyze_dqn_robustness import TRAINING_SEEDS
from scripts.evaluate_dqn_robustness import (
    ROBUSTNESS_SCENARIOS,
    build_frozen_policies,
    run_robustness_scenarios,
    save_robustness_results,
)
from scripts.evaluate_historical import (
    CALIBRATION_PATH,
    CHECKPOINT_DIRECTORY,
    PROJECT_ROOT,
    build_evaluation_settings,
    load_training_calibration,
)

EXPECTED_SESSIONS = {
    "validation": 250,
    "test": 187,
}

OUTPUT_PATHS = {
    "validation": (PROJECT_ROOT / "results" / "final" / "2025_preflight.csv"),
    "test": (PROJECT_ROOT / "results" / "final" / "2026_held_out.csv"),
}


def load_evaluation_sessions(
    split_name: str,
    confirm_held_out: bool = False,
) -> dict:
    """loads the selected dataset, protecting the held-out test period"""

    if split_name not in EXPECTED_SESSIONS:
        raise ValueError("Unsupported evaluation split.")

    if split_name == "test" and not confirm_held_out:
        raise PermissionError("The held-out test requires explicit confirmation.")

    sessions = load_dataset_split(
        CHECKPOINT_DIRECTORY,
        split_name,
    )

    if len(sessions) != EXPECTED_SESSIONS[split_name]:
        raise ValueError(f"Unexpected number of {split_name} trading sessions.")

    return sessions


def run_final_evaluation(
    split_name: str,
    output_path: Path,
    confirm_held_out: bool = False,
) -> dict:
    """evaluates all frozen policies using the finalized market scenarios"""

    if split_name == "test" and output_path.exists():
        raise FileExistsError(
            "Held-out test results already exist. "
            "The final test should not be overwritten."
        )

    sessions = load_evaluation_sessions(
        split_name,
        confirm_held_out=confirm_held_out,
    )

    config, cost_config = build_evaluation_settings()

    calibration = load_training_calibration(CALIBRATION_PATH)

    forecast_liquidity = build_relative_liquidity_profile(
        calibration["volume_fractions"],
        config,
    )

    policies, checkpoint_metadata = build_frozen_policies(
        config,
        cost_config,
        calibration,
        forecast_liquidity,
        TRAINING_SEEDS,
    )

    results = run_robustness_scenarios(
        sessions,
        config,
        cost_config,
        policies,
        forecast_liquidity,
        ROBUSTNESS_SCENARIOS,
    )

    save_robustness_results(
        results,
        ROBUSTNESS_SCENARIOS,
        checkpoint_metadata,
        output_path,
    )

    return results


def print_nominal_summary(results: dict) -> None:
    """prints the nominal execution costs across the saved policies"""

    nominal_results = results["nominal"]

    print("\nNominal execution costs:")

    for policy_name in ("TWAP", "AC Risk-Neutral"):
        average_cost = mean(nominal_results[policy_name].values())
        print(f"{policy_name}: {average_cost:.4f} bps")

    for model_name in ("Basic DQN", "Liquidity-Aware DQN"):
        seed_costs = []

        for training_seed in TRAINING_SEEDS:
            policy_name = f"{model_name} (seed {training_seed})"

            seed_costs.append(mean(nominal_results[policy_name].values()))

        print(f"{model_name}: {mean(seed_costs):.4f} bps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DQN models on historical market data."
    )

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )

    parser.add_argument(
        "--confirm-held-out",
        action="store_true",
        help="Explicitly authorize the held-out 2026 evaluation.",
    )

    arguments = parser.parse_args()

    torch.set_num_threads(1)

    output_path = OUTPUT_PATHS[arguments.split]

    results = run_final_evaluation(
        arguments.split,
        output_path,
        confirm_held_out=arguments.confirm_held_out,
    )

    print("ExecAI - Frozen Model Evaluation")
    print("--------------------------------")
    print(f"Dataset: {arguments.split}")
    print(f"Training seeds: {len(TRAINING_SEEDS)}")
    print(f"Market scenarios: {len(ROBUSTNESS_SCENARIOS)}")
    print(f"Trading sessions: {len(results['nominal']['TWAP'])}")

    print_nominal_summary(results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
