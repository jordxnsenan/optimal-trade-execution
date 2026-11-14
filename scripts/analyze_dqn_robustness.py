"""summarizes saved DQN robustness results across independent training seeds"""

import argparse
import csv
import math

from datetime import date
from pathlib import Path
from statistics import mean, stdev

from execai.evaluation.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    two_way_bootstrap_confidence_interval,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "robustness" / "2025_frozen_dqn.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "analysis"

TRAINING_SEEDS = (7, 42, 123, 314, 777, 1001, 2026, 4096)

SCENARIOS = (
    "nominal",
    "low_impact",
    "high_impact",
    "square_root",
    "reduced_liquidity",
    "late_liquidity_drop",
)

BASELINE_POLICIES = (
    "Immediate",
    "TWAP",
    "VWAP",
    "AC Risk-Neutral",
    "AC Risk-Sensitive",
)

DQN_MODELS = ("Basic DQN", "Liquidity-Aware DQN")

EXPECTED_TRADING_SESSIONS = 250

REQUIRED_COLUMNS = {
    "scenario",
    "trading_date",
    "policy",
    "training_seed",
    "best_episode",
    "shortfall_bps",
}


def load_results(input_path: Path) -> dict[tuple[str, str, str], float]:
    """reads unique, finite execution costs from the robustness CSV"""

    observations = {}

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError("Robustness CSV is missing required columns.")

        for row in reader:
            scenario = row["scenario"]
            trading_date = row["trading_date"]
            policy = row["policy"]
            key = (scenario, trading_date, policy)

            if not scenario or not policy:
                raise ValueError("Scenario and policy names cannot be empty.")

            if " (seed " in policy:
                seed_from_name = policy.rsplit(" (seed ", 1)[1].removesuffix(")")

                if row["training_seed"] != seed_from_name or not row["best_episode"]:
                    raise ValueError(f"Invalid checkpoint metadata: {key}")

            elif row["training_seed"] or row["best_episode"]:
                raise ValueError(f"Unexpected checkpoint metadata: {key}")

            try:
                date.fromisoformat(trading_date)
                shortfall = float(row["shortfall_bps"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid result: {key}") from error

            if not math.isfinite(shortfall):
                raise ValueError(f"Nonfinite execution cost: {key}")

            if key in observations:
                raise ValueError(f"Duplicate execution result: {key}")

            observations[key] = shortfall

    if not observations:
        raise ValueError("Robustness results file is empty.")

    return observations


def summarize_results(
    observations: dict[tuple[str, str, str], float],
    scenarios: tuple[str, ...] = SCENARIOS,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
) -> tuple[list[dict], list[dict]]:
    """computed scenario means and variability between trained DQN agents"""

    trading_dates = sorted(
        {
            trading_date
            for scenario, trading_date, policy in observations
            if scenario == scenarios[0] and policy == "TWAP"
        }
    )

    if len(trading_dates) < 2:
        raise ValueError("At least two matching trading sessions are required.")

    expected_policies = list(BASELINE_POLICIES)

    expected_policies.extend(
        f"{model} (seed {seed})" for model in DQN_MODELS for seed in training_seeds
    )

    expected_keys = {
        (scenario, trading_date, policy)
        for scenario in scenarios
        for trading_date in trading_dates
        for policy in expected_policies
    }

    if set(observations) != expected_keys:
        raise ValueError("Missing or unexpected scenario, date, or policy results.")

    scenario_rows = []
    seed_rows = []

    for scenario in scenarios:
        baseline_means = {}

        for policy in BASELINE_POLICIES:
            baseline_mean = mean(
                observations[scenario, trading_date, policy]
                for trading_date in trading_dates
            )

            baseline_means[policy] = baseline_mean

            scenario_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "mean_shortfall_bps": baseline_mean,
                    "seed_std_bps": "",
                }
            )

        for model in DQN_MODELS:
            seed_means = []

            for seed in training_seeds:
                policy = f"{model} (seed {seed})"

                seed_mean = mean(
                    observations[scenario, trading_date, policy]
                    for trading_date in trading_dates
                )

                seed_means.append(seed_mean)

                seed_rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "training_seed": seed,
                        "mean_shortfall_bps": seed_mean,
                        "minus_twap_bps": seed_mean - baseline_means["TWAP"],
                        "minus_ac_bps": seed_mean - baseline_means["AC Risk-Neutral"],
                    }
                )

            scenario_rows.append(
                {
                    "scenario": scenario,
                    "policy": model,
                    "mean_shortfall_bps": mean(seed_means),
                    "seed_std_bps": stdev(seed_means),
                }
            )

    return scenario_rows, seed_rows


def summarize_paired_intervals(
    observations: dict[tuple[str, str, str], float],
    scenarios: tuple[str, ...] = SCENARIOS,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
    num_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> list[dict]:
    """compares DQN models with baselines across seeds and days"""

    trading_dates = sorted(
        {
            trading_date
            for scenario, trading_date, policy in observations
            if scenario == scenarios[0] and policy == "TWAP"
        }
    )

    comparisons = ("TWAP", "AC Risk-Neutral")
    summary_rows = []

    for scenario in scenarios:
        for model in DQN_MODELS:
            for baseline in comparisons:
                differences_by_seed = []

                for seed in training_seeds:
                    policy_name = f"{model} (seed {seed})"

                    daily_differences = [
                        observations[scenario, trading_date, policy_name]
                        - observations[scenario, trading_date, baseline]
                        for trading_date in trading_dates
                    ]

                    differences_by_seed.append(daily_differences)

                mean_difference = mean(
                    mean(differences) for differences in differences_by_seed
                )

                lower, upper = two_way_bootstrap_confidence_interval(
                    differences_by_seed,
                    num_samples=num_samples,
                )

                summary_rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "baseline": baseline,
                        "mean_difference_bps": mean_difference,
                        "ci_lower_bps": lower,
                        "ci_upper_bps": upper,
                        "num_seeds": len(training_seeds),
                        "num_sessions": len(trading_dates),
                    }
                )

    return summary_rows


def save_csv(output_path: Path, rows: list[dict]) -> None:
    """writes a summary table with consistent column names"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=EXPECTED_TRADING_SESSIONS,
        help="Expected number of historical trading sessions.",
    )

    arguments = parser.parse_args()

    if arguments.expected_sessions < 2:
        parser.error("Expected sessions must be at least 2.")

    observations = load_results(arguments.input)
    scenario_rows, seed_rows = summarize_results(observations)
    paired_rows = summarize_paired_intervals(observations)

    trading_dates = {
        trading_date
        for scenario, trading_date, policy in observations
        if scenario == "nominal" and policy == "TWAP"
    }

    if len(trading_dates) != arguments.expected_sessions:
        raise ValueError(
            f"Expected {arguments.expected_sessions} historical sessions, "
            f"found {len(trading_dates)}."
        )

    save_csv(
        arguments.output_dir / "scenario_summary.csv",
        scenario_rows,
    )

    save_csv(
        arguments.output_dir / "seed_summary.csv",
        seed_rows,
    )

    save_csv(
        arguments.output_dir / "paired_intervals.csv",
        paired_rows,
    )

    print(f"Validated {len(observations):,} execution results.")
    print(f"Saved scenario and seed summaries to: {arguments.output_dir}")

    for scenario in SCENARIOS:
        print(f"\n{scenario}:")

        for row in scenario_rows:
            if row["scenario"] != scenario or row["policy"] not in (
                "TWAP",
                "AC Risk-Neutral",
                *DQN_MODELS,
            ):
                continue

            extra = ""

            if row["seed_std_bps"] != "":
                extra = f" | seed SD {row['seed_std_bps']:.4f}"

            print(f"  {row['policy']}: {row['mean_shortfall_bps']:.4f} bps{extra}")

    print("\nNominal paired comparisons (exploratory 95% intervals):")

    for row in paired_rows:
        if row["scenario"] != "nominal":
            continue

        print(
            f"{row['model']} minus {row['baseline']}: "
            f"{row['mean_difference_bps']:+.4f} bps "
            f"| 95% CI "
            f"[{row['ci_lower_bps']:+.4f}, "
            f"{row['ci_upper_bps']:+.4f}]"
        )


if __name__ == "__main__":
    main()
