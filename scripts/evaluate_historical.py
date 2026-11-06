import argparse
import csv
import json
import math

from datetime import date
from functools import partial
from pathlib import Path
from statistics import mean, stdev

from execai.data.dataset import load_dataset_split
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.liquidity import build_relative_liquidity_profile
from execai.evaluation.historical import evaluate_historical_sessions
from execai.policies.baselines import immediate_action, twap_action
from execai.data.splits import DEFAULT_SPLITS
from execai.policies.vwap import build_vwap_schedule, vwap_action
from execai.policies.schedules import scheduled_action
from execai.policies.almgren_chriss import (
    calculate_expected_cost_bps,
    calculate_inventory_risk_bps_squared,
    optimize_almgren_chriss_schedule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "SPY_5min_SIP"

OUTPUT_PATH = PROJECT_ROOT / "results" / "validation" / "2025_baselines.csv"

LIQUIDITY_AWARE_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "validation" / "2025_baselines_liquidity_aware.csv"
)

SQUARE_ROOT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "validation" / "2025_baselines_square_root.csv"
)

SQUARE_ROOT_LIQUIDITY_AWARE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "2025_baselines_square_root_liquidity_aware.csv"
)

CALIBRATION_PATH = PROJECT_ROOT / "configs" / "spy_training_calibration.json"
ORDER_SHARES = 10_000
TIME_STEPS = 12
INVENTORY_LOTS = 20

HALF_SPREAD_BPS = 5.0
TEMPORARY_IMPACT_BPS = 20.0
FEE_PER_SHARE = 0.01

AC_RISK_NEUTRAL = 0.0
AC_RISK_SENSITIVE = 0.05
OPTIMIZATION_REFERENCE_PRICE = 100.0


def load_training_calibration(calibration_path: Path) -> dict:
    """loads calibration parameters derived from the training period"""

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    training_range = DEFAULT_SPLITS.training

    if calibration.get("training_start") != str(
        training_range.start_date
    ) or calibration.get("training_end") != str(training_range.end_date):
        raise ValueError("Calibration must use the defined training period.")

    return calibration


def build_experiment_liquidity_profile(
    config: ExecutionConfig,
    calibration: dict,
    liquidity_aware: bool,
) -> list[float] | None:
    """returns forecast liquidity when enabled for an experiment"""

    if not liquidity_aware:
        return None

    return build_relative_liquidity_profile(
        calibration["volume_fractions"],
        config,
    )


def load_vwap_schedule(
    config: ExecutionConfig,
    calibration_path: Path,
) -> list[int]:
    """builds a VWAP schedule from training-derived volume forecasts"""

    calibration = load_training_calibration(calibration_path)

    return build_vwap_schedule(
        config,
        calibration["volume_fractions"],
    )


def build_evaluation_settings(
    impact_model: str = "linear",
    impact_strength_bps: float = TEMPORARY_IMPACT_BPS,
) -> tuple[ExecutionConfig, ExecutionCostConfig]:
    """creates the execution settings shared by historical experiments"""

    config = ExecutionConfig(
        total_shares=ORDER_SHARES,
        num_time_steps=TIME_STEPS,
        num_lots=INVENTORY_LOTS,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=HALF_SPREAD_BPS,
        temporary_impact_bps=impact_strength_bps,
        fee_per_share=FEE_PER_SHARE,
        impact_model=impact_model,
    )

    return config, cost_config


def evaluate_validation_baselines(
    checkpoint_directory: Path,
    calibration_path: Path = CALIBRATION_PATH,
    liquidity_aware: bool = False,
    impact_model: str = "linear",
    impact_strength_bps: float = TEMPORARY_IMPACT_BPS,
) -> dict[str, dict[date, float]]:
    """evaluates execution baselines on the validation dataset"""

    sessions = load_dataset_split(
        checkpoint_directory,
        "validation",
    )

    config, cost_config = build_evaluation_settings(
        impact_model,
        impact_strength_bps,
    )

    calibration = load_training_calibration(calibration_path)

    liquidity_profile = build_experiment_liquidity_profile(
        config,
        calibration,
        liquidity_aware,
    )

    vwap_schedule = build_vwap_schedule(
        config,
        calibration["volume_fractions"],
    )

    volatility = calibration["volatility_per_step"]

    risk_neutral_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=OPTIMIZATION_REFERENCE_PRICE,
        volatility_per_step=volatility,
        risk_aversion=AC_RISK_NEUTRAL,
        liquidity_profile=liquidity_profile,
    )

    risk_sensitive_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=OPTIMIZATION_REFERENCE_PRICE,
        volatility_per_step=volatility,
        risk_aversion=AC_RISK_SENSITIVE,
        liquidity_profile=liquidity_profile,
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

    return evaluate_historical_sessions(
        sessions,
        config,
        cost_config,
        policies,
        liquidity_profile=liquidity_profile,
    )


def save_daily_results(
    results: dict[str, dict[date, float]],
    output_path: Path,
) -> None:
    """saves paired execution costs for each trading date"""

    policy_columns = {
        "Immediate": "immediate_bps",
        "TWAP": "twap_bps",
        "VWAP": "vwap_bps",
        "AC Risk-Neutral": "ac_risk_neutral_bps",
        "AC Risk-Sensitive": "ac_risk_sensitive_bps",
    }

    if any(
        policy_name not in results or not results[policy_name]
        for policy_name in policy_columns
    ):
        raise ValueError("Policies must have matching trading dates.")

    trading_dates = set(results["Immediate"])

    if any(
        set(results[policy_name]) != trading_dates for policy_name in policy_columns
    ):
        raise ValueError("Policies must have matching trading dates.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "trading_date",
                *policy_columns.values(),
                "twap_minus_immediate_bps",
                "vwap_minus_twap_bps",
                "ac_risk_neutral_minus_twap_bps",
                "ac_risk_sensitive_minus_twap_bps",
            ]
        )

        for trading_date in sorted(trading_dates):
            costs = {
                policy_name: results[policy_name][trading_date]
                for policy_name in policy_columns
            }

            writer.writerow(
                [
                    trading_date.isoformat(),
                    *(costs[policy_name] for policy_name in policy_columns),
                    costs["TWAP"] - costs["Immediate"],
                    costs["VWAP"] - costs["TWAP"],
                    costs["AC Risk-Neutral"] - costs["TWAP"],
                    costs["AC Risk-Sensitive"] - costs["TWAP"],
                ]
            )


def summarize_almgren_chriss_risk(
    results: dict[str, dict[date, float]],
    calibration_path: Path,
    liquidity_aware: bool = False,
    impact_model: str = "linear",
    impact_strength_bps: float = TEMPORARY_IMPACT_BPS,
) -> dict[str, dict[str, float]]:
    """compares modeled inventory risk and historical cost variability"""

    config, cost_config = build_evaluation_settings(
        impact_model,
        impact_strength_bps,
    )

    calibration = load_training_calibration(calibration_path)
    volatility = calibration["volatility_per_step"]

    liquidity_profile = build_experiment_liquidity_profile(
        config,
        calibration,
        liquidity_aware,
    )

    risk_settings = {
        "AC Risk-Neutral": AC_RISK_NEUTRAL,
        "AC Risk-Sensitive": AC_RISK_SENSITIVE,
    }

    summaries = {}

    for policy_name, risk_aversion in risk_settings.items():
        daily_costs = list(results[policy_name].values())

        if len(daily_costs) < 2:
            raise ValueError("At least two historical sessions are required.")

        schedule = optimize_almgren_chriss_schedule(
            config,
            cost_config,
            reference_price=OPTIMIZATION_REFERENCE_PRICE,
            volatility_per_step=volatility,
            risk_aversion=risk_aversion,
            liquidity_profile=liquidity_profile,
        )

        expected_cost = calculate_expected_cost_bps(
            schedule,
            config,
            cost_config,
            reference_price=OPTIMIZATION_REFERENCE_PRICE,
            liquidity_profile=liquidity_profile,
        )

        inventory_risk = calculate_inventory_risk_bps_squared(
            schedule,
            config,
            volatility,
        )

        comparison_objective = expected_cost + AC_RISK_SENSITIVE * inventory_risk

        summaries[policy_name] = {
            "expected_cost_bps": expected_cost,
            "inventory_risk_bps_squared": inventory_risk,
            "comparison_objective_bps": comparison_objective,
            "historical_std_bps": stdev(daily_costs),
        }

    return summaries


def get_experiment_output_path(
    impact_model: str,
    liquidity_aware: bool,
    impact_strength_bps: float = TEMPORARY_IMPACT_BPS,
) -> Path:
    """returns the output file for the selected market assumptions"""

    if impact_model == "linear":
        if liquidity_aware:
            output_path = LIQUIDITY_AWARE_OUTPUT_PATH
        else:
            output_path = OUTPUT_PATH

    elif impact_model == "square_root":
        if liquidity_aware:
            output_path = SQUARE_ROOT_LIQUIDITY_AWARE_OUTPUT_PATH
        else:
            output_path = SQUARE_ROOT_OUTPUT_PATH

    else:
        raise ValueError("Unsupported market impact model.")

    if impact_strength_bps == TEMPORARY_IMPACT_BPS:
        return output_path

    filename = (
        f"{output_path.stem}_impact_{impact_strength_bps:g}bps" f"{output_path.suffix}"
    )

    return output_path.with_name(filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate execution baselines on historical SPY data."
    )

    parser.add_argument(
        "--liquidity-aware",
        action="store_true",
        help="Use the training-calibrated relative liquidity profile.",
    )

    parser.add_argument(
        "--impact-model",
        choices=("linear", "square_root"),
        default="linear",
        help="Temporary market-impact model used for evaluation.",
    )

    parser.add_argument(
        "--impact-bps",
        type=float,
        default=TEMPORARY_IMPACT_BPS,
        help="Temporary market-impact coefficient in basis points.",
    )

    arguments = parser.parse_args()

    if not math.isfinite(arguments.impact_bps) or arguments.impact_bps < 0:
        parser.error("Impact strength must be finite and non-negative.")

    output_path = get_experiment_output_path(
        arguments.impact_model,
        arguments.liquidity_aware,
        arguments.impact_bps,
    )

    results = evaluate_validation_baselines(
        CHECKPOINT_DIRECTORY,
        liquidity_aware=arguments.liquidity_aware,
        impact_model=arguments.impact_model,
        impact_strength_bps=arguments.impact_bps,
    )

    save_daily_results(results, output_path)

    immediate_costs = results["Immediate"]

    print("Historical Validation Evaluation")
    print("---------------------------------")
    print("Dataset: 2025 validation period")
    print(f"Trading sessions: {len(immediate_costs)}")

    print("\nMean implementation shortfall:")

    liquidity_model = (
        "Training-derived forecast liquidity"
        if arguments.liquidity_aware
        else "Constant reference liquidity"
    )

    print(f"Liquidity model: {liquidity_model}")

    for policy_name, daily_costs in results.items():
        average_cost = mean(daily_costs.values())
        print(f"{policy_name}: {average_cost:.4f} bps")

    trading_dates = sorted(results["Immediate"])

    comparisons = [
        ("TWAP", "Immediate"),
        ("VWAP", "TWAP"),
        ("AC Risk-Neutral", "TWAP"),
        ("AC Risk-Sensitive", "TWAP"),
    ]

    print("\nPaired mean differences:")

    for first_policy, second_policy in comparisons:
        differences = [
            results[first_policy][trading_date] - results[second_policy][trading_date]
            for trading_date in trading_dates
        ]

        print(f"{first_policy} minus {second_policy}: " f"{mean(differences):+.4f} bps")

    print("\nNegative differences favor the first named policy on cost.")
    print("Risk-sensitive AC also optimizes an inventory-risk penalty.")

    risk_summaries = summarize_almgren_chriss_risk(
        results,
        CALIBRATION_PATH,
        liquidity_aware=arguments.liquidity_aware,
        impact_model=arguments.impact_model,
        impact_strength_bps=arguments.impact_bps,
    )

    print("\nAlmgren-Chriss Risk Analysis")
    print("---------------------------")
    print(f"Comparison risk aversion: {AC_RISK_SENSITIVE}")

    liquidity_model = (
        "Training-derived forecast liquidity"
        if arguments.liquidity_aware
        else "Constant reference liquidity"
    )

    print(f"Liquidity model: {liquidity_model}")
    print(f"Impact model: {arguments.impact_model}")
    print(f"Impact strength: {arguments.impact_bps:.1f} bps")

    for policy_name, summary in risk_summaries.items():
        print(f"\n{policy_name}")

        print(f"Expected cost: " f"{summary['expected_cost_bps']:.4f} bps")

        print(f"Inventory risk: " f"{summary['inventory_risk_bps_squared']:.4f} bps^2")

        print(
            f"Risk-adjusted objective: "
            f"{summary['comparison_objective_bps']:.4f} bps"
        )

        print(
            f"Historical shortfall standard deviation: "
            f"{summary['historical_std_bps']:.4f} bps"
        )

    print(f"\nSaved daily results to: {output_path}")


if __name__ == "__main__":
    main()
