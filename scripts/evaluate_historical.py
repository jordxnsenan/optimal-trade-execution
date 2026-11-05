import csv
import json

from datetime import date
from functools import partial
from pathlib import Path
from statistics import mean

from execai.data.dataset import load_dataset_split
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.historical import evaluate_historical_sessions
from execai.policies.baselines import immediate_action, twap_action
from execai.data.splits import DEFAULT_SPLITS
from execai.policies.vwap import build_vwap_schedule, vwap_action
from execai.policies.almgren_chriss import optimize_almgren_chriss_schedule
from execai.policies.schedules import scheduled_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "SPY_5min_SIP"
OUTPUT_PATH = PROJECT_ROOT / "results" / "validation" / "2025_baselines.csv"
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


def evaluate_validation_baselines(
    checkpoint_directory: Path,
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, dict[date, float]]:
    """evaluates execution baselines on the validation dataset"""

    sessions = load_dataset_split(
        checkpoint_directory,
        "validation",
    )

    config = ExecutionConfig(
        total_shares=ORDER_SHARES,
        num_time_steps=TIME_STEPS,
        num_lots=INVENTORY_LOTS,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=HALF_SPREAD_BPS,
        temporary_impact_bps=TEMPORARY_IMPACT_BPS,
        fee_per_share=FEE_PER_SHARE,
    )

    calibration = load_training_calibration(calibration_path)

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
    )

    risk_sensitive_schedule = optimize_almgren_chriss_schedule(
        config,
        cost_config,
        reference_price=OPTIMIZATION_REFERENCE_PRICE,
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
    }

    return evaluate_historical_sessions(
        sessions,
        config,
        cost_config,
        policies,
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


def main() -> None:
    results = evaluate_validation_baselines(CHECKPOINT_DIRECTORY)

    save_daily_results(results, OUTPUT_PATH)

    immediate_costs = results["Immediate"]
    twap_costs = results["TWAP"]

    paired_differences = [
        twap_costs[trading_date] - immediate_costs[trading_date]
        for trading_date in immediate_costs
    ]

    print("Historical Validation Evaluation")
    print("---------------------------------")
    print("Dataset: 2025 validation period")
    print(f"Trading sessions: {len(immediate_costs)}")

    print("\nMean implementation shortfall:")

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

    print(f"\nSaved daily results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
