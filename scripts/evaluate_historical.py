import csv

from datetime import date
from functools import partial
from pathlib import Path
from statistics import mean

from execai.data.dataset import load_dataset_split
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.historical import evaluate_historical_sessions
from execai.policies.baselines import immediate_action, twap_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "SPY_5min_SIP"
OUTPUT_PATH = PROJECT_ROOT / "results" / "validation" / "2025_baselines.csv"

ORDER_SHARES = 10_000
TIME_STEPS = 12
INVENTORY_LOTS = 20

HALF_SPREAD_BPS = 5.0
TEMPORARY_IMPACT_BPS = 20.0
FEE_PER_SHARE = 0.01


def evaluate_validation_baselines(
    checkpoint_directory: Path,
) -> dict[str, dict[date, float]]:
    """evaluates traditional execution policies on the validation dataset"""

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

    policies = {
        "Immediate": immediate_action,
        "TWAP": partial(twap_action, config=config),
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

    immediate_costs = results["Immediate"]
    twap_costs = results["TWAP"]

    if not immediate_costs or immediate_costs.keys() != twap_costs.keys():
        raise ValueError("Policies must have matching trading dates.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "trading_date",
                "immediate_bps",
                "twap_bps",
                "twap_minus_immediate_bps",
            ]
        )

        for trading_date in sorted(immediate_costs):
            immediate_cost = immediate_costs[trading_date]
            twap_cost = twap_costs[trading_date]

            writer.writerow(
                [
                    trading_date.isoformat(),
                    immediate_cost,
                    twap_cost,
                    twap_cost - immediate_cost,
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
    print(f"Immediate: {mean(immediate_costs.values()):.4f} bps")
    print(f"TWAP: {mean(twap_costs.values()):.4f} bps")

    print(f"\nTWAP minus Immediate: " f"{mean(paired_differences):+.4f} bps")

    print("Negative differences favor TWAP.")
    print(f"\nSaved daily results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
