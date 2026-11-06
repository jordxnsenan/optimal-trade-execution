import argparse

from datetime import date
from functools import partial
from pathlib import Path

from execai.data.dataset import load_spy_sessions
from execai.data.splits import DEFAULT_SPLITS
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.historical import build_historical_price_path
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.policies.baselines import immediate_action, twap_action

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "SPY_5min_SIP"
)


def compare_baselines(bars: list[dict]) -> dict[str, dict]:
    """compares execution strategies using one historical trading session"""

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

    prices = build_historical_price_path(bars, config)

    policies = {
        "Immediate Execution": immediate_action,
        "TWAP": partial(twap_action, config=config),
    }

    results = {}

    for policy_name, choose_action in policies.items():
        environment = ExecutionEnvironment(config, cost_config)

        execution_cost = run_policy(
            environment,
            prices,
            choose_action,
        )

        execution_schedule = [
            record.shares_sold for record in environment.execution_history
        ]

        results[policy_name] = {
            "execution_schedule": execution_schedule,
            "implementation_shortfall_bps": execution_cost,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare execution strategies on historical SPY data."
    )

    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date(2025, 10, 15),
        help="Historical validation date in YYYY-MM-DD format.",
    )

    arguments = parser.parse_args()

    validation_range = DEFAULT_SPLITS.validation

    if not (validation_range.start_date <= arguments.date <= validation_range.end_date):
        parser.error("The requested date must belong to the validation period.")

    sessions = load_spy_sessions(
        CHECKPOINT_DIRECTORY,
        arguments.date,
        arguments.date,
    )

    bars = sessions[arguments.date]
    results = compare_baselines(bars)

    print("Historical Execution Comparison")
    print("--------------------------------")
    print(f"Trading date: {arguments.date}")
    print("Execution window: 10:00-11:00 AM Eastern")
    print("Order size: 10,000 shares")

    for policy_name, result in results.items():
        print(f"\n{policy_name}")
        print(f"Execution schedule: {result['execution_schedule']}")
        print(
            f"Implementation shortfall: "
            f"{result['implementation_shortfall_bps']:.4f} bps"
        )


if __name__ == "__main__":
    main()
