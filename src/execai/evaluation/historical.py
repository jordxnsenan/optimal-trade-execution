from collections.abc import Callable
from datetime import date

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.historical import build_historical_price_path
from execai.evaluation.comparison import evaluate_policies


def evaluate_historical_sessions(
    sessions: dict[date, list[dict]],
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    policies: dict[str, Callable[[dict], int]],
) -> dict[str, dict[date, float]]:
    """evaluates policies across trading sessions"""

    if not sessions:
        raise ValueError("At least one trading session is required.")

    if not policies:
        raise ValueError("At least one execution policy is required.")

    trading_dates = sorted(sessions)
    price_paths = []

    for trading_date in trading_dates:
        prices = build_historical_price_path(
            sessions[trading_date],
            config,
        )

        price_paths.append(prices)

    execution_costs = evaluate_policies(
        config,
        cost_config,
        price_paths,
        policies,
    )

    results = {}

    for policy_name, costs in execution_costs.items():
        results[policy_name] = dict(zip(trading_dates, costs))

    return results
