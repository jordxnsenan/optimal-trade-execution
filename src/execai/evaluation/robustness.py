import math

from collections.abc import Callable
from dataclasses import replace
from datetime import date

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.historical import evaluate_historical_sessions


def evaluate_frozen_scenario(
    sessions: dict,
    config: ExecutionConfig,
    nominal_cost_config: ExecutionCostConfig,
    frozen_policies: dict[str, Callable[[dict], int]],
    forecast_liquidity: list[float],
    impact_model: str,
    impact_strength_bps: float,
    liquidity_scale: float = 1.0,
) -> dict[str, dict[date, float]]:
    """evaluates unchanged policies under alternative market assumptions"""

    if not frozen_policies:
        raise ValueError("At least one execution policy is required.")

    if len(forecast_liquidity) != config.num_time_steps:
        raise ValueError(
            "Liquidity profile length must match the number of time steps."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in forecast_liquidity
    ):
        raise ValueError("Forecast liquidity must be positive and finite.")

    if (
        isinstance(liquidity_scale, bool)
        or not isinstance(liquidity_scale, (int, float))
        or not math.isfinite(liquidity_scale)
        or liquidity_scale <= 0
    ):
        raise ValueError("Liquidity scale must be positive and finite.")

    scenario_liquidity = [value * liquidity_scale for value in forecast_liquidity]

    if any(not math.isfinite(value) for value in scenario_liquidity):
        raise ValueError("Scaled liquidity must be finite.")

    scenario_cost_config = replace(
        nominal_cost_config,
        temporary_impact_bps=impact_strength_bps,
        impact_model=impact_model,
    )

    return evaluate_historical_sessions(
        sessions,
        config,
        scenario_cost_config,
        frozen_policies,
        liquidity_profile=scenario_liquidity,
    )
