from collections.abc import Callable
from functools import partial

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.experiment_settings import (
    AC_RISK_NEUTRAL,
    AC_RISK_SENSITIVE,
    OPTIMIZATION_REFERENCE_PRICE,
)
from execai.policies.almgren_chriss import optimize_almgren_chriss_schedule
from execai.policies.baselines import immediate_action, twap_action
from execai.policies.schedules import scheduled_action
from execai.policies.vwap import build_vwap_schedule, vwap_action


def build_baseline_policies(
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    calibration: dict,
    liquidity_profile: list[float] | None = None,
) -> dict[str, Callable[[dict], int]]:
    """builds the classical execution policies for an experiment"""

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

    return {
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
