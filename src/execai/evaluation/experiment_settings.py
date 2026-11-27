"""This file contains the shared settings to be used for historical execution experiments"""

import json

from pathlib import Path

from execai.data.splits import DEFAULT_SPLITS
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.liquidity import build_relative_liquidity_profile

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
    """loads calibration parameters from the training period"""

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


def build_evaluation_settings(
    impact_model: str = "linear",
    impact_strength_bps: float = TEMPORARY_IMPACT_BPS,
) -> tuple[ExecutionConfig, ExecutionCostConfig]:
    """builds the execution settings shared by historical experiments"""

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
