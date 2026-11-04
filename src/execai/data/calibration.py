import math

from datetime import date
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

from execai.data.dataset import load_dataset_split
from execai.data.session_validation import (
    BAR_INTERVAL_MINUTES,
    validate_trading_session,
)

REGULAR_SESSION_MINUTES = 390
REGULAR_SESSION_BAR_COUNT = REGULAR_SESSION_MINUTES // BAR_INTERVAL_MINUTES


def estimate_volatility_per_step(
    sessions: dict[date, list[dict]],
) -> float:
    """estimates five-minute volatility from historical closing prices"""

    log_returns = []

    for bars in sessions.values():
        for index in range(1, len(bars)):
            previous_close = bars[index - 1]["c"]
            current_close = bars[index]["c"]

            log_return = math.log(current_close / previous_close)
            log_returns.append(log_return)

    if len(log_returns) < 2:
        raise ValueError("At least two intraday returns are required for calibration.")

    return stdev(log_returns)


def calibrate_training_volatility(
    checkpoint_directory: Path,
) -> float:
    """calculates volatility using only historical training data"""

    training_sessions = load_dataset_split(
        checkpoint_directory,
        "training",
    )

    return estimate_volatility_per_step(training_sessions)


@dataclass(frozen=True)
class IntradayVolumeProfile:
    """stores the average distribution of volume throughout a trading day"""

    volume_fractions: tuple[float, ...]
    full_sessions_used: int
    shortened_sessions_excluded: int


def estimate_intraday_volume_profile(
    sessions: dict[date, list[dict]],
) -> IntradayVolumeProfile:
    """estimates the average five-minute volume distribution"""

    volume_fraction_totals = [0.0] * REGULAR_SESSION_BAR_COUNT
    full_sessions_used = 0
    shortened_sessions_excluded = 0

    for trading_date, bars in sessions.items():
        if len(bars) != REGULAR_SESSION_BAR_COUNT:

            # ensures only valid shortened sessions can be excluded
            validate_trading_session(bars, trading_date)

            shortened_sessions_excluded += 1

            continue

        daily_volumes = [bar["v"] for bar in bars]

        if any(type(volume) is not int or volume < 0 for volume in daily_volumes):
            raise ValueError(f"Invalid trading volume on {trading_date}.")

        total_daily_volume = sum(daily_volumes)

        if total_daily_volume == 0:
            raise ValueError(f"No trading volume on {trading_date}.")

        for index, volume in enumerate(daily_volumes):
            volume_fraction_totals[index] += volume / total_daily_volume

        full_sessions_used += 1

    if full_sessions_used == 0:
        raise ValueError("At least one complete regular session is required.")

    average_volume_fractions = tuple(
        volume_total / full_sessions_used for volume_total in volume_fraction_totals
    )

    return IntradayVolumeProfile(
        volume_fractions=average_volume_fractions,
        full_sessions_used=full_sessions_used,
        shortened_sessions_excluded=shortened_sessions_excluded,
    )


def calibrate_training_volume_profile(
    checkpoint_directory: Path,
) -> IntradayVolumeProfile:
    """estimates the volume profile using only training data"""

    training_sessions = load_dataset_split(
        checkpoint_directory,
        "training",
    )

    return estimate_intraday_volume_profile(training_sessions)
