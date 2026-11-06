import json

from datetime import date
from pathlib import Path

from execai.data.calibration import (
    estimate_intraday_volume_profile,
    estimate_volatility_per_step,
)
from execai.data.dataset import load_dataset_split
from execai.data.splits import DEFAULT_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "SPY_5min_SIP"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "spy_training_calibration.json"


def build_calibration(sessions: dict[date, list[dict]]) -> dict:
    """calculates market parameters from training sessions"""

    volatility = estimate_volatility_per_step(sessions)
    volume_profile = estimate_intraday_volume_profile(sessions)

    training_range = DEFAULT_SPLITS.training

    return {
        "symbol": "SPY",
        "source": "Alpaca",
        "feed": "sip",
        "timeframe": "5Min",
        "adjustment": "raw",
        "training_start": str(training_range.start_date),
        "training_end": str(training_range.end_date),
        "volatility_per_step": volatility,
        "regular_sessions_used": volume_profile.full_sessions_used,
        "shortened_sessions_excluded": volume_profile.shortened_sessions_excluded,
        "volume_fractions": list(volume_profile.volume_fractions),
    }


def main() -> None:
    """calibrates historical SPY data and save the results"""

    training_sessions = load_dataset_split(
        CHECKPOINT_DIRECTORY,
        "training",
    )

    calibration = build_calibration(training_sessions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Historical Market Calibration")
    print("-----------------------------")
    print(f"Training sessions: {len(training_sessions)}")
    print(f"Five-minute volatility: {calibration['volatility_per_step']:.6f}")
    print(f"Regular sessions used: {calibration['regular_sessions_used']}")
    print(f"Shortened sessions excluded: {calibration['shortened_sessions_excluded']}")
    print(f"Volume intervals: {len(calibration['volume_fractions'])}")
    print(f"Saved calibration to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
