import argparse

from datetime import date
from pathlib import Path
from execai.data.coverage import audit_market_data

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "SPY_5min_SIP"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit historical SPY market data coverage."
    )

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="First date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="Last date in YYYY-MM-DD format.",
    )

    arguments = parser.parse_args()

    try:
        report = audit_market_data(
            CHECKPOINT_DIRECTORY,
            arguments.start,
            arguments.end,
        )
    except (ValueError, OSError) as error:
        print(f"Audit failed: {error}")
        return 1

    print("\nExecAI - Historical Data Coverage")
    print("--------------------------------")
    print(f"Date range: {arguments.start} to {arguments.end}")
    print(f"Expected sessions: {report.expected_sessions}")
    print(f"Completed sessions: {report.completed_sessions}")
    print(f"Expected bars: {report.expected_bars}")
    print(f"Completed bars: {report.completed_bars}")
    print(f"Coverage: {report.coverage_percent:.2f}%")

    if report.missing_dates:
        print("\nMissing trading sessions:")

        for missing_date in report.missing_dates:
            print(f"  {missing_date}")

        return 1

    print("\nAll expected trading sessions are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
