"""generates figures from the saved DQN robustness summaries"""

import argparse
import csv
import math
import matplotlib

import matplotlib.pyplot as plt

matplotlib.use("Agg")

from pathlib import Path
from statistics import mean, stdev
from scripts.analyze_dqn_robustness import (
    DQN_MODELS,
    SCENARIOS,
    TRAINING_SEEDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures"

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "low_impact": "Low impact",
    "high_impact": "High impact",
    "square_root": "Square-root",
    "reduced_liquidity": "Low liquidity",
    "late_liquidity_drop": "Late liquidity drop",
}


def read_csv_rows(path: Path) -> list[dict]:
    """reads a previously generated summary CSV"""

    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No summary data found in {path}")

    return rows


def get_seed_values(
    rows: list[dict],
    scenario: str,
    model: str,
    column: str,
) -> list[float]:
    """returns one value per training seed in the expected order"""

    values_by_seed = {}

    for row in rows:
        if row["scenario"] != scenario or row["model"] != model:
            continue

        training_seed = int(row["training_seed"])
        value = float(row[column])

        if training_seed in values_by_seed:
            raise ValueError("Duplicate training seed in summary.")

        if not math.isfinite(value):
            raise ValueError("Summary values must be finite.")

        values_by_seed[training_seed] = value

    if set(values_by_seed) != set(TRAINING_SEEDS):
        raise ValueError("Missing or unexpected training seeds.")

    return [values_by_seed[training_seed] for training_seed in TRAINING_SEEDS]


def get_baseline_cost(
    rows: list[dict],
    scenario: str,
    policy: str,
) -> float:
    """finds the mean execution cost of one classical strategy"""

    matching_rows = [
        row for row in rows if row["scenario"] == scenario and row["policy"] == policy
    ]

    if len(matching_rows) != 1:
        raise ValueError(f"Expected one summary row for {policy}.")

    return float(matching_rows[0]["mean_shortfall_bps"])


def plot_nominal_training_seeds(
    seed_rows: list[dict],
    scenario_rows: list[dict],
    output_path: Path,
) -> None:
    """compares the eight trained DQN models with nominal baselines"""

    x_positions = list(range(len(TRAINING_SEEDS)))

    figure, axis = plt.subplots(figsize=(10, 5))

    for model in DQN_MODELS:
        costs = get_seed_values(
            seed_rows,
            "nominal",
            model,
            "mean_shortfall_bps",
        )

        axis.scatter(
            x_positions,
            costs,
            s=50,
            label=model,
        )

    for policy, line_style in (
        ("TWAP", "--"),
        ("AC Risk-Neutral", ":"),
    ):
        reference_cost = get_baseline_cost(
            scenario_rows,
            "nominal",
            policy,
        )

        axis.axhline(
            reference_cost,
            linestyle=line_style,
            label=policy,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(seed) for seed in TRAINING_SEEDS])

    axis.set_title("DQN Historical Validation Across Training Seeds")
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Mean implementation shortfall (bps)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_robustness_gaps(
    seed_rows: list[dict],
    output_path: Path,
) -> None:
    """plots DQN cost differences and variability across training seeds"""

    x_positions = list(range(len(SCENARIOS)))

    figure, axis = plt.subplots(figsize=(11, 5))

    for model, offset in (
        ("Basic DQN", -0.1),
        ("Liquidity-Aware DQN", 0.1),
    ):
        average_differences = []
        seed_standard_deviations = []

        for scenario in SCENARIOS:
            differences = get_seed_values(
                seed_rows,
                scenario,
                model,
                "minus_ac_bps",
            )

            average_differences.append(mean(differences))
            seed_standard_deviations.append(stdev(differences))

        shifted_positions = [position + offset for position in x_positions]

        axis.errorbar(
            shifted_positions,
            average_differences,
            yerr=seed_standard_deviations,
            fmt="o",
            capsize=4,
            label=model,
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
        label="Equal cost to AC",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [SCENARIO_LABELS[scenario] for scenario in SCENARIOS],
        rotation=20,
        ha="right",
    )

    axis.set_title("Frozen DQN Performance Under Market Stress")
    axis.set_ylabel("Additional cost relative to AC (bps)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    arguments = parser.parse_args()

    seed_rows = read_csv_rows(arguments.input_dir / "seed_summary.csv")

    scenario_rows = read_csv_rows(arguments.input_dir / "scenario_summary.csv")

    nominal_output = arguments.output_dir / "nominal_training_seeds.png"

    robustness_output = arguments.output_dir / "robustness_cost_gaps.png"

    plot_nominal_training_seeds(
        seed_rows,
        scenario_rows,
        nominal_output,
    )

    plot_robustness_gaps(
        seed_rows,
        robustness_output,
    )

    print(f"Saved: {nominal_output}")
    print(f"Saved: {robustness_output}")


if __name__ == "__main__":
    main()
