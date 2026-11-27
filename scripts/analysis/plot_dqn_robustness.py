import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts.analysis.analyze_dqn_robustness import (
    SCENARIOS,
    TRAINING_SEEDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures"

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "low_impact": "Low impact",
    "high_impact": "High impact",
    "square_root": "Square-root impact",
    "reduced_liquidity": "Reduced liquidity",
    "late_liquidity_drop": "Late liquidity drop",
}

BASIC_DQN = "Basic DQN"
LIQUIDITY_AWARE_DQN = "Liquidity-Aware DQN"
RISK_NEUTRAL_AC = "AC Risk-Neutral"
REDUCED_LIQUIDITY = "reduced_liquidity"

MODEL_STYLES = {
    BASIC_DQN: ("#006B59", "o"),
    LIQUIDITY_AWARE_DQN: ("#694465", "D"),
}

TEXT_COLOR = "#232C30"
SECONDARY_TEXT_COLOR = "#49565A"
GRID_COLOR = "#E1E5E4"

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": SECONDARY_TEXT_COLOR,
    "ytick.color": SECONDARY_TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
}


def add_figure_heading(figure: Figure, title: str, dataset_label: str) -> None:
    """places the dataset label above the chart title"""
    heading = "EXECAI"

    if dataset_label:
        heading += f"    ·    {dataset_label.upper()}"

    figure.text(
        0.045,
        0.955,
        heading,
        fontsize=9,
        weight="bold",
        color=SECONDARY_TEXT_COLOR,
    )

    figure.text(0.045, 0.876, title, fontsize=17, weight="bold")


def save_figure(figure: Figure, output_path: Path) -> None:
    """saves a PNG and vector PDF of the figure"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=320)
    figure.savefig(output_path.with_suffix(".pdf"))

    plt.close(figure)


def read_csv_rows(path: Path) -> list[dict]:
    """reads a summary CSV and rejects an empty file"""

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
    """returns one value per seed in the same order as the chart labels"""

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
    """reads the mean cost of one baseline in the requested scenario"""

    matching_rows = [
        row for row in rows if row["scenario"] == scenario and row["policy"] == policy
    ]

    if len(matching_rows) != 1:
        raise ValueError(f"Expected one summary row for {policy}.")

    return float(matching_rows[0]["mean_shortfall_bps"])


def add_legend(figure: Figure, axis: Axes, top: float) -> None:
    """places the legend below the heading"""
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.04, top),
        frameon=False,
        ncol=2,
        fontsize=9,
        borderaxespad=0,
        handlelength=2.8,
        columnspacing=2.2,
    )


def draw_seed_pairs(
    axis: Axes,
    seed_positions: list[int],
    basic_costs: list[float],
    liquidity_aware_costs: list[float],
) -> None:
    """connects results for models trained with the same seed"""
    marker_offset = 0.075
    basic_positions = [position - marker_offset for position in seed_positions]
    liquidity_positions = [position + marker_offset for position in seed_positions]

    for seed_index in range(len(seed_positions)):
        axis.plot(
            [basic_positions[seed_index], liquidity_positions[seed_index]],
            [basic_costs[seed_index], liquidity_aware_costs[seed_index]],
            color="#B7C1BF",
            linewidth=1,
            zorder=2,
        )

    model_points = [
        (BASIC_DQN, basic_positions, basic_costs),
        (LIQUIDITY_AWARE_DQN, liquidity_positions, liquidity_aware_costs),
    ]
    for model_name, positions, costs in model_points:
        color, marker = MODEL_STYLES[model_name]
        axis.scatter(
            positions,
            costs,
            label=model_name,
            color=color,
            marker=marker,
            s=47,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )


def add_nominal_baselines(axis: Axes, scenario_rows: list[dict]) -> None:
    """gives the two nearly overlapping baselines distinct lines and labels"""
    twap_cost = get_baseline_cost(scenario_rows, "nominal", "TWAP")
    ac_cost = get_baseline_cost(scenario_rows, "nominal", RISK_NEUTRAL_AC)

    axis.axhline(
        twap_cost,
        label=f"TWAP: {twap_cost:.3f} bps",
        color="#586768",
        linestyle=(0, (6, 3)),
        linewidth=1.15,
        zorder=1,
    )
    axis.axhline(
        ac_cost,
        label=f"Risk-neutral AC: {ac_cost:.3f} bps",
        color="#252D32",
        linestyle=(0, (1, 2)),
        linewidth=1.15,
        zorder=1,
    )


@plt.rc_context(PLOT_STYLE)
def plot_nominal_training_seeds(
    seed_rows: list[dict],
    scenario_rows: list[dict],
    output_path: Path,
    dataset_label: str = "",
) -> None:
    """compares both models with TWAP and Almgren-Chriss under nominal conditions"""
    basic_costs = get_seed_values(seed_rows, "nominal", BASIC_DQN, "mean_shortfall_bps")
    liquidity_aware_costs = get_seed_values(
        seed_rows, "nominal", LIQUIDITY_AWARE_DQN, "mean_shortfall_bps"
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.9))
    figure.subplots_adjust(left=0.105, right=0.97, top=0.72, bottom=0.19)
    add_figure_heading(figure, "Execution cost by training seed", dataset_label)

    seed_positions = list(range(len(TRAINING_SEEDS)))
    draw_seed_pairs(axis, seed_positions, basic_costs, liquidity_aware_costs)
    add_nominal_baselines(axis, scenario_rows)

    axis.set_xticks(seed_positions, [str(seed) for seed in TRAINING_SEEDS])
    axis.set_xlim(seed_positions[0] - 0.48, seed_positions[-1] + 0.48)
    axis.margins(y=0.12)
    axis.set_xlabel("Training seed", labelpad=7)
    axis.set_ylabel("Mean implementation shortfall (bps)", labelpad=8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    axis.tick_params(length=0, pad=5)

    add_legend(figure, axis, top=0.817)
    figure.text(
        0.045,
        0.041,
        "Each point averages one model’s cost across the same sessions. Lower is better.",
        fontsize=9,
        color=SECONDARY_TEXT_COLOR,
    )
    save_figure(figure, output_path)


def shade_scenario_rows(axis: Axes) -> None:
    """guides the eye across the chart and highlights the row w/ seed detail"""
    for row_index, scenario in enumerate(SCENARIOS):
        if scenario == REDUCED_LIQUIDITY:
            row_color = "#E9EEEB"
        elif row_index % 2 == 0:
            row_color = "#F4F6F5"
        else:
            continue

        axis.axhspan(
            row_index - 0.5,
            row_index + 0.5,
            color=row_color,
            linewidth=0,
            zorder=0,
        )


def draw_reduced_liquidity_seeds(
    axis: Axes,
    model_name: str,
    seed_cost_gaps: list[float],
    model_row: float,
    row_offset: float,
) -> None:
    """staggers nearby seed markers vertically without moving their cost values"""
    color, marker = MODEL_STYLES[model_name]
    ordered_cost_gaps = sorted(seed_cost_gaps)
    seed_positions = []

    for seed_index in range(len(ordered_cost_gaps)):
        lane = 1
        if seed_index % 2 == 1:
            lane = 2
        seed_positions.append(model_row + lane * row_offset)

    axis.scatter(
        ordered_cost_gaps,
        seed_positions,
        marker=marker,
        s=17,
        facecolors="none",
        edgecolors=color,
        linewidths=0.8,
        zorder=3,
    )


def draw_model_cost_gaps(
    axis: Axes,
    seed_rows: list[dict],
    model_name: str,
    row_offset: float,
) -> None:
    """draws one mean and one between-seed whisker per market scenario"""
    mean_cost_gaps = []
    seed_standard_deviations = []
    model_rows = []

    for row_index, scenario in enumerate(SCENARIOS):
        seed_cost_gaps = get_seed_values(
            seed_rows, scenario, model_name, "minus_ac_bps"
        )
        model_row = row_index + row_offset

        model_rows.append(model_row)
        mean_cost_gaps.append(mean(seed_cost_gaps))
        seed_standard_deviations.append(stdev(seed_cost_gaps))

        if scenario == REDUCED_LIQUIDITY:
            draw_reduced_liquidity_seeds(
                axis, model_name, seed_cost_gaps, model_row, row_offset
            )

    color, marker = MODEL_STYLES[model_name]
    axis.errorbar(
        mean_cost_gaps,
        model_rows,
        xerr=seed_standard_deviations,
        label=model_name,
        fmt=marker,
        color=color,
        markersize=5.9,
        elinewidth=1.4,
        capsize=3,
        linewidth=0,
        zorder=4,
    )


@plt.rc_context(PLOT_STYLE)
def plot_robustness_gaps(
    seed_rows: list[dict],
    output_path: Path,
    dataset_label: str = "",
) -> None:
    """compares the cost of each frozen model with risk-neutral Almgren-Chriss"""
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    figure.subplots_adjust(left=0.235, right=0.97, top=0.75, bottom=0.22)
    add_figure_heading(figure, "Execution cost under market stress", dataset_label)
    shade_scenario_rows(axis)

    # dashed ref. marks equal mean execution cost
    axis.axvline(0, color=TEXT_COLOR, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    draw_model_cost_gaps(axis, seed_rows, BASIC_DQN, row_offset=-0.14)
    draw_model_cost_gaps(axis, seed_rows, LIQUIDITY_AWARE_DQN, row_offset=0.14)

    scenario_positions = list(range(len(SCENARIOS)))
    scenario_names = [SCENARIO_LABELS[scenario] for scenario in SCENARIOS]
    axis.set_yticks(scenario_positions, scenario_names)
    axis.set_ylim(len(SCENARIOS) - 0.5, -0.5)
    axis.margins(x=0.07)
    axis.set_xlabel(
        "Mean cost difference vs. risk-neutral Almgren–Chriss (bps)",
        labelpad=9,
    )
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    axis.tick_params(length=0, pad=6)

    add_legend(figure, axis, top=0.835)
    figure.text(
        0.045,
        0.035,
        f"Filled markers: means. Whiskers: ±1 SD across {len(TRAINING_SEEDS)} training seeds.\n"
        "Hollow markers: individual seeds under reduced liquidity. Positive gaps favour AC.",
        fontsize=9,
        color=SECONDARY_TEXT_COLOR,
    )
    save_figure(figure, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-label",
        default="",
        help="Dataset label above the title, for example '2026 held-out test'.",
    )
    arguments = parser.parse_args()

    seed_rows = read_csv_rows(arguments.input_dir / "seed_summary.csv")
    scenario_rows = read_csv_rows(arguments.input_dir / "scenario_summary.csv")
    seed_figure_path = arguments.output_dir / "nominal_training_seeds.png"
    stress_figure_path = arguments.output_dir / "robustness_cost_gaps.png"

    plot_nominal_training_seeds(
        seed_rows,
        scenario_rows,
        seed_figure_path,
        dataset_label=arguments.dataset_label,
    )
    plot_robustness_gaps(
        seed_rows, stress_figure_path, dataset_label=arguments.dataset_label
    )

    for figure_path in (seed_figure_path, stress_figure_path):
        print(f"Saved: {figure_path}")
        print(f"Saved: {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
