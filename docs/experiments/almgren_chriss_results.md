# Almgren–Chriss: Experimental Results

## Overview

After implementing the basic execution strategies, I wanted a stronger benchmark for evaluating the reinforcement learning agents. I implemented a discrete version of Almgren–Chriss optimal execution using dynamic programming.

The optimizer calculates how much inventory to sell at each decision while accounting for the expected execution cost. I also included an optional penalty for holding shares over time since delaying execution exposes the remaining order to changes in market prices.

I used the same execution-cost model and inventory constraints as the trading simulator. Before running the historical experiments, I checked the optimizer against exhaustive search in a smaller environment to verify that it could identify the optimal execution schedule.

I implemented two configurations:

- **Risk-neutral:** Minimizes expected execution cost without penalizing inventory risk.
- **Risk-sensitive:** Includes an inventory-risk penalty, encouraging the strategy to complete more of the order earlier.

The risk-sensitive configuration used a risk-aversion parameter of 0.05.

## Historical Evaluation

I evaluated both configurations on 250 historical SPY trading sessions from the 2025 validation period. Each strategy executed a 10,000-share sell order across twelve five-minute intervals.

For this initial experiment, I assumed constant liquidity throughout the execution window. Under that assumption the risk-neutral optimizer produced the same schedule as TWAP. Both achieved an average implementation shortfall of **5.6848 basis points**.

The risk-sensitive optimizer executed a greater portion of the order earlier. This increased its expected execution cost but reduced the amount of inventory exposed to future price movements.

The results were:

| Metric                                  |  Risk-Neutral | Risk-Sensitive |
| --------------------------------------- | ------------: | -------------: |
| Expected execution cost                 |    7.8000 bps |     9.6000 bps |
| Inventory risk                          | 149.4818 bps² |   49.9139 bps² |
| Risk-adjusted objective                 |   15.2741 bps |    12.0957 bps |
| Historical mean shortfall               |    5.6848 bps |     7.9720 bps |
| Historical shortfall standard deviation |   20.5993 bps |    12.9355 bps |

The risk-adjusted objective combines expected execution cost with the inventory-risk penalty. I used the same risk-aversion value of 0.05 when comparing the two schedules under this objective.

## Discussion

The results showed the trade-off between minimizing expected trading costs and reducing inventory exposure.

The risk-neutral optimizer achieved the lower average historical shortfall. Its schedule spread execution across the available intervals, keeping temporary market impact relatively low.

The risk-sensitive optimizer had a higher average shortfall of 7.9720 bps. However, its inventory-risk measure was considerably lower, and the standard deviation of its historical execution costs decreased from 20.5993 to 12.9355 bps.

I found this comparison useful because it showed how changing the optimization objective could produce a different trading schedule even when both strategies used the same market data and execution constraints.

These results belong to the initial constant-liquidity experiments. I later extended the simulator to account for changing relative liquidity using volume estimates from the 2023–2024 training dataset. Under those assumptions the risk-neutral Almgren–Chriss optimizer could adjust its schedule based on forecast liquidity.

The later experiments are documented in [Market Impact and Liquidity Results](market_impact_results.md).

## Reproducing the Experiment

The historical baseline evaluation can be run from the repository root using:

```bash
python -m scripts.evaluation.evaluate_historical
```

By default, this command uses the linear market-impact model with constant reference liquidity. It evaluates the execution strategies on the 2025 validation dataset and reports the Almgren–Chriss risk analysis.

The experiment requires the historical SPY data stored locally under `data/raw/SPY_5min_SIP/` and the calibration file in `configs/spy_training_calibration.json`.