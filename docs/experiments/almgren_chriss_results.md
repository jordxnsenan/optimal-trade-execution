# Almgren–Chriss: Experimental Results

## Overview

I implemented a discrete version of Almgren–Chriss optimal execution to provide a stronger baseline for the reinforcement learning agents. The optimizer uses dynamic programming to find execution schedules that minimize expected trading costs with an optional penalty for holding inventory while market prices change.

The implementation uses the same transaction-cost model and inventory constraints as the existing simulator. I also verified the optimizer against exhaustive search in a small environment.

## Historical Evaluation

I evaluated the risk-neutral and risk-sensitive schedules using 250 historical SPY trading sessions from the 2025 validation period. The risk-neutral optimizer produced the same schedule as TWAP under the current constant-liquidity model. Both strategies achieved a mean implementation shortfall of 5.6848 basis points. The risk-sensitive optimizer used a risk-aversion parameter of 0.05 and executed a larger portion of the order earlier.

| Metric                                  | Risk-Neutral  | Risk-Sensitive |
| --------------------------------------- | ------------- | -------------- |
| Expected execution cost                 | 7.8000 bps    | 9.6000 bps     |
| Inventory risk                          | 149.4818 bps² | 49.9139 bps²   |
| Risk-adjusted objective                 | 15.2741 bps   | 12.0957 bps    |
| Historical mean shortfall               | 5.6848 bps    | 7.9720 bps     |
| Historical shortfall standard deviation | 20.5993 bps   | 12.9355 bps    |