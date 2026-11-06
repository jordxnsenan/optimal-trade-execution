# Market Impact: Experimental Results

## Overview

I extended ExecAI's original execution-cost model to account for changing market liquidity and introduced a square-root alternative to the existing linear impact calculation.

The original model assumed that market liquidity stayed constant throughout the execution window. With the updated simulator, temporary impact changes depending on the relative liquidity available during each trading interval.

I used the intraday volume profile calculated from the 2023–2024 SPY training dataset to estimate liquidity during the 10:00–11:00 a.m. execution window. These values are forecasts based on historical training data, so the policies do not receive future realized trading volume from the evaluation sessions.

I also wanted to investigate how sensitive the execution strategies were to the impact model itself. For this, I compared linear and square-root impact using several impact coefficients.

## Experimental Setup

All experiments used the 250 SPY trading sessions from the 2025 validation period. Each strategy executed the same 10,000-share sell order over twelve five-minute intervals.

The experiments compared Immediate Execution, TWAP, forecast-volume VWAP, and the risk-neutral and risk-sensitive Almgren–Chriss optimizers.

I used impact coefficients of 10, 20, and 40 basis points. The half-spread and transaction-fee settings remained unchanged.

The Almgren–Chriss schedules were recalculated for each impact configuration so that the optimizer and simulator used the same assumptions.

## Results

The following table shows the mean implementation shortfall under forecast liquidity.

| Impact Model | Strength | Immediate |    TWAP |    VWAP | AC Neutral | AC Sensitive |
| ------------ | -------: | --------: | ------: | ------: | ---------: | -----------: |
| Linear       |       10 |   12.7477 |  4.7636 |  4.7599 |     4.7493 |       6.8372 |
| Linear       |       20 |   20.3332 |  5.6426 |  5.6461 |     5.6250 |       7.8583 |
| Linear       |       40 |   35.5041 |  7.4007 |  7.4185 |     7.3765 |       9.4373 |
| Square-root  |       10 |   13.8717 |  6.8322 |  6.8306 |     6.8168 |       9.3732 |
| Square-root  |       20 |   22.5812 |  9.7799 |  9.7876 |     9.7602 |      12.1000 |
| Square-root  |       40 |   40.0001 | 15.6752 | 15.7014 |    15.6469 |      16.9109 |

## Discussion

The first thing I noticed was how much the impact assumptions affected the cost of immediate execution. As the coefficient increased, selling the entire order at once became considerably more expensive.

The strategies that distributed their trades across the execution window were less sensitive to increases in linear impact.

The square-root model generally produced higher execution costs for distributed schedules under the same coefficient. This difference was expected because the square-root function imposes greater impact on many of the smaller trades in our execution problem.

However, using identical coefficients does not make the two models economically equivalent. The results show how policy performance changes under different hypothetical impact assumptions.

Risk-neutral Almgren–Chriss achieved the lowest average execution cost in every configuration. It was able to account for forecast liquidity when selecting its schedule. The improvement over TWAP was small, so I would need statistical analysis before drawing stronger conclusions about its advantage.

The risk-sensitive optimizer produced higher average shortfall but reduced inventory exposure. Its behavior reflects the different objective it was designed to minimize.

These experiments also highlight an important limitation of the current simulator. The liquidity estimates come from historical volume patterns, while the hypothetical impact of our trades is still calculated using an assumed model. We do not have direct observations of how these particular trades would have affected market prices.

## Next Steps

The next stage of the project will introduce the Deep Q-Network agent. Once it has been trained, I plan to evaluate its learned policy under different impact settings without changing the policy during testing.

This will help determine whether the agent has learned a useful execution strategy or whether its performance depends heavily on the specific simulation assumptions used during training.

The final evaluation will also include paired confidence intervals and separate historical test data that has not been used during development.
