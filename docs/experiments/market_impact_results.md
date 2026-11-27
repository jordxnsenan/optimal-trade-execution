# Market Impact and Liquidity: Experimental Results

## Overview

After implementing the initial execution strategies I wanted to investigate how their performance would change under different market-impact assumptions.

The original simulator assumed constant liquidity throughout the execution window. I extended it to support changing relative liquidity, allowing the cost of a trade to depend on how much liquidity was available at that point in time.

I used the intraday volume profile calculated from the 2023–2024 SPY training data to estimate liquidity during the 10:00–11:00 a.m. execution window. These estimates were based on historical trading patterns so the strategies did not have access to future realized volume from the evaluation sessions.

I also implemented a square-root market-impact model alongside the existing linear model. I wanted to see how much the choice of impact function affected execution costs and whether the relative performance of the traditional strategies would change.

## Experimental Setup

I evaluated the strategies using 250 historical SPY trading sessions from the 2025 validation period. Every strategy executed the same 10,000-share sell order over twelve five-minute intervals.

The comparison included Immediate Execution, TWAP, forecast-volume VWAP, as well as the risk-neutral and risk-sensitive Almgren–Chriss optimizers.

I tested both market-impact models using coefficients of 10, 20 and 40 basis points. The half-spread remained at 5 basis points and the transaction fee was $0.01 per share.

For each configuration, I recalculated the Almgren–Chriss schedules using the selected impact model and forecast liquidity. This kept the optimizer's assumptions consistent with those used by the simulator.

I evaluated every strategy on the same historical sessions so that differences in execution cost could be compared directly.

## Results

The table shows the mean implementation shortfall under forecast liquidity. All values are in basis points where a lower value indicates a better execution result.

| Impact Model | Coefficient (bps) | Immediate |    TWAP |    VWAP | AC Neutral | AC Sensitive |
| ------------ | ----------------: | --------: | ------: | ------: | ---------: | -----------: |
| Linear       |                10 |   12.7477 |  4.7636 |  4.7599 |     4.7493 |       6.8372 |
| Linear       |                20 |   20.3332 |  5.6426 |  5.6461 |     5.6250 |       7.8583 |
| Linear       |                40 |   35.5041 |  7.4007 |  7.4185 |     7.3765 |       9.4373 |
| Square-root  |                10 |   13.8717 |  6.8322 |  6.8306 |     6.8168 |       9.3732 |
| Square-root  |                20 |   22.5812 |  9.7799 |  9.7876 |     9.7602 |      12.1000 |
| Square-root  |                40 |   40.0001 | 15.6752 | 15.7014 |    15.6469 |      16.9109 |

## Discussion

The first thing I noticed was how strongly the impact coefficient affected Immediate Execution. Under the linear model, increasing the coefficient from 10 to 40 basis points raised its average shortfall from 12.7477 to 35.5041 bps.

This was expected because Immediate Execution sells the entire order at once. The simulator assigns a larger temporary-impact cost when a greater fraction of the order is executed in a single trade.

The strategies that spread the order across the execution window were less affected. Under the same linear-impact settings, TWAP's average shortfall increased from 4.7636 to 7.4007 bps.

The square-root model produced higher costs for the distributed strategies when I used the same numerical impact coefficient. For example, at a coefficient of 20 basis points, TWAP's shortfall increased from 5.6426 bps under linear impact to 9.7799 bps under square-root impact.

This difference comes from how the two functions calculate temporary impact. The square-root model assigns relatively greater impact to smaller trades under these settings, which changes the cost of spreading execution across multiple intervals.

I also had to be careful when interpreting these results. Using the same coefficient in two different impact functions does not mean they represent equivalent real-world trading conditions. The experiment measures sensitivity to the assumptions I selected for the simulator.

### Comparing the Execution Strategies

Risk-neutral Almgren–Chriss achieved the lowest average shortfall in every configuration. Its advantage over TWAP was small. Under the nominal linear-impact coefficient of 20 basis points, Almgren–Chriss averaged 5.6250 bps compared with 5.6426 bps for TWAP, representing a difference of approximately 0.018 bps.

The optimizer could account for the forecast liquidity profile when constructing its schedule. TWAP continued distributing the order according to time, regardless of the estimated liquidity available in each interval.

The risk-sensitive Almgren–Chriss optimizer consistently produced a higher average shortfall. Its objective included a penalty for holding inventory which encouraged earlier execution and reduced exposure to future price movements.

I kept the risk-sensitive strategy in the experiments because it represented a different execution objective. Its higher implementation shortfall alone does not capture the inventory risk it was designed to reduce.

### Limitations of the Impact Model

The liquidity estimates came from historical SPY trading volumes. They describe typical trading activity during the training period and do not measure the liquidity that would necessarily have been available for my simulated orders.

The market-impact models are also assumptions built into the simulator. Although the evaluation used real historical price movements, I did not observe how a hypothetical 10,000-share sell order would have affected those prices in the live market.

## How These Experiments Informed the DQN Evaluation

These experiments established the market-impact settings I later used to evaluate the trained DQN agents. After completing DQN training, I tested the saved policies under different impact coefficients and liquidity conditions. I also included a scenario where liquidity dropped during the later part of the execution window.

The DQN models were kept fixed throughout those evaluations. I wanted to examine how their learned decisions performed under changed conditions without allowing the agents to adapt through additional training.

I eventually repeated the evaluation on the held-out 2026 dataset using the finalized checkpoints and calculated paired confidence intervals for comparisons against TWAP and risk-neutral Almgren–Chriss.

The later results showed that the liquidity-aware DQN was particularly sensitive to reductions in liquidity. Basic DQN generally performed more consistently, although the traditional execution strategies still achieved lower average costs.

The DQN experiments are documented in [DQN Experimental Results](dqn_results.md) and [DQN Historical Validation](dqn_historical_validation.md). The final held-out results are also summarized in the project's main README.

## Reproducing the Experiments

The historical baseline experiments can be reproduced using `scripts/evaluation/evaluate_historical.py`.

The commands require the historical SPY dataset under `data/raw/SPY_5min_SIP/` and the market calibration file in `configs/spy_training_calibration.json`.

To run the nominal linear-impact experiment with forecast liquidity:

```bash
python -m scripts.evaluation.evaluate_historical \
  --liquidity-aware \
  --impact-model linear \
  --impact-bps 20
```

For the corresponding square-root experiment:

```bash
python -m scripts.evaluation.evaluate_historical \
  --liquidity-aware \
  --impact-model square_root \
  --impact-bps 20
```

The impact coefficient can be changed to reproduce the other configurations. For example:

```bash
python -m scripts.evaluation.evaluate_historical \
  --liquidity-aware \
  --impact-model linear \
  --impact-bps 40
```

The script reports mean implementation shortfall for the classical strategies, along with paired cost differences and the Almgren–Chriss inventory-risk calculations.

Daily evaluation results are saved under `results/validation/` using filenames that distinguish the impact model, liquidity setting and coefficient.