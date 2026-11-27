# DQN Historical Validation Results

## Overview

After completing the initial synthetic market experiments I wanted to see how the trained DQN agents would perform on historical SPY price movements.

The earlier experiments showed that DQN could learn optimal execution schedules when market prices remained constant. Performance varied considerably between training seeds once I introduced stochastic prices.

I also wanted to investigate whether giving the agent information about available liquidity would improve its execution decisions. The basic DQN only observed time and remaining inventory so it had no way to distinguish between intervals with different liquidity conditions. I trained a second DQN configuration with an additional liquidity feature and compared both versions using historical market data.

## Training and Checkpoint Selection

Both DQN configurations were trained on synthetic market scenarios calibrated using the 2023–2024 SPY training dataset. The basic DQN observes the current time step and remaining inventory. The liquidity-aware version also receives the relative liquidity available during the current interval.

During training I varied liquidity between synthetic market episodes. Both configurations experienced the same market scenarios although only the liquidity-aware agent received liquidity as part of its observation.

I also introduced validation-based checkpoint selection. Earlier experiments had shown that the performance of a trained network could deteriorate as training continued so keeping the network from the final episode was not necessarily the best approach.

Each agent was trained for 2,000 episodes. Every 200 episodes I evaluated its greedy policy on a separate collection of 100 synthetic validation scenarios. I retained the checkpoint with the lowest average validation cost.

For the initial historical evaluation I trained both configurations using seeds 42, 7, 123 and 2026 which produced eight saved models.

## Historical Evaluation

I evaluated the saved models on 250 historical SPY trading sessions from 2025.

Each episode simulated a 10,000-share sell order between 10:00 and 11:00 a.m. Eastern, divided into twelve five-minute decisions.

The execution settings were:

| Parameter              | Value                     |
| ---------------------- | ------------------------- |
| Order size             | 10,000 shares             |
| Inventory lots         | 20                        |
| Decision intervals     | 12                        |
| Half-spread            | 5 bps                     |
| Temporary impact model | Linear                    |
| Impact coefficient     | 20 bps                    |
| Transaction fee        | $0.01 per share           |
| Liquidity              | Training-derived forecast |

The liquidity forecast came from the intraday volume profile calculated using the 2023–2024 data. I did not provide the agents with future realized trading volumes from the historical sessions. All strategies used the same execution simulator. I evaluated the saved DQN checkpoints with exploration disabled and made no further updates to their network weights.

## Results

The following table shows the average implementation shortfall for each trained model across the 250 historical sessions.

| Training Seed | Basic DQN (bps) | Liquidity-Aware DQN (bps) |
| ------------- | --------------: | ------------------------: |
| 42            |          6.3213 |                    9.2095 |
| 7             |          5.7589 |                    6.2834 |
| 123           |          6.0310 |                    7.7349 |
| 2026          |          6.9051 |                    5.7711 |
| **Mean**      |      **6.2541** |                **7.2497** |

For comparison, TWAP achieved an average shortfall of 5.6426 bps. Risk-neutral Almgren–Chriss achieved 5.6250 bps on the same sessions. The basic DQN performed better than the liquidity-aware version in three of the four training runs. It also showed less variation between independently trained models.

Seed 2026 was the exception as its liquidity-aware model achieved 5.7711 bps, considerably lower than the corresponding basic model's 6.9051 bps. Neither configuration consistently matched the traditional execution strategies.

## Discussion

I had expected the liquidity feature to help the agent identify better opportunities to execute its remaining inventory. The historical results did not support that expectation under the training conditions I used. A possible explanation was the difference between synthetic training and historical evaluation.

During training liquidity varied between market episodes. Historical evaluation used a fixed liquidity forecast derived from the training dataset. The liquidity-aware agent may have learned relationships between its observations and execution costs that did not transfer well to the historical setting.

The variation between training seeds also made it difficult to draw a strong conclusion from four independently trained models. Some liquidity-aware agents performed well while others learned considerably more expensive execution schedules.

Based on these initial results I decided to treat basic DQN as the primary reinforcement learning configuration for the remaining experiments. I retained the liquidity-aware models because I wanted to investigate their behaviour under different liquidity conditions.

## Extended Evaluation

I later increased the number of independent training seeds from four to eight, retaining the original models and adding seeds 314, 777, 1001 and 4096.

I evaluated all 16 saved checkpoints under six market scenarios using the 2025 validation dataset. These experiments examined how the trained policies responded to changes in market impact and available liquidity.

Under the nominal evaluation settings, the extended eight-seed results were:

| Execution Strategy          | Mean Shortfall (bps) |
| --------------------------- | -------------------: |
| TWAP                        |               5.6426 |
| Risk-neutral Almgren–Chriss |               5.6250 |
| Basic DQN                   |               6.2275 |
| Liquidity-aware DQN         |               6.9472 |

The additional training seeds reduced the average cost estimates for both DQN configurations. The basic DQN continued to outperform the liquidity-aware version on average while TWAP and risk-neutral Almgren–Chriss remained less expensive.

After finalizing the experimental settings, I audited all 16 checkpoints and evaluated the frozen policies on the separate 2026 dataset.

The final historical results are summarized in the project's [README](../../README.md).

## Limitations

The 2025 dataset was used throughout model development so I treated these results as validation findings. The January–September 2026 dataset was kept separate for the final evaluation.

The historical experiments replay observed SPY price movements but the trades themselves are simulated. The execution costs depend on the assumed spread, transaction fees and market-impact function.

The liquidity forecast also represents typical trading activity estimated from historical volume data. It cannot establish exactly how much liquidity would have been available for a particular order in the live market.

## Reproducing the Historical Evaluation

The historical evaluation requires the SPY dataset under `data/raw/SPY_5min_SIP/` and the saved DQN checkpoints under `results/models/`.

To evaluate the basic and liquidity-aware models trained using seed 42:

```bash id="v8k2m1"
python -m scripts.evaluation.evaluate_dqn_historical \
  --seed 42
```

The script loads both checkpoints and evaluates them alongside the traditional execution strategies on the 2025 validation sessions. It reports mean implementation shortfall and paired cost differences, including 95% bootstrap confidence intervals.

Repeat the command using seeds `7`, `123`, and `2026` to evaluate the remaining models from the initial four-seed experiment.

Once all 16 checkpoints are available, the extended eight-seed evaluation can be run using:

```bash id="r4p7n2"
python -m scripts.evaluation.evaluate_dqn_final \
  --split validation
```

This evaluates the frozen policies across all six market scenarios and saves the results to `results/final/2025_preflight.csv`.

The original model checkpoints and raw historical market data are stored locally and are excluded from Git. Instructions for preparing the data and training new checkpoints are available in the main README.