# DQN: Initial Experimental Results

## Overview

After experimenting with tabular Q-learning I moved on to Deep Q-Networks (DQN). I wanted to see whether a neural network could learn better execution decisions on the full-sized trading problem. The Q-learning experiments had shown that the agent could find an optimal schedule in a small deterministic environment but it struggled when the number of decisions increased. Some agents learned to wait until late in the execution window before selling large amounts of inventory.

I began the DQN work with the same deterministic benchmarks so I could check whether the new implementation was learning correctly. I then introduced synthetic price movements and compared the learned policies against traditional execution strategies.

## DQN Implementation

I implemented DQN using PyTorch with a neural network that estimates the value of each available trading action. 

The initial agent observes the current time step and how many inventory lots remain. Its actions represent the number of lots to sell during the current interval. I used experience replay to store transitions collected during training. The network learns from batches of these experiences while a separate target network provides more stable estimates when calculating the learning targets.During training, the agent uses epsilon-greedy exploration. The exploration rate gradually decreases as the agent completes more episodes.

I also implemented action masking to prevent the network from selecting trades that exceed the remaining inventory. At the final decision, the agent must execute whatever shares are left.

I started with constant-price environments because their optimal execution schedules could be calculated independently. This made it easier to identify whether poor performance came from the learning algorithm or from the difficulty of predicting changing market prices.

## Small Deterministic Benchmark

The first benchmark used a 400-share sell order divided into four inventory lots with three decisions available to complete execution. The market price remained constant at $100. Under the simulator's linear market-impact assumptions, the known optimal implementation shortfall was **13.5 basis points**.

I trained four DQN agents for 2,000 episodes each, using different random seeds.

| Training Seed | First 100 Episodes | Last 100 Episodes | Greedy Evaluation |
| ------------- | -----------------: | ----------------: | ----------------: |
| 42            |            19.4750 |           14.2250 |           13.5000 |
| 7             |            19.1750 |           14.2500 |           13.5000 |
| 123           |            19.1250 |           14.2750 |           13.5000 |
| 2026          |            19.7000 |           13.9750 |           13.5000 |

*All values are implementation shortfall in basis points.*

All four agents learned the same optimal execution schedule of `[1, 1, 2]` inventory lots corresponding to trades of 100, 100 and 200 shares.

The training costs remained slightly above the optimum because the agents continued taking exploratory actions. I evaluated each learned policy separately with exploration disabled. This was a useful first result. The DQN implementation was able to learn the known optimal schedule consistently across the four training seeds. I then wanted to check whether it could do the same thing with the larger inventory and action space.

## Full-Sized Deterministic Benchmark

I repeated the deterministic experiment using the main execution configuration:

| Parameter          |         Value |
| ------------------ | ------------: |
| Order size         | 10,000 shares |
| Inventory lots     |            20 |
| Shares per lot     |           500 |
| Decision intervals |            12 |
| Starting price     |          $100 |

Each interval represents five minutes, giving the agent one hour to sell the entire order.

I trained DQN for 2,000 episodes using random seed 42. The average execution cost during the first 100 training episodes was **16.3920 bps**. During the last 100 episodes, it decreased to **9.2340 bps**. 

When I evaluated the learned policy without exploration, it achieved **7.8 bps**, matching TWAP and the independently established optimum for this constant-price environment.

This showed that the network could learn an optimal execution schedule on the full-sized problem. The next question was how well those decisions would perform when market prices changed between intervals.

## Stochastic Synthetic Market Evaluation

I introduced synthetic price paths generated using the volatility estimated from the 2023–2024 SPY training dataset. The experiment used the full 10,000-share order, with twelve decisions and 20 inventory lots. I kept market liquidity constant and used the linear temporary-impact model.

I trained DQN for 2,000 episodes, then evaluated its greedy policy on 250 new synthetic price paths. The comparison strategies were evaluated on the same paths.

### Results

| Execution Strategy              | Mean Shortfall (bps) |
| ------------------------------- | -------------------: |
| Immediate Execution             |              26.0000 |
| TWAP                            |               6.9296 |
| VWAP                            |               6.8811 |
| Almgren–Chriss (Risk-Neutral)   |               6.9296 |
| Almgren–Chriss (Risk-Sensitive) |               8.8593 |
| DQN                             |               7.1646 |

DQN performed well in this experiment although it did not achieve the lowest execution cost. Its mean shortfall was approximately **0.2351 bps higher than TWAP**. VWAP achieved the lowest average cost among the strategies tested.

I found the result encouraging because the agent had learned a useful schedule despite only observing the current time step and remaining inventory. It had no direct information about current prices or future price movements. However, this was one trained model evaluated on one collection of synthetic market scenarios. I wanted to investigate whether the result would remain similar with different training seeds.

## Comparing Independent Training Seeds

I repeated stochastic DQN training using four independent random seeds. Each model was evaluated on the same 250 synthetic price paths. This allowed me to compare the learned policies without introducing differences between their evaluation scenarios.

The results were:

| Training Seed | TWAP (bps) | DQN (bps) | DQN − TWAP (bps) |
| ------------- | ---------: | --------: | ---------------: |
| 42            |     7.1664 |    7.7040 |          +0.5375 |
| 7             |     7.1664 |   10.5545 |          +3.3881 |
| 123           |     7.1664 |    9.1991 |          +2.0327 |
| 2026          |     7.1664 |    7.3844 |          +0.2179 |

TWAP produced the same average cost in every row because it was evaluated on the same market scenarios.

The trained DQN models behaved differently. Seed 2026 came closest to matching TWAP, with an average disadvantage of approximately 0.22 bps. Seed 7 performed considerably worse, averaging 3.39 bps more than TWAP. All four DQN models were more expensive than TWAP.

I expected some variation between training seeds but the differences were larger than I wanted. A model that performed well in one training run did not necessarily represent what DQN would learn consistently. This made training stability an important part of the remaining project. I needed a way to select checkpoints based on their evaluation performance and compare more independent models before drawing conclusions about DQN's effectiveness.

## What I Learned from These Experiments

The deterministic benchmarks showed that DQN could learn optimal execution schedules in both the small and full-sized environments. Its performance under stochastic market conditions was less consistent. The first synthetic experiment produced a relatively small gap from TWAP, but the independent training-seed comparison showed that some agents learned substantially worse schedules.

The basic DQN also had a limited observation space. Its decisions depended on time and remaining inventory, which meant it could not directly account for periods when trading liquidity was higher or lower. These findings influenced the next stage of development as I decided to extend the agent to support a liquidity feature and introduced validation-based checkpoint selection during training. I also increased the number of independent training seeds to eight for the later experiments.

The resulting policies were evaluated on historical SPY sessions from 2025, followed by a separate held-out evaluation using 2026 data. I also tested how the frozen models performed when market-impact assumptions and liquidity conditions changed.

Those later experiments are documented in [DQN Historical Validation](dqn_historical_validation.md) and summarized in the main [README](../../README.md).

## Reproducing the Experiments

The DQN training scripts are located under `scripts/training/`.

I developed these experiments using Python 3.14 on macOS. From the repository root, install the project in a virtual environment before running the commands.

### Small Deterministic Benchmark

To reproduce the small benchmark using seed 42:

```bash
python -m scripts.training.benchmark_dqn \
  --episodes 2000 \
  --seed 42
```

The script trains a DQN on the 400-share problem and compares its learned schedule against Immediate Execution and TWAP.

### Full-Sized Deterministic Benchmark

Add the `--full-size` option to run the same experiment using 10,000 shares:

```bash
python -m scripts.training.benchmark_dqn \
  --full-size \
  --episodes 2000 \
  --seed 42
```

The known optimal execution cost is 7.8 bps.

### Stochastic Synthetic Market Experiment

The synthetic-market comparison uses volatility and the historical volume profile from `configs/spy_training_calibration.json`.

Run the experiment with:

```bash
python -m scripts.training.compare_dqn_synthetic \
  --episodes 2000 \
  --evaluation-episodes 250 \
  --seed 42
```

The script reports the DQN's average shortfall alongside the traditional execution strategies. It also calculates a paired confidence interval for the difference between DQN and TWAP.

To compare independent training runs, repeat the command using seeds `7`, `123` and `2026`. The script uses a common evaluation seed so the models are tested on the same synthetic market paths.

The experiments in this document were used during development. The final historical evaluation used separately trained checkpoints and a more extensive validation procedure.