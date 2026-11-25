# Tabular Q-Learning: Experimental Results

## Overview

I started the reinforcement learning implementation with tabular Q-learning. The goal was to train an agent to decide how much of a stock order to sell at each interval while minimizing execution costs.

I wanted to establish that the learning algorithm worked on a problem with a known optimal solution before attempting the full 10,000-share execution task. I also wanted to see how well Q-learning would handle a longer execution window where the consequences of an early trading decision might only become apparent several steps later.

I compared the learned policies against Immediate Execution and Time-Weighted Average Price (TWAP). Both strategies use the same simulator and execution-cost assumptions as the Q-learning agent.

The experiments documented here were conducted during the early development stage. They use synthetic market scenarios and are separate from the later DQN experiments and historical evaluations.

## Experimental Setup

### Execution Environment

I used two execution configurations:

| Parameter          | Small Benchmark | Full-Size Problem |
| ------------------ | --------------: | ----------------: |
| Order size         |      400 shares |     10,000 shares |
| Decision intervals |               3 |                12 |
| Inventory lots     |               4 |                20 |
| Shares per lot     |             100 |               500 |
| Starting price     |            $100 |              $100 |

In the full-sized environment, each decision represents a five-minute interval, giving the agent one hour to complete the order.

The agent selects how many inventory lots to sell at each step. It can choose to wait, but it cannot sell more shares than it has remaining. Any inventory left at the final decision must be executed.

The simulator includes a transaction-cost model with the following settings:

| Parameter                    |           Value |
| ---------------------------- | --------------: |
| Half-spread                  |  5 basis points |
| Temporary impact coefficient | 20 basis points |
| Transaction fee              | $0.01 per share |

Larger trades incur greater temporary market impact. When prices remain constant, distributing the order across multiple intervals generally reduces execution costs.

I measured performance using implementation shortfall in basis points. This measures the difference between the order's value at the initial market price and the net proceeds received after execution. Lower shortfall indicates better performance.

### Q-Learning Implementation

I represented each state using the current time step and the number of inventory lots remaining.

The Q-table stores the estimated value of selling a particular quantity from each state. I implemented it using Python dictionaries with Q-values initially set to zero. During training, the agent uses epsilon-greedy action selection to explore different trades while gradually relying more on its learned estimates.

The initial training settings were:

| Hyperparameter           | Value  |
| ------------------------ | ------ |
| Learning rate            | 0.1    |
| Discount factor          | 1.0    |
| Initial exploration rate | 1.0    |
| Final exploration rate   | 0.1    |
| Exploration decay        | Linear |

I used a discount factor of 1.0 because the objective is to minimize the total execution cost over a fixed number of decisions. The state representation was deliberately simple. The agent could learn how to distribute inventory over time but it could not directly respond to changes in market prices or volatility.

For the stochastic experiments, I generated synthetic price paths using a seeded geometric Brownian motion model. Training and evaluation used separate paths, with every strategy evaluated on the same market scenarios.

## Small Deterministic Benchmark

I began with the 400-share execution problem. The order was divided into four lots with three decisions available. Market prices remained constant at $100.

Before training Q-learning, I used exhaustive search to calculate the execution cost of every valid schedule. The minimum was **13.5 basis points**, with several different schedules achieving that result. I trained the agent for 5,000 episodes using random seed 42.

### Results

| Strategy            | Execution Schedule       | Shortfall |
| ------------------- | ------------------------ | --------: |
| Immediate Execution | [400]                    |  26.0 bps |
| TWAP                | [200, 100, 100]          |  13.5 bps |
| Q-Learning          | [100, 100, 200]          |  13.5 bps |
| Known optimum       | Multiple valid schedules |  13.5 bps |

Q-learning matched the independently calculated optimal cost. Its schedule differed from TWAP but both distributed the order efficiently enough to achieve the same result.

This gave me a useful check of the learning implementation. The agent had learned an optimal schedule through repeated interaction with the simulator and the Q-value updates were also passing their unit tests.

The constant-price environment was relatively predictable. I continued with the full-sized problem to investigate whether the same approach could learn effective decisions over a longer execution window.

## Full-Size Q-Learning Experiments

### Initial Training

I increased the order size to 10,000 shares which was divided into 20 lots over twelve decisions.

The larger problem introduced many more possible state-action combinations. A decision to wait also meant that the agent had less time to sell its remaining inventory, potentially increasing the cost of later trades.

I initially trained five agents for 5,000 episodes each using different random seeds. The experiments used synthetic price paths with a volatility setting of 0.002 per time step. All five agents were evaluated on the same 500 validation paths.

TWAP achieved an average implementation shortfall of **8.2047 bps**, while Immediate Execution cost 26.0 bps.

The Q-learning agents performed considerably worse than TWAP. Several learned schedules involved waiting through much of the execution window before selling large portions of the order. These concentrated trades incurred higher temporary market-impact costs.

### Increasing the Training Budget

I repeated the experiment using 20,000 training episodes per agent to see whether additional experience would improve the learned schedules.

| Training Episodes | Mean Difference from TWAP |
| ----------------- | ------------------------: |
| 5,000             |               +8.5917 bps |
| 20,000            |               +8.0659 bps |

The additional training produced a small improvement but the agents continued to learn expensive schedules. There was also substantial variation between training seeds.

One agent trained for 20,000 episodes learned to sell 9,500 of its 10,000 shares during a single interval. On a constant-price reference path, this schedule cost 24.1 bps, compared with the known full-sized optimum of 7.8 bps.

Increasing the training budget alone had not resolved the problem, so I investigated whether random price movements were contributing to the difficulty.

### Training Without Price Volatility

I repeated the 20,000-episode experiment with training volatility set to zero. Every training episode used a constant market price of $100.

I kept the original stochastic validation paths which used a volatility setting of 0.002. This allowed me to compare policies learned under simpler conditions against the same evaluation scenarios.

Removing volatility improved performance, but none of the five agents reached the known optimal cost on the flat-market reference path.

| Agent Seed | Flat-Market Cost | Difference from Optimum |
| ---------- | ---------------: | ----------------------: |
| 42         |          9.3 bps |                +1.5 bps |
| 43         |         11.9 bps |                +4.1 bps |
| 44         |         12.7 bps |                +4.9 bps |
| 45         |         13.7 bps |                +5.9 bps |
| 46         |         15.8 bps |                +8.0 bps |

Seed 42 produced the best schedule but it still included some relatively large trades.

The improvement suggested that price variation was contributing to the learning difficulty. The remaining gap from the known optimum led me to investigate how frequently different actions were being updated.

## Investigating the Q-Table

I added counters to record how often each state-action pair received a Q-learning update.

Before adding these diagnostics, I had primarily been checking the number of states visited during training. Most runs had visited approximately 220 or 221 states which initially suggested that the agent was exploring much of the available state space.

In the 20,000-episode deterministic experiment, all 2,341 recorded state-action pairs had received at least one update but It was apparent that the number of updates varied considerably between actions.

At the initial state, selling zero lots received roughly 9,000 updates whereas selling two lots received only around 600.

I also found that the estimated Q-values for waiting were much higher than their actual optimal values.

For seed 42, the estimated Q-value for selling zero lots at the initial decision was approximately **−1.60**. The best possible total reward after taking that action in the constant-price environment was −7.9.

The Q-value for immediately selling the entire order was correctly estimated at −26.0.

The difference made sense when I considered how the rewards were received. Immediate execution ends the episode and gives the agent the complete cost of its decision. Waiting produces an immediate reward of zero while the cost of completing the order occurs during later decisions. The agent needs repeated updates to learn how those future costs affect the value of waiting. In these experiments the estimates were not consistently capturing those costs.

I also noticed a potential issue with action selection. Because the Q-values initially started at zero, several actions could have identical estimated values. The original greedy action-selection method always chose the smallest action when values were tied. In this environment, that often meant selling nothing.

I decided to test whether removing that preference would improve learning.

## Randomized Tie-Breaking

I modified the Q-learning agent to support randomized greedy tie-breaking. When multiple actions have the same highest Q-value the agent can randomly choose among them. I kept the original behaviour available so I could compare both approaches under the same training settings.

For this experiment, I trained five agents for 20,000 episodes each with zero price volatility. The policies were evaluated on the same 500 stochastic validation paths.

| Agent Seed  | Original Difference from TWAP | Randomized Tie-Breaking |
| ----------- | ----------------------------: | ----------------------: |
| 42          |                   +1.9412 bps |             +4.5645 bps |
| 43          |                   +4.2885 bps |             +1.2646 bps |
| 44          |                   +5.0068 bps |             +1.7981 bps |
| 45          |                   +6.2274 bps |             +2.7027 bps |
| 46          |                   +7.9187 bps |             +1.9077 bps |
| **Average** |               **+5.0765 bps** |         **+2.4475 bps** |

Randomized tie-breaking reduced the average disadvantage relative to TWAP by approximately **2.63 bps**.

Four of the five agents improved. Their learned schedules generally distributed trades across more intervals, reducing the large executions that had appeared in earlier experiments.

Seed 42 performed worse with randomized tie-breaking, so the improvement was not consistent across every run.

I found this result interesting because the Q-learning update equation remained unchanged. The difference came from how the agent selected between actions with identical estimated values.

## Findings

Tabular Q-learning successfully learned the known optimal schedule in the small deterministic benchmark. Its performance was much less reliable on the full-sized execution problem.

The Q-table diagnostics helped explain some of the difficulty. Many state-action pairs were updated infrequently and the agent often overestimated the value of waiting. The longer execution window made it difficult to account for costs that occurred several decisions after an action was taken.

Randomized tie-breaking improved the average result but the five trained agents still performed worse than TWAP.

These experiments motivated my move to Deep Q-Networks. I wanted to investigate whether a neural network could learn more useful estimates of future execution costs and produce more consistent schedules on the full-sized problem.

The subsequent DQN development and evaluation are documented separately in [DQN Experimental Results](dqn_results.md).

## Reproducing the Experiments

The Q-learning scripts are available under `scripts/training/` and `scripts/evaluation/`.

I developed the experiments using Python 3.14.3 on macOS.

From the repository root, create and activate a virtual environment, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To run the automated tests:

```bash
python -m unittest discover -s tests -v
```

### Small Deterministic Benchmark

Run the original 400-share experiment:

```bash
python -m scripts.training.compare_q_learning
```

This trains Q-learning for 5,000 episodes and compares its learned execution schedule against TWAP and Immediate Execution. The known optimal cost is 13.5 bps.

### Full-Size Experiments

Run the original stochastic experiment with 5,000 training episodes:

```bash
python -m scripts.evaluation.evaluate_q_learning \
  --full-size \
  --training-episodes 5000
```

Increase the training budget to 20,000 episodes:

```bash
python -m scripts.evaluation.evaluate_q_learning \
  --full-size \
  --training-episodes 20000
```

Repeat the experiment with constant prices during training:

```bash
python -m scripts.evaluation.evaluate_q_learning \
  --full-size \
  --training-episodes 20000 \
  --training-volatility 0
```

Enable randomized greedy tie-breaking:

```bash
python -m scripts.evaluation.evaluate_q_learning \
  --full-size \
  --training-episodes 20000 \
  --training-volatility 0 \
  --random-tie-breaking
```

The evaluation script trains five agents using seeds 42 through 46. It reports each agent's learned schedule, its cost on a constant-price reference path, as well as its average difference from TWAP across 500 shared stochastic validation episodes.