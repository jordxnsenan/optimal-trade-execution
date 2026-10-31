# ExecAI

## Optimal Trade Execution Using Reinforcement Learning

ExecAI is a research project investigating how reinforcement learning can
complete a fixed stock order within a limited time while controlling
execution cost.

The system will simulate trade execution, train execution policies and
compare their performance with traditional scheduling and optimization
methods under common market scenarios.

## Project status

Planning and initial development.

The methods and experiments laid out below are planned. I will document implementation progress and results as the project develops.

## Research question

Under which market and liquidity assumptions can reinforcement learning
reduce execution cost relative to strong traditional execution policies,
and how stable is that conclusion when those assumptions change?

## Initial scope

- Execute a fixed sell order in SPY
- Use twelve five-minute decision slots
- Represent the parent order as twenty equal inventory lots
- Evaluate policies using implementation shortfall and a separate
  risk-sensitive objective
- Use chronological data partitions and causal market features
- Investigate sensitivity to market-impact assumptions

## Planned methods

The learning methods are tabular Q-learning and a Deep Q-Network.

The comparison policies are immediate execution, TWAP, forecast-volume
VWAP, Almgren-Chris and an adaptive linear-cost optimizer.

All methods will use common execution accounting and declared constraints.

## Evaluation approach

Evaluation will use shared market scenarios, independent training seeds, paired cost comparisons and confidence intervals.

Historical price-path experiments will use explicitly modeled execution
costs. Simulator assumptions and limitations will be documented alongside the findings.


## Development

### Requirements

- Python 3.10 or newer
- Git

The initial development environment was verified using Python 3.14.3 on macOS.

### Setting Up the Project

Clone the repository and navigate to its root directory.

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install ExecAI in editable mode:

```bash
python -m pip install -e .
```

### Running Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

### Project Structure

- `src/execai/` - Main Python package
- `tests/` - Automated tests
- `docs/` - Project design and research documentation
- `.github/workflows/` - Continuous integration configuration

### Documentation
- [Q-learning experimental results](docs/experiments/q_learning_results.md)