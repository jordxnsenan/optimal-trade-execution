# Historical Market Data

## Data Source

ExecAI uses SPY market data attained from Alpaca's Historical Market Data API. I selected Alpaca because it provides five-minute price and volume data through its SIP feed which is suitable for the planned execution simulations.

The intended use of the data is to study historical price movements and estimate market characteristics. It will also be used to evaluate execution strategies on market paths that were not used during training.

## Data Access Verification

I tested the Alpaca API using the following three historical trading sessions:

| Date              | Bars Returned | Result     |
|-------------------|---------------|------------|
| February 15, 2023 | 78            | Successful |
| July 17, 2024     | 78            | Successful |
| October 15, 2025  | 78            | Successful |

Each request returned the expected 78 five-minute bars for a regular US trading session. The timestamps also matched the expected market opening times. These checks verify that historical SPY data is accessible across the years I plan to use.

## Data Format

The requests use the following settings:

| Parameter        | Value                         |
|------------------|-------------------------------|
| Symbol           | SPY                           |
| Timeframe        | 5 minutes                     |
| Feed             | SIP                           |
| Price adjustment | Raw                           |
| Market session   | 9:30 a.m. – 4:00 p.m. Eastern |
| Timestamp format | UTC                           |

The main fields are:
- The timestamp
- Open price
- High price
- Low price
- Close price
- Trading volume


## Historical Dataset Coverage

The historical SPY dataset was collected using Alpaca's SIP market data feed. It contains five-minute OHLCV observations from January 2023 through September 2026.

The data is divided chronologically into the following separate periods:

| Partition  | Period                 | Trading Sessions | Five-Minute Bars |
|------------|------------------------|------------------|------------------|
| Training   | 2023–2024              | 502              | 38,976           |
| Validation | 2025 | 250             | 19,392           |
| Final Test | January–September 2026 | 187              | 14,586           |
| Total      |                        | 939              | 72,954           |

All three periods achieved 100% coverage against the expected NYSE trading calendar, with the validation process also accounting for scheduled early market closures.

Each trading session is stored as an individual CSV checkpoint. The ingestion pipeline verifies existing checkpoints before making additional API requests to allow interrupted downloads to resume. The raw market data is stored locally under `data/raw/SPY_5min_SIP/`
and is excluded from Git.

### Data Separation

The 2023–2024 training period will be used to estimate market characteristics and train the reinforcement learning agents. The 2025 validation period will support model development and hyperparameter selection. The January–September 2026 period is reserved for final testing. Model performance on this period will not be examined until the
experimental configuration has been finalized. Historical prices and volumes will be used to inform market simulation and replay. Transaction costs and the effect of our hypothetical trades will continue to be modeled separately.