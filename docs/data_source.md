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
- Trading volume.