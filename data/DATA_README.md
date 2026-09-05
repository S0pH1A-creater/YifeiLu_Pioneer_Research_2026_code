# Data layout

See [`../docs/DataCollection.md`](../docs/DataCollection.md).

Tickers: SPY, AAPL, MSFT, AMZN.

```
data/
  equity/                 # daily adj close and log returns
  equity/short_interval/  # 1-minute RTH (5-day hourly study)
  rates/                  # FRED DGS3MO
  options/processed/
  options/processed/short_interval/
```
