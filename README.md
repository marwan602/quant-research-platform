# S&P 500 Quantitative Research Platform

Research platform for predicting 5-day forward returns on the S&P 500 using daily price and volume data.

## Data

The dataset covers daily prices and index constituents from 2015 through August 2026. It was generated using the pipeline from [K0D1Z/sp500-quantitative-dataset](https://github.com/K0D1Z/sp500-quantitative-dataset):

- Daily S&P 500 composition is reconstructed from Wikipedia change records so delisted and acquired companies are kept during their active years (avoiding survivorship bias).
- Price data comes from Yahoo Finance, with missing and delisted stocks backfilled using the Tiingo API.

Files in `data/raw/`:
- `s_and_p_500_daily_composition.parquet`: Which stocks were in the index on each date.
- `s_and_p_500_prices.parquet`: Daily OHLCV price history.

## Project Layout

- `data/`: Raw price/composition files and processed feature tables.
- `src/`: Feature generation, dataset loaders, models, and evaluation.
- `configs/`: Split dates and model parameters.
