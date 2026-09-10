from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.providers.base import LocalProvider, MarketDataProvider, UniverseProvider
from src.providers.store import RollingPriceStore


def test_universe_provider(tmp_path):
    comp_file = tmp_path / "comp.parquet"
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-01", "2026-08-01", "2026-08-31", "2026-08-31"]),
        "Ticker": ["AAPL", "MSFT", "AAPL", "NVDA"],
    })
    df.to_parquet(comp_file, index=False)

    provider = UniverseProvider(composition_path=str(comp_file))
    tickers_aug31 = provider.get_active_tickers("2026-08-31")
    assert sorted(tickers_aug31) == ["AAPL", "NVDA"]

    tickers_future = provider.get_active_tickers("2026-09-10")
    assert sorted(tickers_future) == ["AAPL", "NVDA"]


def test_local_provider_slicing(tmp_path):
    prices_file = tmp_path / "prices.parquet"
    dates = pd.date_range("2026-08-01", "2026-08-10")
    records = []
    for dt in dates:
        records.append({"Date": dt, "Ticker": "AAPL", "Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Volume": 1000})
        records.append({"Date": dt, "Ticker": "MSFT", "Open": 200.0, "High": 205.0, "Low": 199.0, "Close": 202.0, "Volume": 2000})
    df = pd.DataFrame(records)
    df.to_parquet(prices_file, index=False)

    provider = LocalProvider(prices_path=str(prices_file))
    sliced = provider.get_historical_bars(["AAPL"], "2026-08-02", "2026-08-05")
    assert len(sliced) == 4
    assert set(sliced["Ticker"]) == {"AAPL"}
    assert "VWAP" in sliced.columns


def test_rolling_price_store(tmp_path):
    store_file = tmp_path / "store.parquet"
    dates = pd.date_range("2026-01-01", periods=150, freq="B")
    records = []
    for dt in dates:
        records.append({"Date": dt, "Ticker": "AAPL", "Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Volume": 1000})
    df = pd.DataFrame(records)
    df.to_parquet(store_file, index=False)

    store = RollingPriceStore(initial_path=str(store_file))
    assert store.get_max_date() == dates[-1]

    trailing = store.get_trailing_window(tickers=["AAPL"], lookback_days=120)
    assert len(trailing) == 120

    new_date = dates[-1] + pd.Timedelta(days=1)
    new_bar = pd.DataFrame([{"Date": new_date, "Ticker": "AAPL", "Open": 103.0, "High": 106.0, "Low": 101.0, "Close": 104.0, "Volume": 1500}])
    store.append_bars(new_bar)
    assert store.get_max_date() == new_date

    save_path = tmp_path / "store_saved.parquet"
    store.save(str(save_path))
    assert save_path.exists()
