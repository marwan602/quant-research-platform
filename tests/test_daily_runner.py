import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from scripts.daily_update import get_provider, run_daily_update
from src.providers.base import LocalProvider, MarketDataProvider, UniverseProvider
from src.providers.polygon import PolygonProvider
from src.providers.store import RollingPriceStore


def test_get_provider():
    p_poly = get_provider("polygon", api_key="test")
    assert isinstance(p_poly, PolygonProvider)

    p_local = get_provider("local")
    assert isinstance(p_local, LocalProvider)

    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("invalid_provider")


def test_run_daily_update_pipeline(tmp_path):
    dates = pd.date_range("2026-01-01", periods=130, freq="B")
    tickers = ["AAPL", "MSFT", "NVDA"]

    records = []
    for dt in dates:
        for t in tickers:
            records.append({
                "Date": dt,
                "Ticker": t,
                "Open": 100.0,
                "High": 105.0,
                "Low": 99.0,
                "Close": 102.0,
                "Volume": 10000.0,
                "VWAP": 102.0,
            })
    prices_df = pd.DataFrame(records)
    store_file = tmp_path / "prices.parquet"
    prices_df.to_parquet(store_file, index=False)

    comp_records = [{"Date": dates[-1], "Ticker": t} for t in tickers]
    comp_df = pd.DataFrame(comp_records)
    comp_file = tmp_path / "comp.parquet"
    comp_df.to_parquet(comp_file, index=False)

    out_dir = tmp_path / "live"

    mock_provider = MagicMock(spec=MarketDataProvider)
    mock_provider.sync_delta.return_value = 0

    res = run_daily_update(
        provider_name="mock",
        target_date=dates[-1].strftime("%Y-%m-%d"),
        store_path=str(store_file),
        composition_path=str(comp_file),
        output_dir=str(out_dir),
        device="cpu",
        dry_run=False,
        provider_instance=mock_provider,
    )

    assert res["status"] == "success"
    assert res["evaluated"] == 3
    assert Path(res["rankings_file"]).exists()
    assert Path(res["portfolio_file"]).exists()
    assert Path(res["status_file"]).exists()

    with open(res["rankings_file"], "r", encoding="utf-8") as f:
        rankings = json.load(f)
    assert rankings["model"] == "AttentiveTransformer"
    assert rankings["total_universe"] == 3
    assert len(rankings["rankings"]) == 3
    assert set(r["ticker"] for r in rankings["rankings"]) == set(tickers)

    with open(res["portfolio_file"], "r", encoding="utf-8") as f:
        portfolio = json.load(f)
    assert "holdings" in portfolio
    assert portfolio["strategy"] == "Top Decile Long-Only"

    with open(res["status_file"], "r", encoding="utf-8") as f:
        status = json.load(f)
    assert status["status"] == "healthy"
    assert status["evaluated_stocks_count"] == 3
    assert status["device"] == "cpu"
