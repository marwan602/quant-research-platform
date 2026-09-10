from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.providers.polygon import PolygonProvider
from src.providers.base import UniverseProvider
from src.providers.store import RollingPriceStore


def test_polygon_provider_missing_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = PolygonProvider(api_key=None)
    with pytest.raises(ValueError, match="Polygon API key is required"):
        provider.get_latest_bars(["AAPL"], "2026-09-10")


def test_polygon_provider_successful_parse():
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "OK",
        "queryCount": 3,
        "resultsCount": 3,
        "results": [
            {
                "T": "AAPL",
                "o": 220.0,
                "h": 225.0,
                "l": 219.0,
                "c": 224.0,
                "v": 50000000.0,
                "vw": 222.8,
            },
            {
                "T": "MSFT",
                "o": 440.0,
                "h": 445.0,
                "l": 438.0,
                "c": 442.0,
                "v": 20000000.0,
                "vw": None,
            },
            {
                "T": "UNKNOWN",
                "o": 10.0,
                "h": 11.0,
                "l": 9.5,
                "c": 10.5,
                "v": 10000.0,
                "vw": 10.2,
            },
        ],
    }
    mock_session.get.return_value = mock_response

    provider = PolygonProvider(api_key="mock_key", session=mock_session)
    df = provider.get_latest_bars(["AAPL", "MSFT"], "2026-09-10")

    assert len(df) == 2
    assert list(df.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"]
    assert set(df["Ticker"]) == {"AAPL", "MSFT"}

    aapl_row = df[df["Ticker"] == "AAPL"].iloc[0]
    assert aapl_row["Date"] == pd.to_datetime("2026-09-10")
    assert aapl_row["Close"] == 224.0
    assert aapl_row["VWAP"] == 222.8

    msft_row = df[df["Ticker"] == "MSFT"].iloc[0]
    expected_msft_vwap = (445.0 + 438.0 + 442.0) / 3.0
    assert abs(msft_row["VWAP"] - expected_msft_vwap) < 1e-5


def test_polygon_provider_empty_results():
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "OK",
        "queryCount": 0,
        "resultsCount": 0,
        "results": [],
    }
    mock_session.get.return_value = mock_response

    provider = PolygonProvider(api_key="mock_key", session=mock_session)
    df = provider.get_latest_bars(["AAPL"], "2026-09-13")

    assert df.empty
    assert list(df.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"]


def test_polygon_provider_error_handling():
    mock_session = MagicMock()

    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403
    mock_resp_403.text = "Forbidden"
    mock_session.get.return_value = mock_resp_403

    provider = PolygonProvider(api_key="bad_key", session=mock_session)
    with pytest.raises(PermissionError, match="Polygon authentication failed"):
        provider.get_latest_bars(["AAPL"], "2026-09-10")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Rate limit exceeded"
    mock_session.get.return_value = mock_resp_429

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        provider.get_latest_bars(["AAPL"], "2026-09-10")


def test_polygon_provider_historical_and_sync_delta(tmp_path):
    mock_session = MagicMock()

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        date_str = url.split("/")[-1]
        resp.json.return_value = {
            "status": "OK",
            "results": [
                {"T": "AAPL", "o": 100.0, "h": 105.0, "l": 99.0, "c": 102.0, "v": 1000.0, "vw": 102.0},
            ],
        }
        return resp

    mock_session.get.side_effect = side_effect

    provider = PolygonProvider(api_key="mock_key", session=mock_session, rate_limit_delay=0.0)
    hist_df = provider.get_historical_bars(["AAPL"], "2026-09-08", "2026-09-09")
    assert len(hist_df) == 2

    comp_file = tmp_path / "comp.parquet"
    comp_df = pd.DataFrame({"Date": pd.to_datetime(["2026-09-01"]), "Ticker": ["AAPL"]})
    comp_df.to_parquet(comp_file, index=False)
    universe = UniverseProvider(composition_path=str(comp_file))

    store = RollingPriceStore(initial_path=None)
    store.append_bars(pd.DataFrame([{
        "Date": pd.to_datetime("2026-09-07"),
        "Ticker": "AAPL",
        "Open": 99.0, "High": 101.0, "Low": 98.0, "Close": 100.0, "Volume": 500.0, "VWAP": 100.0
    }]))

    synced_days = provider.sync_delta(store, "2026-09-09", universe)
    assert synced_days == 2
    assert store.get_max_date() == pd.to_datetime("2026-09-09")
