import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.dataset import RobustStandardScaler
from src.inference import PortfolioConstructor, prepare_inference_features, run_pipeline


def test_prepare_inference_features_shape():
    dates = pd.date_range("2026-01-01", periods=125, freq="B")
    records = []
    np.random.seed(42)

    for ticker in ["AAPL", "MSFT"]:
        c = 100.0
        for dt in dates:
            c *= (1.0 + np.random.randn() * 0.01)
            records.append({
                "Date": dt,
                "Ticker": ticker,
                "Open": c * 0.99,
                "High": c * 1.01,
                "Low": c * 0.98,
                "Close": c,
                "Volume": float(np.random.randint(100000, 500000)),
                "VWAP": c,
            })
    df = pd.DataFrame(records)

    scaler = RobustStandardScaler()
    dummy_feats = np.random.randn(200, 158).astype(np.float32)
    scaler.fit(dummy_feats)

    tensor, tickers, as_of_date = prepare_inference_features(
        trailing_df=df,
        scaler=scaler,
        required_feature_steps=60,
    )

    assert tensor.shape == (2, 60, 158)
    assert sorted(tickers) == ["AAPL", "MSFT"]
    assert as_of_date == dates[-1]


def test_prepare_inference_features_insufficient_history():
    dates = pd.date_range("2026-08-01", periods=30, freq="B")
    records = []
    for dt in dates:
        records.append({
            "Date": dt,
            "Ticker": "AAPL",
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0, "VWAP": 100.0
        })
    df = pd.DataFrame(records)
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, 158)))

    with pytest.raises(ValueError, match="No tickers had sufficient valid historical bars"):
        prepare_inference_features(df, scaler=scaler, required_feature_steps=60)


def test_portfolio_constructor():
    preds_df = pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(20)],
        "pred_return_5d": np.linspace(0.10, -0.10, 20),
    })

    result = PortfolioConstructor.construct_portfolios(preds_df, top_quantile=0.1, bottom_quantile=0.1)
    rankings = result["rankings"]
    lo_weights = result["long_only_weights"]
    ls_weights = result["long_short_weights"]

    assert len(rankings) == 20
    assert rankings["rank"].iloc[0] == 1
    assert rankings["Ticker"].iloc[0] == "T0"

    assert len(result["top_tickers"]) == 2
    assert len(result["bottom_tickers"]) == 2

    assert np.isclose(sum(lo_weights.values()), 1.0)
    assert np.isclose(sum(ls_weights.values()), 0.0)

    for t in result["top_tickers"]:
        assert lo_weights[t] == 0.5
        assert ls_weights[t] == 0.25

    for t in result["bottom_tickers"]:
        assert ls_weights[t] == -0.25


def test_portfolio_rebalance_orders():
    target = {"AAPL": 0.5, "MSFT": 0.5}
    current = {"AAPL": 0.2, "GOOG": 0.3}

    orders = PortfolioConstructor.compute_rebalance_orders(
        target_weights=target,
        current_weights=current,
        portfolio_value=100000.0,
    )

    assert len(orders) == 3
    aapl_order = orders[orders["Ticker"] == "AAPL"].iloc[0]
    assert aapl_order["Action"] == "BUY"
    assert np.isclose(aapl_order["TradeValue"], 30000.0)

    goog_order = orders[orders["Ticker"] == "GOOG"].iloc[0]
    assert goog_order["Action"] == "SELL"
    assert np.isclose(goog_order["TradeValue"], 30000.0)


def test_end_to_end_replay_inference():
    result = run_pipeline(as_of_date="2026-08-31", device="cpu")
    rankings = result["rankings"]
    lo_weights = result["long_only_weights"]

    assert not rankings.empty
    assert "pred_return_5d" in rankings.columns
    assert "rank" in rankings.columns
    assert np.isclose(sum(lo_weights.values()), 1.0)
    assert len(result["top_tickers"]) > 0
