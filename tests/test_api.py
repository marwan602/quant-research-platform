from fastapi.testclient import TestClient
import pytest

from src.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_system_status():
    resp = client.get("/api/v1/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api_status"] == "online"
    assert data["active_model"] == "AttentiveTransformer"
    assert "pipeline_status" in data


def test_rankings_latest():
    resp = client.get("/api/v1/rankings/latest?top_k=5&bottom_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_k"] == 5
    assert data["bottom_k"] == 5
    assert len(data["top_picks"]) == 5
    assert len(data["bottom_picks"]) == 5

    first_pick = data["top_picks"][0]
    assert "rank" in first_pick
    assert "ticker" in first_pick
    assert "pred_return_5d" in first_pick
    assert "pred_return_pct" in first_pick
    assert "decile" in first_pick


def test_rankings_validation():
    resp_invalid_low = client.get("/api/v1/rankings/latest?top_k=0")
    assert resp_invalid_low.status_code == 422

    resp_invalid_high = client.get("/api/v1/rankings/latest?top_k=500")
    assert resp_invalid_high.status_code == 422


def test_portfolio_current():
    resp = client.get("/api/v1/portfolio/current")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "Top Decile Long-Only"
    assert "holdings" in data
    assert len(data["holdings"]) > 0

    first_holding = data["holdings"][0]
    assert "ticker" in first_holding
    assert "weight" in first_holding
    assert first_holding["weight"] > 0.0


def test_portfolio_rebalance():
    payload = {
        "portfolio_value": 50000.0,
        "top_quantile": 0.05,
        "current_holdings": {"COHR": 0.05, "AAPL": 0.02},
    }
    resp = client.post("/api/v1/portfolio/rebalance", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_value"] == 50000.0
    assert data["top_quantile"] == 0.05
    assert data["orders_count"] > 0
    assert len(data["orders"]) == data["orders_count"]

    first_order = data["orders"][0]
    assert "Ticker" in first_order
    assert "Action" in first_order
    assert first_order["Action"] in ("BUY", "SELL")
    assert "TradeValue" in first_order
    assert "WeightDelta" in first_order


def test_portfolio_rebalance_validation():
    payload = {
        "portfolio_value": -1000.0,
        "top_quantile": 0.10,
    }
    resp = client.post("/api/v1/portfolio/rebalance", json=payload)
    assert resp.status_code == 422


def test_models_benchmark():
    resp = client.get("/api/v1/models/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "benchmark" in data

    models = data["models"]
    assert "LightGBM" in models
    assert "ALSTM" in models
    assert "AttentiveTransformer" in models

    trans = models["AttentiveTransformer"]
    assert "annualized_return_net" in trans
    assert "sharpe_net" in trans
    assert "rank_ic_mean" in trans
    assert "max_drawdown_net" in trans

    assert trans["sharpe_net"] > 1.15
    assert trans["rank_ic_mean"] > 0.02

    benchmark = data["benchmark"]
    assert benchmark["name"] == "S&P 500 Equal-Weighted Index"
    assert "sharpe" in benchmark
