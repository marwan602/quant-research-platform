import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_docs_data_files_exist():
    docs_data = PROJECT_ROOT / "docs/data"
    assert docs_data.exists()
    required_files = [
        "company_meta.json",
        "rankings.json",
        "portfolio.json",
        "system_status.json",
        "forward_tracking.json",
        "prediction_archive.json",
        "backtest_series.json",
        "benchmark_metrics.json",
    ]
    for filename in required_files:
        p = docs_data / filename
        assert p.exists(), f"Missing file: {filename}"
        assert p.stat().st_size > 0, f"Empty file: {filename}"


def test_rankings_schema_and_metadata():
    rankings_file = PROJECT_ROOT / "docs/data/rankings.json"
    with open(rankings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "rankings" in data
    assert len(data["rankings"]) >= 500

    first = data["rankings"][0]
    for key in ["rank", "ticker", "pred_return_pct", "decile", "name", "sector", "factors"]:
        assert key in first, f"Missing key '{key}' in ranking item"

    factors = first["factors"]
    for f_name in ["momentum", "volatility", "reversal", "liquidity", "trend"]:
        assert f_name in factors, f"Missing factor '{f_name}' in factors dict"
        assert 0.0 <= factors[f_name] <= 100.0


def test_portfolio_schema_and_sectors():
    port_file = PROJECT_ROOT / "docs/data/portfolio.json"
    with open(port_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "holdings" in data
    assert len(data["holdings"]) == 50
    assert "sector_composition" in data
    assert len(data["sector_composition"]) > 0

    first_h = data["holdings"][0]
    for key in ["ticker", "weight", "name", "sector"]:
        assert key in first_h, f"Missing key '{key}' in holding item"
    assert pytest.approx(first_h["weight"], abs=1e-3) == 0.02


def test_backtest_series_structure():
    bt_file = PROJECT_ROOT / "docs/data/backtest_series.json"
    with open(bt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "rebalance_dates" in data
    assert len(data["rebalance_dates"]) >= 100
    assert len(data["transformer_lo"]) == len(data["rebalance_dates"])
    assert len(data["benchmark"]) == len(data["rebalance_dates"])
    assert "drawdowns" in data
    assert "transformer" in data["drawdowns"]
    assert "benchmark" in data["drawdowns"]
    assert "lightgbm" in data["drawdowns"]
    assert "alstm" in data["drawdowns"]
    assert len(data["drawdowns"]["lightgbm"]) == len(data["rebalance_dates"])
    assert len(data["drawdowns"]["alstm"]) == len(data["rebalance_dates"])
    assert "rank_ic" in data


def test_benchmark_metrics():
    bm_file = PROJECT_ROOT / "docs/data/benchmark_metrics.json"
    with open(bm_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    for expected in ["transformer", "alstm", "lightgbm", "benchmark"]:
        assert expected in model_ids


def test_frontend_assets():
    docs_dir = PROJECT_ROOT / "docs"
    html_path = docs_dir / "index.html"
    css_path = docs_dir / "style.css"
    js_path = docs_dir / "app.js"
    rebal_path = docs_dir / "rebalance.js"

    assert html_path.exists()
    assert css_path.exists()
    assert js_path.exists()
    assert rebal_path.exists()

    html_content = html_path.read_text(encoding="utf-8")
    assert "S&P 500 Quantitative Research Platform" in html_content
    assert "pane-overview" in html_content
    assert "pane-signals" in html_content
    assert "pane-portfolio" in html_content
    assert "pane-research" in html_content
    assert "pane-methodology" in html_content
    assert "TODO" not in html_content
    assert "rebalance.js" in html_content
    assert "dataLoadErrorBanner" in html_content
    assert "customHoldingsError" in html_content

    rebal_content = rebal_path.read_text(encoding="utf-8")
    assert "parseHoldingsInput" in rebal_content
    assert "calculateRebalanceOrders" in rebal_content
    assert "classifyOrder" in rebal_content
    assert "computeDrawdownSeries" in rebal_content
    assert "HOLD (BELOW MIN)" in rebal_content
    assert "HOLD (NO CHANGE)" in rebal_content
    assert "0.94 + ((idx % 7) * 0.02)" not in rebal_content

    js_content = js_path.read_text(encoding="utf-8")
    assert "parseHoldingsInput" in js_content
    assert "calculateRebalanceOrders" in js_content
    assert "0.94 + ((idx % 7) * 0.02)" not in js_content
    assert "(+98.7% Net)" not in js_content
    assert "(+81.5% Net)" not in js_content
    assert "(+78.1% Net)" not in js_content
    assert "(+40.3%)" not in js_content
