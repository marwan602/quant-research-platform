import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.tracker import (
    get_realized_forward_returns,
    update_forward_tracking,
    update_prediction_archive,
)


def test_get_realized_forward_returns():
    dates = pd.date_range("2026-09-01", periods=10, freq="B")
    records = []
    for i, dt in enumerate(dates):
        records.append({"Date": dt, "Ticker": "AAPL", "Close": 100.0 + i * 2.0})
        records.append({"Date": dt, "Ticker": "MSFT", "Close": 200.0 + i * 5.0})

    df = pd.DataFrame(records)
    res = get_realized_forward_returns(df, dates[0], forward_days=5)

    assert not res.empty
    assert set(res["Ticker"]) == {"AAPL", "MSFT"}

    aapl_ret = res[res["Ticker"] == "AAPL"].iloc[0]["realized_return_5d"]
    expected_aapl = (110.0 - 100.0) / 100.0
    assert abs(aapl_ret - expected_aapl) < 1e-5

    no_res = get_realized_forward_returns(df, dates[8], forward_days=5)
    assert no_res.empty


def test_update_prediction_archive_and_resolve(tmp_path):
    archive_file = tmp_path / "archive.json"
    as_of = "2026-09-01"

    preds = [
        {"rank": 1, "ticker": "AAPL", "pred_return_5d": 0.05, "decile": 1},
        {"rank": 2, "ticker": "MSFT", "pred_return_5d": 0.02, "decile": 1},
        {"rank": 3, "ticker": "NVDA", "pred_return_5d": -0.01, "decile": 2},
    ]

    arch = update_prediction_archive(archive_file, preds, as_of)
    assert as_of in arch["dates"]
    assert arch["dates"][as_of]["status"] == "pending"

    dates = pd.date_range("2026-09-01", periods=10, freq="B")
    records = []
    for i, dt in enumerate(dates):
        records.append({"Date": dt, "Ticker": "AAPL", "Close": 100.0 + i * 2.0})
        records.append({"Date": dt, "Ticker": "MSFT", "Close": 200.0 + i * 1.0})
        records.append({"Date": dt, "Ticker": "NVDA", "Close": 300.0 - i * 3.0})
    store_df = pd.DataFrame(records)

    arch_resolved = update_prediction_archive(archive_file, preds, as_of, store_df=store_df, forward_days=5)
    entry = arch_resolved["dates"][as_of]

    aapl_pred = [p for p in entry["predictions"] if p["ticker"] == "AAPL"][0]
    assert aapl_pred["realized_return_5d"] is not None


def test_update_forward_tracking(tmp_path):
    tracking_file = tmp_path / "tracking.json"
    archive = {
        "dates": {
            "2026-09-10": {
                "as_of_date": "2026-09-10",
                "status": "pending",
                "rank_ic": None,
                "directional_accuracy": None,
                "predictions": [
                    {"rank": 1, "ticker": "AAPL", "pred_return_5d": 0.04, "decile": 1},
                    {"rank": 2, "ticker": "MSFT", "pred_return_5d": 0.02, "decile": 1},
                ],
            }
        }
    }

    tracking = update_forward_tracking(tracking_file, archive, deployment_date="2026-09-10")
    assert tracking["deployment_date"] == "2026-09-10"
    assert tracking["status"] == "active"
    assert "signal_metrics" in tracking
    assert "portfolio_metrics" in tracking
    assert len(tracking["equity_curve"]) >= 1
    assert tracking["equity_curve"][0]["model_nav"] == 1.0
