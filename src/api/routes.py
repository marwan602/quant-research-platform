import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
from pydantic import BaseModel, Field

from src.inference import PortfolioConstructor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
router = APIRouter(prefix="/api/v1")


def _resolve_report_path(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / "reports" / "live" / filename,
        PROJECT_ROOT / "reports" / filename,
        Path("reports") / "live" / filename,
        Path("reports") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_json_file(filename: str) -> dict:
    path = _resolve_report_path(filename)
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Report file {filename} not found. Ensure daily pipeline has executed.",
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RebalanceRequest(BaseModel):
    portfolio_value: float = Field(default=100000.0, gt=0.0)
    top_quantile: float = Field(default=0.10, gt=0.0, le=0.5)
    current_holdings: dict[str, float] = Field(default_factory=dict)


@router.get("/system/status")
def get_system_status() -> dict[str, Any]:
    try:
        data = _load_json_file("system_status.json")
        return {
            "api_status": "online",
            "service_version": "1.0.0",
            "pipeline_status": data.get("status", "unknown"),
            "last_sync_date": data.get("last_sync_date"),
            "active_model": "AttentiveTransformer",
            "active_universe_count": data.get("active_universe_count", 0),
            "evaluated_stocks_count": data.get("evaluated_stocks_count", 0),
            "device": data.get("device", "cpu"),
            "inference_duration_seconds": data.get("duration_seconds", 0.0),
            "updated_at": data.get("updated_at"),
        }
    except HTTPException:
        return {
            "api_status": "online",
            "service_version": "1.0.0",
            "pipeline_status": "pending_first_run",
            "active_model": "AttentiveTransformer",
        }


@router.get("/rankings/latest")
def get_latest_rankings(
    top_k: int = Query(default=10, ge=1, le=100),
    bottom_k: int = Query(default=10, ge=0, le=100),
) -> dict[str, Any]:
    data = _load_json_file("rankings.json")
    all_rankings = data.get("rankings", [])

    top_picks = all_rankings[:top_k]
    bottom_picks = all_rankings[-bottom_k:] if bottom_k > 0 else []

    return {
        "as_of_date": data.get("as_of_date"),
        "model": data.get("model", "AttentiveTransformer"),
        "total_universe": data.get("total_universe", len(all_rankings)),
        "top_k": top_k,
        "bottom_k": bottom_k,
        "top_picks": top_picks,
        "bottom_picks": bottom_picks,
    }


@router.get("/portfolio/current")
def get_current_portfolio() -> dict[str, Any]:
    data = _load_json_file("portfolio.json")
    return {
        "as_of_date": data.get("as_of_date"),
        "strategy": data.get("strategy", "Top Decile Long-Only"),
        "holdings_count": data.get("holdings_count", len(data.get("holdings", []))),
        "holdings": data.get("holdings", []),
        "long_short_top": data.get("long_short_top", []),
        "long_short_bottom": data.get("long_short_bottom", []),
    }


@router.post("/portfolio/rebalance")
def simulate_rebalance(req: RebalanceRequest) -> dict[str, Any]:
    rankings_data = _load_json_file("rankings.json")
    raw_list = rankings_data.get("rankings", [])
    if not raw_list:
        raise HTTPException(status_code=503, detail="No rankings available for rebalance.")

    records = [{"Ticker": r["ticker"], "pred_return_5d": r["pred_return_5d"]} for r in raw_list]
    preds_df = pd.DataFrame(records)

    portfolio_res = PortfolioConstructor.construct_portfolios(preds_df, top_quantile=req.top_quantile)
    target_weights = portfolio_res["long_only_weights"]

    orders_df = PortfolioConstructor.compute_rebalance_orders(
        target_weights=target_weights,
        current_weights=req.current_holdings,
        portfolio_value=req.portfolio_value,
    )

    orders_list = orders_df.to_dict(orient="records") if not orders_df.empty else []

    return {
        "as_of_date": rankings_data.get("as_of_date"),
        "portfolio_value": req.portfolio_value,
        "top_quantile": req.top_quantile,
        "orders_count": len(orders_list),
        "orders": orders_list,
    }


@router.get("/models/benchmark")
def get_model_benchmarks() -> dict[str, Any]:
    lgbm_path = _resolve_report_path("lightgbm_metrics.json")
    alstm_path = _resolve_report_path("alstm_metrics.json")
    trans_path = _resolve_report_path("transformer_metrics.json")

    metrics = {}
    for name, p in [("LightGBM", lgbm_path), ("ALSTM", alstm_path), ("AttentiveTransformer", trans_path)]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            metrics[name] = {
                "annualized_return_net": round(float(raw.get("long_only_annualized_return_net", 0.0)), 4),
                "annualized_vol_net": round(float(raw.get("long_only_annualized_vol_net", 0.0)), 4),
                "sharpe_net": round(float(raw.get("long_only_sharpe_net", 0.0)), 3),
                "max_drawdown_net": round(float(raw.get("long_only_max_drawdown_net", 0.0)), 4),
                "rank_ic_mean": round(float(raw.get("rank_ic_mean", 0.0)), 4),
                "rank_ic_ir": round(float(raw.get("rank_ic_ir", 0.0)), 3),
                "rank_ic_win_rate": round(float(raw.get("rank_ic_pct_positive", 0.0)), 3),
                "annualized_turnover": round(float(raw.get("long_only_mean_turnover", 0.0)), 3),
                "long_short_sharpe_net": round(float(raw.get("long_short_sharpe_net", 0.0)), 3),
                "long_short_annualized_return_net": round(float(raw.get("long_short_annualized_return_net", 0.0)), 4),
            }

    benchmark_data = {}
    ref_path = trans_path if trans_path.exists() else lgbm_path
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_raw = json.load(f)
        benchmark_data = {
            "name": "S&P 500 Equal-Weighted Index",
            "annualized_return": round(float(ref_raw.get("benchmark_annualized_return", 0.1369)), 4),
            "annualized_vol": round(float(ref_raw.get("benchmark_annualized_vol", 0.1294)), 4),
            "sharpe": round(float(ref_raw.get("benchmark_sharpe", 1.058)), 3),
            "max_drawdown": round(float(ref_raw.get("benchmark_max_drawdown", 0.1579)), 4),
        }

    return {
        "models": metrics,
        "benchmark": benchmark_data,
    }
