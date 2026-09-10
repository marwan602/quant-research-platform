import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceEngine, PortfolioConstructor
from src.providers.base import LocalProvider, UniverseProvider, YahooProvider
from src.providers.polygon import PolygonProvider
from src.providers.store import RollingPriceStore


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return p


def get_provider(name: str, **kwargs):
    normalized = name.lower().strip()
    if normalized == "polygon":
        return PolygonProvider(**kwargs)
    elif normalized == "yahoo":
        return YahooProvider()
    elif normalized == "local":
        return LocalProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}. Choose 'polygon', 'yahoo', or 'local'.")


def run_daily_update(
    provider_name: str = "polygon",
    target_date: str | None = None,
    store_path: str = "data/raw/s_and_p_500_prices.parquet",
    composition_path: str = "data/raw/s_and_p_500_daily_composition.parquet",
    output_dir: str = "reports/live",
    device: str = "cpu",
    dry_run: bool = False,
    provider_instance=None,
) -> dict:
    start_time = time.time()
    resolved_store_path = _resolve_path(store_path)
    resolved_comp_path = _resolve_path(composition_path)
    resolved_out_dir = _resolve_path(output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    universe_provider = UniverseProvider(composition_path=str(resolved_comp_path))
    store = RollingPriceStore(initial_path=str(resolved_store_path) if resolved_store_path.exists() else None)

    as_of = pd.to_datetime(target_date) if target_date else pd.Timestamp.now().normalize()

    provider = provider_instance or get_provider(provider_name)

    synced_days = 0
    if not dry_run:
        synced_days = provider.sync_delta(store, as_of, universe_provider)
        if synced_days > 0:
            store.save(str(resolved_store_path))

    effective_date = store.get_max_date()
    if effective_date is None:
        raise RuntimeError("RollingPriceStore has no data after synchronization.")

    active_tickers = universe_provider.get_active_tickers(as_of_date=effective_date)
    trailing_df = store.get_trailing_window(
        tickers=active_tickers,
        lookback_days=125,
        as_of_date=effective_date,
    )

    engine = InferenceEngine(device=device)
    preds_df = engine.predict(trailing_df, as_of_date=effective_date)
    portfolio_res = PortfolioConstructor.construct_portfolios(preds_df)

    rankings_df = portfolio_res["rankings"]
    top_tickers = portfolio_res["top_tickers"]
    bottom_tickers = portfolio_res["bottom_tickers"]
    lo_weights = portfolio_res["long_only_weights"]

    date_str = effective_date.strftime("%Y-%m-%d")

    rankings_payload = {
        "as_of_date": date_str,
        "model": "AttentiveTransformer",
        "total_universe": len(rankings_df),
        "rankings": [
            {
                "rank": int(row["rank"]),
                "ticker": row["Ticker"],
                "pred_return_5d": float(row["pred_return_5d"]),
                "pred_return_pct": round(float(row["pred_return_5d"]) * 100.0, 3),
                "decile": int(row["decile"]),
            }
            for _, row in rankings_df.iterrows()
        ],
    }

    portfolio_payload = {
        "as_of_date": date_str,
        "strategy": "Top Decile Long-Only",
        "holdings_count": len(top_tickers),
        "holdings": [
            {
                "ticker": t,
                "weight": round(float(lo_weights.get(t, 0.0)), 5),
            }
            for t in top_tickers
        ],
        "long_short_top": top_tickers,
        "long_short_bottom": bottom_tickers,
    }

    elapsed = round(time.time() - start_time, 2)
    system_status_payload = {
        "status": "healthy",
        "last_sync_date": date_str,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "synced_days": synced_days,
        "active_universe_count": len(active_tickers),
        "evaluated_stocks_count": len(preds_df),
        "device": device,
        "duration_seconds": elapsed,
    }

    rankings_file = resolved_out_dir / "rankings.json"
    portfolio_file = resolved_out_dir / "portfolio.json"
    status_file = resolved_out_dir / "system_status.json"

    with open(rankings_file, "w", encoding="utf-8") as f:
        json.dump(rankings_payload, f, indent=2)

    with open(portfolio_file, "w", encoding="utf-8") as f:
        json.dump(portfolio_payload, f, indent=2)

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(system_status_payload, f, indent=2)

    return {
        "status": "success",
        "date": date_str,
        "evaluated": len(preds_df),
        "duration_seconds": elapsed,
        "rankings_file": str(rankings_file),
        "portfolio_file": str(portfolio_file),
        "status_file": str(status_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="polygon" if os.getenv("POLYGON_API_KEY") else "local")
    parser.add_argument("--target-date", type=str, default=None)
    parser.add_argument("--store-path", type=str, default="data/raw/s_and_p_500_prices.parquet")
    parser.add_argument("--composition-path", type=str, default="data/raw/s_and_p_500_daily_composition.parquet")
    parser.add_argument("--output-dir", type=str, default="reports/live")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = run_daily_update(
        provider_name=args.provider,
        target_date=args.target_date,
        store_path=args.store_path,
        composition_path=args.composition_path,
        output_dir=args.output_dir,
        device=args.device,
        dry_run=args.dry_run,
    )
    print(f"Daily update completed in {res['duration_seconds']}s. Evaluated {res['evaluated']} stocks as of {res['date']}.")
