import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.providers.base import UniverseProvider, YahooProvider
from src.providers.store import RollingPriceStore


def sync_market_data(
    target_date: str = "2026-09-10",
    store_path: str = "data/raw/s_and_p_500_prices.parquet",
    composition_path: str = "data/raw/s_and_p_500_daily_composition.parquet",
) -> int:
    store = RollingPriceStore(initial_path=store_path)
    current_max = store.get_max_date()
    target_dt = pd.to_datetime(target_date)

    if current_max is not None and current_max >= target_dt:
        print(f"Store is already up to date (max date: {current_max.strftime('%Y-%m-%d')}).")
        return 0

    universe = UniverseProvider(composition_path=composition_path)
    tickers = universe.get_active_tickers(target_dt)

    start_dt = current_max + pd.Timedelta(days=1) if current_max is not None else target_dt - pd.Timedelta(days=30)
    print(f"Syncing market data for {len(tickers)} tickers from {start_dt.strftime('%Y-%m-%d')} to {target_dt.strftime('%Y-%m-%d')}...")

    provider = YahooProvider()
    delta_df = provider.get_historical_bars(tickers, start_dt, target_dt)

    if delta_df.empty:
        print("No new trading bars were returned by the data provider.")
        return 0

    new_dates = sorted(delta_df["Date"].unique())
    print(f"Retrieved {len(delta_df)} new price records across {len(new_dates)} trading dates: {[pd.to_datetime(d).strftime('%Y-%m-%d') for d in new_dates]}")

    store.append_bars(delta_df)
    store.save(store_path)
    print(f"Updated store saved to {store_path}. New max date: {store.get_max_date().strftime('%Y-%m-%d')}")
    return len(delta_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", type=str, default="2026-09-10")
    parser.add_argument("--store-path", type=str, default="data/raw/s_and_p_500_prices.parquet")
    args = parser.parse_args()

    sync_market_data(target_date=args.target_date, store_path=args.store_path)
