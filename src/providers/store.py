from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return p


class RollingPriceStore:
    def __init__(self, initial_path: str | None = "data/raw/s_and_p_500_prices.parquet"):
        self._df = None
        if initial_path is not None:
            resolved = _resolve_path(initial_path)
            if resolved.exists():
                self.load(str(resolved))

    def load(self, path: str) -> None:
        p = _resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Store path not found: {path}")
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        if "VWAP" not in df.columns:
            df["VWAP"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        self._df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    def save(self, path: str) -> None:
        if self._df is None:
            return
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(p, index=False)

    def get_max_date(self) -> pd.Timestamp | None:
        if self._df is None or self._df.empty:
            return None
        return pd.to_datetime(self._df["Date"].max())

    def append_bars(self, new_df: pd.DataFrame) -> None:
        if new_df.empty:
            return
        df_to_add = new_df.copy()
        df_to_add["Date"] = pd.to_datetime(df_to_add["Date"])
        if "VWAP" not in df_to_add.columns:
            df_to_add["VWAP"] = (df_to_add["High"] + df_to_add["Low"] + df_to_add["Close"]) / 3.0

        if self._df is None or self._df.empty:
            self._df = df_to_add.sort_values(["Date", "Ticker"]).reset_index(drop=True)
            return

        combined = pd.concat([self._df, df_to_add], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Ticker"], keep="last")
        self._df = combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    def get_trailing_window(
        self,
        tickers: list[str] | None = None,
        lookback_days: int = 120,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if self._df is None or self._df.empty:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"])

        if as_of_date is None:
            target_dt = self.get_max_date()
        else:
            target_dt = pd.to_datetime(as_of_date)

        history = self._df[self._df["Date"] <= target_dt]
        if history.empty:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"])

        unique_dates = sorted(history["Date"].unique())
        selected_dates = unique_dates[-lookback_days:]

        mask = history["Date"].isin(selected_dates)
        if tickers is not None:
            mask = mask & (history["Ticker"].isin(set(tickers)))

        out_df = history[mask].sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return out_df
