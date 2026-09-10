from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return p


class UniverseProvider:
    def __init__(self, composition_path: str = "data/raw/s_and_p_500_daily_composition.parquet"):
        self.composition_path = _resolve_path(composition_path)
        self._df = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(self.composition_path)
            self._df["Date"] = pd.to_datetime(self._df["Date"])
        return self._df

    def get_active_tickers(self, as_of_date: str | pd.Timestamp | None = None) -> list[str]:
        df = self._load_data()
        if as_of_date is None:
            target_dt = df["Date"].max()
        else:
            target_dt = pd.to_datetime(as_of_date)

        filtered = df[df["Date"] <= target_dt]
        if filtered.empty:
            target_dt = df["Date"].min()
            filtered = df[df["Date"] == target_dt]
        else:
            latest_available = filtered["Date"].max()
            filtered = df[df["Date"] == latest_available]

        tickers = filtered["Ticker"].dropna().unique().tolist()
        return sorted(tickers)


class MarketDataProvider(ABC):
    @abstractmethod
    def get_historical_bars(
        self,
        tickers: list[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_latest_bars(
        self,
        tickers: list[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        pass

    def sync_delta(self, store, target_date: str | pd.Timestamp, universe_provider: UniverseProvider) -> int:
        target_dt = pd.to_datetime(target_date)
        max_dt = store.get_max_date()

        if max_dt is not None and max_dt >= target_dt:
            return 0

        start_dt = max_dt + pd.Timedelta(days=1) if max_dt is not None else target_dt - pd.Timedelta(days=180)
        tickers = universe_provider.get_active_tickers(target_dt)
        delta_df = self.get_historical_bars(tickers, start_dt, target_dt)

        if delta_df.empty:
            return 0

        store.append_bars(delta_df)
        return len(delta_df["Date"].unique())


class LocalProvider(MarketDataProvider):
    def __init__(self, prices_path: str = "data/raw/s_and_p_500_prices.parquet"):
        self.prices_path = _resolve_path(prices_path)
        self._df = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(self.prices_path)
            self._df["Date"] = pd.to_datetime(self._df["Date"])
            if "VWAP" not in self._df.columns:
                self._df["VWAP"] = (self._df["High"] + self._df["Low"] + self._df["Close"]) / 3.0
        return self._df

    def get_historical_bars(
        self,
        tickers: list[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        df = self._load_data()
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        ticker_set = set(tickers)

        mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt) & (df["Ticker"].isin(ticker_set))
        sliced = df[mask].sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return sliced

    def get_latest_bars(
        self,
        tickers: list[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        return self.get_historical_bars(tickers, date, date)


class YahooProvider(MarketDataProvider):
    def __init__(self):
        pass

    @staticmethod
    def _format_ticker(ticker: str) -> str:
        return ticker.replace(".", "-")

    @staticmethod
    def _unformat_ticker(ticker: str) -> str:
        return ticker.replace("-", ".")

    def get_historical_bars(
        self,
        tickers: list[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        import yfinance as yf

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        yf_tickers = [self._format_ticker(t) for t in tickers]

        records = []
        chunk_size = 100
        for i in range(0, len(yf_tickers), chunk_size):
            chunk = yf_tickers[i : i + chunk_size]
            data = yf.download(
                chunk,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for yf_t in chunk:
                    orig_t = self._unformat_ticker(yf_t)
                    try:
                        sub = data.xs(yf_t, axis=1, level="Ticker").dropna(how="all")
                    except KeyError:
                        continue
                    for dt, row in sub.iterrows():
                        c = row.get("Close")
                        if pd.isna(c):
                            continue
                        records.append({
                            "Date": pd.to_datetime(dt),
                            "Ticker": orig_t,
                            "Open": float(row.get("Open", c)),
                            "High": float(row.get("High", c)),
                            "Low": float(row.get("Low", c)),
                            "Close": float(c),
                            "Volume": float(row.get("Volume", 0.0)),
                        })
            else:
                orig_t = self._unformat_ticker(chunk[0])
                for dt, row in data.iterrows():
                    c = row.get("Close")
                    if pd.isna(c):
                        continue
                    records.append({
                        "Date": pd.to_datetime(dt),
                        "Ticker": orig_t,
                        "Open": float(row.get("Open", c)),
                        "High": float(row.get("High", c)),
                        "Low": float(row.get("Low", c)),
                        "Close": float(c),
                        "Volume": float(row.get("Volume", 0.0)),
                    })

        if not records:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"])

        out_df = pd.DataFrame(records)
        out_df["VWAP"] = (out_df["High"] + out_df["Low"] + out_df["Close"]) / 3.0
        out_df = out_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return out_df

    def get_latest_bars(
        self,
        tickers: list[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        return self.get_historical_bars(tickers, date, date)
