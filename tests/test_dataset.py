import pytest
import numpy as np
import pandas as pd
import torch
from src.dataset import (
    split_tabular_data,
    get_feature_columns,
    StockSequenceDataset,
    load_processed_data,
    prepare_sequence_data,
)


def make_sample_processed_df(n_days=100, n_tickers=2):
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    rows = []
    np.random.seed(42)
    feature_names = [f"FEAT_{i}" for i in range(158)]

    for ticker in [f"TICKER_{i}" for i in range(n_tickers)]:
        for date in dates:
            row = {"Date": date, "Ticker": ticker, "target": np.random.normal(0, 0.02)}
            for f in feature_names:
                row[f] = np.random.normal(0, 1)
            rows.append(row)

    return pd.DataFrame(rows)


def test_get_feature_columns():
    df = make_sample_processed_df(10, 1)
    cols = get_feature_columns(df)
    assert len(cols) == 158
    assert "Date" not in cols
    assert "Ticker" not in cols
    assert "target" not in cols


def test_split_tabular_data():
    df = make_sample_processed_df(200, 2)
    config = {
        "split": {
            "train_start": "2021-01-01",
            "train_end": "2021-04-30",
            "val_start": "2021-05-01",
            "val_end": "2021-06-30",
            "test_start": "2021-07-01",
            "test_end": "2021-10-30",
        }
    }

    (X_tr, y_tr, meta_tr), (X_v, y_v, meta_v), (X_te, y_te, meta_te), f_cols = split_tabular_data(
        df, config=config
    )

    assert len(f_cols) == 158
    assert X_tr.shape[1] == 158
    assert X_v.shape[1] == 158
    assert X_te.shape[1] == 158

    # Strict date boundaries
    assert meta_tr["Date"].max() <= pd.to_datetime("2021-04-30")
    assert meta_v["Date"].min() >= pd.to_datetime("2021-05-01")
    assert meta_v["Date"].max() <= pd.to_datetime("2021-06-30")
    assert meta_te["Date"].min() >= pd.to_datetime("2021-07-01")

    # No NaN targets
    assert not np.isnan(y_tr).any()
    assert not np.isnan(y_v).any()
    assert not np.isnan(y_te).any()


def test_sequence_dataset():
    n_days = 30
    lookback = 10
    features = np.random.randn(n_days, 158).astype(np.float32)
    targets = np.random.randn(n_days).astype(np.float32)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B").values
    tickers = np.array(["TEST"] * n_days)

    dataset = StockSequenceDataset(features, targets, dates, tickers, lookback=lookback)
    assert len(dataset) == n_days - lookback + 1

    x_seq, y_val = dataset[0]
    assert x_seq.shape == (lookback, 158)
    assert isinstance(x_seq, torch.Tensor)
    assert np.isclose(y_val.item(), targets[lookback - 1])


def test_split_tabular_data_overlapping_dates():
    df = make_sample_processed_df(50, 1)
    config = {
        "split": {
            "train_start": "2021-01-01",
            "train_end": "2021-03-01",
            "val_start": "2021-02-15",
            "val_end": "2021-04-01",
            "test_start": "2021-04-02",
            "test_end": "2021-05-01",
        }
    }
    with pytest.raises(ValueError, match="Split ranges must be strictly chronological"):
        split_tabular_data(df, config=config)


def test_sequence_dataset_invalid_lookback():
    features = np.ones((10, 158), dtype=np.float32)
    targets = np.ones(10, dtype=np.float32)
    dates = pd.date_range("2021-01-01", periods=10, freq="B").values
    tickers = np.array(["TEST"] * 10)

    with pytest.raises(ValueError, match="lookback must be a positive integer"):
        StockSequenceDataset(features, targets, dates, tickers, lookback=0)

    with pytest.raises(ValueError, match="lookback must be a positive integer"):
        StockSequenceDataset(features, targets, dates, tickers, lookback=-5)


def test_load_processed_data_drops_incomplete_rows(tmp_path):
    df = make_sample_processed_df(10, 1)
    df.loc[0, "FEAT_0"] = np.nan
    df.loc[1, "target"] = np.nan
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)

    loaded = load_processed_data(str(p))
    assert len(loaded) == 8
    assert not loaded["FEAT_0"].isna().any()
    assert not loaded["target"].isna().any()


def test_sequence_dataset_date_filtering():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    n_days = len(dates)
    features = np.random.randn(n_days, 158).astype(np.float32)
    targets = np.random.randn(n_days).astype(np.float32)
    tickers = np.array(["TEST"] * n_days)

    lookback = 10
    start_dt = "2021-02-01"
    end_dt = "2021-03-01"

    dataset = StockSequenceDataset(
        features,
        targets,
        dates.values,
        tickers,
        lookback=lookback,
        target_start_date=start_dt,
        target_end_date=end_dt,
    )

    meta = dataset.get_metadata()
    assert len(meta) == len(dataset)
    assert meta["Date"].min() >= pd.to_datetime(start_dt)
    assert meta["Date"].max() <= pd.to_datetime(end_dt)

    x_seq, y_val = dataset[0]
    assert x_seq.shape == (lookback, 158)
    assert np.isclose(y_val.item(), meta["target"].iloc[0])


def test_sequence_dataset_get_metadata_empty():
    features = np.ones((5, 158), dtype=np.float32)
    targets = np.ones(5, dtype=np.float32)
    dates = pd.date_range("2021-01-01", periods=5, freq="B").values
    tickers = np.array(["TEST"] * 5)

    dataset = StockSequenceDataset(features, targets, dates, tickers, lookback=10)
    meta = dataset.get_metadata()
    assert len(meta) == 0
    assert list(meta.columns) == ["Date", "Ticker", "target"]


def test_prepare_sequence_data():
    df = make_sample_processed_df(120, 2)
    config = {
        "split": {
            "train_start": "2021-01-01",
            "train_end": "2021-03-15",
            "val_start": "2021-03-16",
            "val_end": "2021-04-30",
            "test_start": "2021-05-01",
            "test_end": "2021-06-18",
        }
    }

    lookback = 10
    train_ds, val_ds, test_ds, scaler, feat_cols = prepare_sequence_data(
        df, config=config, lookback=lookback
    )

    assert len(feat_cols) == 158
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    meta_train = train_ds.get_metadata()
    meta_val = val_ds.get_metadata()
    meta_test = test_ds.get_metadata()

    assert meta_train["Date"].max() <= pd.to_datetime("2021-03-15")
    assert meta_val["Date"].min() >= pd.to_datetime("2021-03-16")
    assert meta_val["Date"].max() <= pd.to_datetime("2021-04-30")
    assert meta_test["Date"].min() >= pd.to_datetime("2021-05-01")
    assert meta_test["Date"].max() <= pd.to_datetime("2021-06-18")

    x_val_0, y_val_0 = val_ds[0]
    assert x_val_0.shape == (lookback, 158)
    assert np.all(x_val_0.numpy() >= -5.0)
    assert np.all(x_val_0.numpy() <= 5.0)



