import json
import numpy as np
import pandas as pd
import pytest
import yaml
from src.models.lightgbm_model import (
    train_lightgbm,
    predict_lightgbm,
    get_feature_importance,
    save_model,
    load_model,
    run_pipeline,
)


def make_synthetic_training_data(n_samples=200, n_features=10):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Target linearly correlated with first two features
    y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.05 * np.random.randn(n_samples)).astype(np.float32)
    return X, y


def test_train_and_predict():
    X, y = make_synthetic_training_data(200, 10)
    X_tr, y_tr = X[:150], y[:150]
    X_v, y_v = X[150:], y[150:]

    cfg = {
        "models": {
            "lightgbm": {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 10,
                "random_state": 42,
            }
        }
    }

    model = train_lightgbm(X_tr, y_tr, X_v, y_v, config=cfg)
    preds = predict_lightgbm(model, X_v)

    assert preds.shape == (50,)
    assert not np.isnan(preds).any()
    corr = np.corrcoef(preds, y_v)[0, 1]
    assert corr > 0.5


def test_deterministic_reproducibility():
    X, y = make_synthetic_training_data(100, 5)
    cfg = {
        "models": {
            "lightgbm": {
                "n_estimators": 20,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "deterministic": True,
            }
        }
    }
    m1 = train_lightgbm(X, y, X, y, config=cfg)
    p1 = predict_lightgbm(m1, X)

    m2 = train_lightgbm(X, y, X, y, config=cfg)
    p2 = predict_lightgbm(m2, X)

    assert np.array_equal(p1, p2)


def test_feature_importance():
    X, y = make_synthetic_training_data(100, 5)
    feature_names = [f"FEAT_{i}" for i in range(5)]
    model = train_lightgbm(X, y, X, y, config={})
    imp_df = get_feature_importance(model, feature_names=feature_names)

    assert len(imp_df) == 5
    assert list(imp_df.columns) == ["feature", "importance"]
    assert imp_df["importance"].iloc[0] >= imp_df["importance"].iloc[-1]


def test_model_save_and_load(tmp_path):
    X, y = make_synthetic_training_data(100, 5)
    model = train_lightgbm(X, y, X, y, config={})
    save_file = str(tmp_path / "model.txt")
    save_model(model, save_file)

    loaded_booster = load_model(save_file)
    preds_orig = predict_lightgbm(model, X)
    preds_loaded = loaded_booster.predict(X)

    assert np.allclose(preds_orig, preds_loaded)


def test_run_pipeline_end_to_end(tmp_path):
    n_days = 60
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    rows = []
    np.random.seed(42)
    feat_names = [f"FEAT_{i}" for i in range(5)]

    for d in dates:
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            row = {"Date": d, "Ticker": ticker, "target": np.random.normal(0.001, 0.02)}
            for f in feat_names:
                row[f] = np.random.normal(0, 1)
            rows.append(row)

    df = pd.DataFrame(rows)
    processed_file = tmp_path / "processed.parquet"
    df.to_parquet(processed_file, index=False)

    cfg = {
        "data": {
            "processed_features_path": str(processed_file),
        },
        "split": {
            "train_start": "2021-01-01",
            "train_end": "2021-02-15",
            "val_start": "2021-02-16",
            "val_end": "2021-03-01",
            "test_start": "2021-03-02",
            "test_end": "2021-03-31",
        },
        "target": {
            "horizon": 2,
        },
        "models": {
            "lightgbm": {
                "n_estimators": 20,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 5,
                "random_state": 42,
            }
        },
    }

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)

    out_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"

    metrics = run_pipeline(
        config_path=str(cfg_file),
        output_dir=str(out_dir),
        model_dir=str(model_dir),
    )

    assert "ic_mean" in metrics
    assert "rank_ic_mean" in metrics
    assert (model_dir / "lightgbm_model.txt").exists()
    assert (out_dir / "lightgbm_metrics.json").exists()
    assert (out_dir / "lightgbm_feature_importance.csv").exists()
    assert (out_dir / "lightgbm_test_predictions.parquet").exists()
