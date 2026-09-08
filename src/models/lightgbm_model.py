import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from src.dataset import load_config, load_processed_data, split_tabular_data
from src.evaluate import evaluate_predictions


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict | None = None,
) -> lgb.LGBMRegressor:
    if config is None:
        config = load_config()

    lgb_cfg = config.get("models", {}).get("lightgbm", {})
    n_estimators = lgb_cfg.get("n_estimators", 1000)
    learning_rate = lgb_cfg.get("learning_rate", 0.03)
    num_leaves = lgb_cfg.get("num_leaves", 31)
    subsample = lgb_cfg.get("subsample", 0.8)
    colsample_bytree = lgb_cfg.get("colsample_bytree", 0.8)
    early_stopping_rounds = lgb_cfg.get("early_stopping_rounds", 50)
    random_state = lgb_cfg.get("random_state", 42)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective="regression",
        n_jobs=-1,
        verbosity=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )
    return model


def predict_lightgbm(model: lgb.LGBMRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def get_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    raw_importance = model.booster_.feature_importance(importance_type=importance_type)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importance,
    })
    return df_importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model: lgb.LGBMRegressor, save_path: str = "models/lightgbm_model.txt") -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_model(model_path: str = "models/lightgbm_model.txt") -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


def run_pipeline(
    config_path: str = "configs/config.yaml",
    output_dir: str = "reports",
    model_dir: str = "models",
) -> dict:
    config = load_config(config_path)
    df = load_processed_data(config["data"]["processed_features_path"])

    (X_tr, y_tr, meta_tr), (X_v, y_v, meta_v), (X_te, y_te, meta_te), feat_cols = split_tabular_data(
        df, config=config
    )

    model = train_lightgbm(X_tr, y_tr, X_v, y_v, config=config)

    save_model(model, save_path=f"{model_dir}/lightgbm_model.txt")

    pred_test = predict_lightgbm(model, X_te)

    eval_df = meta_te.copy()
    eval_df["target"] = y_te
    eval_df["pred"] = pred_test

    holding_period = config.get("target", {}).get("horizon", 5)
    metrics = evaluate_predictions(eval_df, holding_period=holding_period)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    metrics_path = out_p / "lightgbm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    importance_df = get_feature_importance(model, feature_names=feat_cols)
    importance_path = out_p / "lightgbm_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    predictions_path = out_p / "lightgbm_test_predictions.parquet"
    eval_df.to_parquet(predictions_path, index=False)

    return metrics


if __name__ == "__main__":
    run_pipeline()
