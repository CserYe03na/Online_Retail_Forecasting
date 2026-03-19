from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from c0c2_analysis import (
    build_prediction_frame,
    compute_overall_metric_bundle,
)


DEFAULT_FEATURE_COLS = [
    "dow",
    "dom",
    "weekofyear",
    "month",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_mean_28",
    "roll_std_28",
    "last_nonzero_sales",
    "sale_rate_7",
    "sale_rate_14",
    "sale_rate_28",
    "roll_mean_pos_7",
    "roll_mean_pos_14",
    "roll_mean_pos_28",
    "days_since_last_sale",
]
PRED_COL = "pred_two_stage_lgbm"
METHOD_NAME = "c0_m2_two_stage_lgbm"
DISPLAY_NAME = "Two-stage LGBM"


@dataclass
class C0M2Artifacts:
    train_panel: pd.DataFrame
    test_panel: pd.DataFrame
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    pred_df: pd.DataFrame
    sku_actual_df: pd.DataFrame
    sku_pred_df: pd.DataFrame
    tuning_best_config: Optional[Dict[str, Any]] = None
    tuning_trials: Optional[pd.DataFrame] = None
    prediction_output_path: Optional[str] = None


def _require_lightgbm() -> None:
    if lgb is None:
        raise ImportError(
            "lightgbm is not installed in the current environment. "
            "Please install `lightgbm` before running c0_m2.py."
        )


def _load_daily(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_daily = pd.read_parquet(train_path)
    test_daily = pd.read_parquet(test_path)

    for df in (train_daily, test_daily):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["product_family_name"] = df["product_family_name"].astype("string").str.strip()
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
        df["total_sales"] = pd.to_numeric(df["total_sales"], errors="coerce").fillna(0.0)

    train_daily = (
        train_daily.groupby(["date", "product_family_name", "cluster"], as_index=False)["total_sales"]
        .sum()
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    test_daily = (
        test_daily.groupby(["date", "product_family_name", "cluster"], as_index=False)["total_sales"]
        .sum()
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    return train_daily, test_daily


def _build_zero_filled_panel(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sku_map = train_raw[["product_family_name", "cluster"]].drop_duplicates()
    sku_list = sku_map["product_family_name"].tolist()

    train_dates = pd.date_range(train_raw["date"].min(), train_raw["date"].max(), freq="D")
    test_dates = pd.date_range(test_raw["date"].min(), test_raw["date"].max(), freq="D")

    train_grid = pd.MultiIndex.from_product([sku_list, train_dates], names=["product_family_name", "date"]).to_frame(index=False)
    test_grid = pd.MultiIndex.from_product([sku_list, test_dates], names=["product_family_name", "date"]).to_frame(index=False)

    train_grid = train_grid.merge(sku_map, on="product_family_name", how="left")
    test_grid = test_grid.merge(sku_map, on="product_family_name", how="left")

    train_panel = train_grid.merge(train_raw, on=["date", "product_family_name", "cluster"], how="left")
    test_panel = test_grid.merge(test_raw, on=["date", "product_family_name", "cluster"], how="left")

    train_panel["total_sales"] = train_panel["total_sales"].fillna(0.0)
    test_panel["total_sales"] = test_panel["total_sales"].fillna(0.0)
    return train_panel, test_panel


def _build_features(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(
        [train_panel.assign(split="train"), test_panel.assign(split="test")],
        ignore_index=True,
    ).sort_values(["product_family_name", "date"])

    all_df["y"] = all_df["total_sales"].astype(float)
    all_df["is_sale"] = (all_df["y"] > 0).astype(int)

    all_df["dow"] = all_df["date"].dt.dayofweek
    all_df["dom"] = all_df["date"].dt.day
    all_df["weekofyear"] = all_df["date"].dt.isocalendar().week.astype(int)
    all_df["month"] = all_df["date"].dt.month
    all_df["quarter"] = all_df["date"].dt.quarter
    all_df["is_weekend"] = all_df["dow"].isin([5, 6]).astype(int)

    g = all_df.groupby("product_family_name", group_keys=False)
    for lag in [1, 7, 14, 28]:
        all_df[f"lag_{lag}"] = g["y"].shift(lag)

    for w in [7, 14, 28]:
        all_df[f"roll_mean_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).mean()
        all_df[f"roll_std_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).std()

    prev_nonzero_sales = (
        all_df["y"]
        .where(all_df["y"] > 0)
        .groupby(all_df["product_family_name"])
        .ffill()
        .groupby(all_df["product_family_name"])
        .shift(1)
    )
    all_df["last_nonzero_sales"] = prev_nonzero_sales

    def _sale_rate(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window, min_periods=1).mean()

    def _positive_roll_mean(s: pd.Series, window: int) -> pd.Series:
        prev = s.shift(1)
        return prev.where(prev > 0).rolling(window, min_periods=1).mean()

    for w in [7, 14, 28]:
        all_df[f"sale_rate_{w}"] = g["is_sale"].transform(lambda s, window=w: _sale_rate(s, window))
        all_df[f"roll_mean_pos_{w}"] = g["y"].transform(lambda s, window=w: _positive_roll_mean(s, window))

    last_sale_date = all_df["date"].where(all_df["y"] > 0)
    all_df["last_sale_date"] = last_sale_date.groupby(all_df["product_family_name"]).ffill()
    all_df["days_since_last_sale"] = (all_df["date"] - all_df["last_sale_date"]).dt.days
    all_df["days_since_last_sale"] = all_df["days_since_last_sale"].fillna(999).astype(int)
    all_df = all_df.drop(columns=["last_sale_date"])

    feature_fill_zero_cols = [
        c
        for c in all_df.columns
        if c.startswith("lag_")
        or c.startswith("roll_")
        or c.startswith("sale_rate_")
        or c == "last_nonzero_sales"
    ]
    all_df[feature_fill_zero_cols] = all_df[feature_fill_zero_cols].fillna(0.0)

    train_feat = all_df[all_df["split"] == "train"].copy()
    test_feat = all_df[all_df["split"] == "test"].copy()
    return train_feat, test_feat


def _split_train_val_by_time(train_feat: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train_feat["date"].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * train_ratio))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_feat[train_feat["date"] <= cut_date].copy()
    va = train_feat[train_feat["date"] > cut_date].copy()
    return tr, va


def _compute_cap_value(train_feat: pd.DataFrame, cap_q: float) -> float:
    pos = train_feat.loc[train_feat["y"] > 0, "y"].astype(float).values
    if len(pos) == 0:
        return np.inf
    return float(np.quantile(pos, cap_q))


def _zero_day_overshoot_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    zero_mask = y_true <= 0
    if int(np.sum(zero_mask)) == 0:
        return 0.0
    overshoot = np.maximum(y_pred[zero_mask], 0.0)
    pos = y_true[y_true > 0]
    scale = max(float(np.median(pos)) if len(pos) > 0 else 0.0, eps)
    return float(np.mean(overshoot) / scale * 100.0)


def _apply_two_stage_postprocess(
    p_sale: np.ndarray,
    pred_pos: np.ndarray,
    tau: float,
    alpha: float,
    cap_value: float,
) -> np.ndarray:
    pred_final = p_sale * pred_pos
    pred_final = np.where(p_sale < tau, 0.0, pred_final)
    pred_final = np.clip(alpha * pred_final, 0.0, cap_value)
    return pred_final


def _fit_two_stage_lgbm(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    cls_params: Optional[Dict[str, Any]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
    tau: float = 0.5,
    alpha: float = 1.0,
    cap_value: float = np.inf,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_lightgbm()

    X_train = train_feat[feature_cols].astype(float)
    y_cls = train_feat["is_sale"].astype(int).values
    X_test = test_feat[feature_cols].astype(float)

    if cls_params is None:
        cls_params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "class_weight": "balanced",
        }
    if reg_params is None:
        reg_params = {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    clf = lgb.LGBMClassifier(**cls_params)
    clf.fit(X_train, y_cls)
    p_sale = clf.predict_proba(X_test)[:, 1]

    reg_train = train_feat[train_feat["is_sale"] == 1].copy()
    X_reg = reg_train[feature_cols].astype(float)
    y_reg = np.log1p(reg_train["y"].astype(float).values)

    reg = lgb.LGBMRegressor(**reg_params)
    sample_weight = np.sqrt(y_reg + 1.0)
    reg.fit(X_reg, y_reg, sample_weight=sample_weight)

    pred_log = reg.predict(X_test)
    pred_pos = np.expm1(pred_log).clip(min=0.0)
    pred_final = _apply_two_stage_postprocess(
        p_sale=p_sale,
        pred_pos=pred_pos,
        tau=tau,
        alpha=alpha,
        cap_value=cap_value,
    )
    return p_sale, pred_pos, pred_final


def _search_two_stage_params(
    train_feat: pd.DataFrame,
    feature_cols: List[str],
    eps_mape: float = 1.0,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    _require_lightgbm()
    tr, va = _split_train_val_by_time(train_feat, train_ratio=0.8)
    y_va = va["y"].astype(float).values

    cls_grid = [
        {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": random_state, "class_weight": "balanced"},
        {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": random_state, "class_weight": "balanced"},
    ]
    reg_grid = [
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.0,
            "random_state": random_state,
        },
        {
            "n_estimators": 450,
            "learning_rate": 0.20,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.0,
            "random_state": random_state,
        },
        {
            "n_estimators": 450,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": random_state,
        },
    ]
    tau_grid = [0.25, 0.35, 0.45]
    alpha_grid = [0.9, 1.0, 1.1]
    cap_q_grid = [0.95, 0.98, 0.995]
    lambda_zero_fp_grid = [25.0, 50.0, 75.0]
    lambda_zero_overshoot_grid = [0.05, 0.10, 0.20]

    cap_values = {cap_q: _compute_cap_value(tr, cap_q=cap_q) for cap_q in cap_q_grid}
    trial_rows = []
    best_score = np.inf
    best_cfg = None

    # Train each model-pair once, then sweep cheap post-processing params on cached predictions.
    for cls_params, reg_params in product(cls_grid, reg_grid):
        p_sale, pred_pos, _ = _fit_two_stage_lgbm(
            train_feat=tr,
            test_feat=va,
            feature_cols=feature_cols,
            cls_params=cls_params,
            reg_params=reg_params,
            tau=tau_grid[0],
            alpha=alpha_grid[0],
            cap_value=cap_values[cap_q_grid[0]],
            random_state=random_state,
        )

        for tau, alpha, cap_q in product(tau_grid, alpha_grid, cap_q_grid):
            cap_value = cap_values[cap_q]
            pred = _apply_two_stage_postprocess(
                p_sale=p_sale,
                pred_pos=pred_pos,
                tau=tau,
                alpha=alpha,
                cap_value=cap_value,
            )
            metric_bundle = compute_overall_metric_bundle(
                y_true=y_va,
                y_pred=pred,
                eps=eps_mape,
                metric_name="bounded_mape",
            )
            zero_overshoot = _zero_day_overshoot_pct(y_va, pred, eps=eps_mape)
            base_mape = float(metric_bundle["CAP_MAPE_0_100"])
            zero_day_fpr = float(metric_bundle["ZERO_DAY_FPR"])

            for lambda_zero_fp, lambda_zero_overshoot in product(
                lambda_zero_fp_grid,
                lambda_zero_overshoot_grid,
            ):
                score = (
                    metric_bundle["WMAPE_0_100"]
                    + lambda_zero_fp * zero_day_fpr
                    + lambda_zero_overshoot * zero_overshoot
                )
                row = {
                    "score": float(score),
                    "metric_mape": float(metric_bundle["MAPE_0_100"]),
                    "metric_epsilon_mape": float(metric_bundle["EPSILON_MAPE_PCT"]),
                    "metric_wmape": float(metric_bundle["WMAPE_0_100"]),
                    "metric_zero_day_fpr": zero_day_fpr,
                    "zero_day_overshoot_pct": float(zero_overshoot),
                    "cls_params": cls_params,
                    "reg_params": reg_params,
                    "tau": tau,
                    "alpha": alpha,
                    "cap_q": cap_q,
                    "cap_value": cap_value,
                    "lambda_zero_fp": lambda_zero_fp,
                    "lambda_zero_overshoot": lambda_zero_overshoot,
                }
                trial_rows.append(row)
                if score < best_score:
                    best_score = score
                    best_cfg = {
                        "cls_params": cls_params,
                        "reg_params": reg_params,
                        "tau": tau,
                        "alpha": alpha,
                        "cap_q": cap_q,
                        "cap_value": cap_value,
                        "lambda_zero_fp": lambda_zero_fp,
                        "lambda_zero_overshoot": lambda_zero_overshoot,
                    }

    trials_df = pd.DataFrame(trial_rows).sort_values("score").reset_index(drop=True)
    if best_cfg is None:
        best_cfg = {
            "cls_params": cls_grid[0],
            "reg_params": reg_grid[0],
            "tau": 0.45,
            "alpha": 1.0,
            "cap_q": 0.98,
            "cap_value": _compute_cap_value(train_feat, cap_q=0.98),
            "lambda_zero_fp": 50.0,
            "lambda_zero_overshoot": 0.10,
        }
    return best_cfg, trials_df


def run_c0_m2_pipeline(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 0,
    feature_cols: Optional[List[str]] = None,
    tune: bool = True,
    eps_mape: float = 1.0,
    random_state: int = 42,
    prediction_output_path: str | Path = "forecasting/c0_m2_prediction.parquet",
) -> C0M2Artifacts:
    _require_lightgbm()
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    train_daily, test_daily = _load_daily(train_path, test_path)
    train_raw = train_daily[train_daily["cluster"] == cluster_id].copy()
    test_raw = test_daily[test_daily["cluster"] == cluster_id].copy()

    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test.")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_feat, test_feat = _build_features(train_panel, test_panel)

    tuning_best_config: Optional[Dict[str, Any]] = None
    tuning_trials: Optional[pd.DataFrame] = None

    if tune:
        tuning_best_config, tuning_trials = _search_two_stage_params(
            train_feat=train_feat,
            feature_cols=feature_cols,
            eps_mape=eps_mape,
            random_state=random_state,
        )
        p_sale, pred_pos, pred_final = _fit_two_stage_lgbm(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            cls_params=tuning_best_config["cls_params"],
            reg_params=tuning_best_config["reg_params"],
            tau=tuning_best_config["tau"],
            alpha=tuning_best_config["alpha"],
            cap_value=tuning_best_config["cap_value"],
            random_state=random_state,
        )
    else:
        p_sale, pred_pos, pred_final = _fit_two_stage_lgbm(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            random_state=random_state,
        )

    pred_df = test_feat[["date", "product_family_name", "cluster", "y", "is_sale"]].copy()
    pred_df["p_sale"] = p_sale
    pred_df["pred_pos_if_sale"] = pred_pos
    pred_df["pred_two_stage_lgbm"] = pred_final

    sku_actual_df = pred_df.pivot(index="date", columns="product_family_name", values="y").sort_index(axis=1)
    sku_pred_df = pred_df.pivot(index="date", columns="product_family_name", values="pred_two_stage_lgbm").sort_index(axis=1)

    prediction_output_path = str(prediction_output_path)
    Path(prediction_output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(prediction_output_path, index=False)

    return C0M2Artifacts(
        train_panel=train_panel,
        test_panel=test_panel,
        train_feat=train_feat,
        test_feat=test_feat,
        pred_df=pred_df,
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        tuning_best_config=tuning_best_config,
        tuning_trials=tuning_trials,
        prediction_output_path=prediction_output_path,
    )
