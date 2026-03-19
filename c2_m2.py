from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from c0c2_analysis import compute_overall_metric_bundle


DATE_COL = "date"
SKU_COL = "product_family_name"
CLUSTER_COL = "cluster"
TARGET_COL = "total_sales"

DEFAULT_NUMERIC_FEATURE_COLS = [
    "dow",
    "dom",
    "weekofyear",
    "month",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
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
    "days_since_last_sale",
    "sku_train_mean",
    "sku_train_std",
    "sku_train_nonzero_rate",
    "sku_train_total",
    "sku_train_recent28_mean",
]
DEFAULT_CATEGORICAL_FEATURE_COLS = [SKU_COL]
DEFAULT_FEATURE_COLS = DEFAULT_NUMERIC_FEATURE_COLS + DEFAULT_CATEGORICAL_FEATURE_COLS
PRED_COL = "pred_global_lgbm"
METHOD_NAME = "c2_m2_global_lgbm"
DISPLAY_NAME = "Global LGBM"


@dataclass
class C2M2Artifacts:
    train_panel: pd.DataFrame
    test_panel: pd.DataFrame
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    pred_df: pd.DataFrame
    sku_actual_df: pd.DataFrame
    sku_pred_df: pd.DataFrame
    model_feature_cols: List[str]
    categorical_feature_cols: List[str]
    tuning_best_config: Optional[Dict[str, Any]] = None
    tuning_trials: Optional[pd.DataFrame] = None
    prediction_output_path: Optional[str] = None


def default_random_search_space() -> Dict[str, List[Any]]:
    return {
        "objective": ["regression", "l1", "poisson", "tweedie"],
        "n_estimators": [250, 350, 500],
        "learning_rate": [0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 95],
        "min_child_samples": [20, 40, 80],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 1.0],
        "tweedie_variance_power": [1.1, 1.3, 1.5],
        "threshold_mode": ["global", "sku"],
        "tau_quantile": [0.0, 0.05, 0.1, 0.2],
        "cap_q": [0.99, 0.995, 0.999],
        "recent_weight_max": [1.0, 1.5, 2.0, 3.0],
    }


def sample_random_param_sets(
    param_space: Dict[str, List[Any]],
    n_iter: int,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    sampled: List[Dict[str, Any]] = []
    seen = set()
    max_attempts = max(100, int(n_iter) * 25)
    attempts = 0

    while len(sampled) < int(n_iter) and attempts < max_attempts:
        attempts += 1
        params = {k: v[int(rng.integers(0, len(v)))] for k, v in param_space.items()}
        key = tuple((k, repr(params[k])) for k in sorted(params))
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)
    return sampled


def _require_lightgbm() -> None:
    if lgb is None:
        raise ImportError(
            "lightgbm is not installed in the current environment. "
            "Please install `lightgbm` before running c2_m2.py."
        )


def _load_daily(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train = train_df[train_df[CLUSTER_COL] == cluster_id].copy()
    test = test_df[test_df[CLUSTER_COL] == cluster_id].copy()

    for df in (train, test):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df[SKU_COL] = df[SKU_COL].astype("string").str.strip()
        df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce").astype("Int64")
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)

    train = (
        train.groupby([DATE_COL, SKU_COL, CLUSTER_COL], as_index=False)[TARGET_COL]
        .sum()
        .sort_values([SKU_COL, DATE_COL])
        .reset_index(drop=True)
    )
    test = (
        test.groupby([DATE_COL, SKU_COL, CLUSTER_COL], as_index=False)[TARGET_COL]
        .sum()
        .sort_values([SKU_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return train, test


def _build_zero_filled_panel(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sku_map = train_raw[[SKU_COL, CLUSTER_COL]].drop_duplicates()
    sku_list = sku_map[SKU_COL].tolist()

    train_dates = pd.date_range(train_raw[DATE_COL].min(), train_raw[DATE_COL].max(), freq="D")
    test_dates = pd.date_range(test_raw[DATE_COL].min(), test_raw[DATE_COL].max(), freq="D")

    train_grid = pd.MultiIndex.from_product([sku_list, train_dates], names=[SKU_COL, DATE_COL]).to_frame(index=False)
    test_grid = pd.MultiIndex.from_product([sku_list, test_dates], names=[SKU_COL, DATE_COL]).to_frame(index=False)

    train_grid = train_grid.merge(sku_map, on=SKU_COL, how="left")
    test_grid = test_grid.merge(sku_map, on=SKU_COL, how="left")

    train_panel = train_grid.merge(train_raw, on=[DATE_COL, SKU_COL, CLUSTER_COL], how="left")
    test_panel = test_grid.merge(test_raw, on=[DATE_COL, SKU_COL, CLUSTER_COL], how="left")

    train_panel[TARGET_COL] = train_panel[TARGET_COL].fillna(0.0)
    test_panel[TARGET_COL] = test_panel[TARGET_COL].fillna(0.0)
    return train_panel, test_panel


def _compute_sku_train_stats(train_panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sku, grp in train_panel.groupby(SKU_COL):
        y = grp[TARGET_COL].astype(float)
        recent = y.tail(28)
        rows.append(
            {
                SKU_COL: sku,
                "sku_train_mean": float(y.mean()),
                "sku_train_std": float(y.std(ddof=0)),
                "sku_train_nonzero_rate": float((y > 0).mean()),
                "sku_train_total": float(y.sum()),
                "sku_train_recent28_mean": float(recent.mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_features(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(
        [train_panel.assign(split="train"), test_panel.assign(split="test")],
        ignore_index=True,
    ).sort_values([SKU_COL, DATE_COL])

    all_df["y"] = all_df[TARGET_COL].astype(float)

    all_df["dow"] = all_df[DATE_COL].dt.dayofweek
    all_df["dom"] = all_df[DATE_COL].dt.day
    all_df["weekofyear"] = all_df[DATE_COL].dt.isocalendar().week.astype(int)
    all_df["month"] = all_df[DATE_COL].dt.month
    all_df["quarter"] = all_df[DATE_COL].dt.quarter
    all_df["is_weekend"] = all_df["dow"].isin([5, 6]).astype(int)
    all_df["is_month_start"] = all_df[DATE_COL].dt.is_month_start.astype(int)
    all_df["is_month_end"] = all_df[DATE_COL].dt.is_month_end.astype(int)

    g = all_df.groupby(SKU_COL, group_keys=False)
    for lag in [1, 7, 14, 28]:
        all_df[f"lag_{lag}"] = g["y"].shift(lag)

    for w in [7, 14, 28]:
        all_df[f"roll_mean_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).mean()
        all_df[f"roll_std_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).std()

    last_sale_date = all_df[DATE_COL].where(all_df["y"] > 0)
    all_df["last_sale_date"] = last_sale_date.groupby(all_df[SKU_COL]).ffill()
    all_df["days_since_last_sale"] = (all_df[DATE_COL] - all_df["last_sale_date"]).dt.days
    all_df["days_since_last_sale"] = all_df["days_since_last_sale"].fillna(999).astype(int)
    all_df = all_df.drop(columns=["last_sale_date"])

    sku_stats = _compute_sku_train_stats(train_panel)
    all_df = all_df.merge(sku_stats, on=SKU_COL, how="left")

    fill_zero_cols = [
        c
        for c in all_df.columns
        if c.startswith("lag_")
        or c.startswith("roll_")
        or c.startswith("sku_train_")
    ]
    all_df[fill_zero_cols] = all_df[fill_zero_cols].fillna(0.0)
    all_df[SKU_COL] = all_df[SKU_COL].astype("category")

    train_feat = all_df[all_df["split"] == "train"].copy()
    test_feat = all_df[all_df["split"] == "test"].copy()
    return train_feat, test_feat


def _split_train_val_by_time(
    train_feat: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train_feat[DATE_COL].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * train_ratio))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_feat[train_feat[DATE_COL] <= cut_date].copy()
    va = train_feat[train_feat[DATE_COL] > cut_date].copy()
    return tr, va


def _prepare_model_matrix(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    categorical_feature_cols: List[str],
    category_levels: Optional[Dict[str, List[Any]]] = None,
) -> pd.DataFrame:
    X = feat_df[feature_cols].copy()
    for col in feature_cols:
        if col in categorical_feature_cols:
            if category_levels is not None and col in category_levels:
                X[col] = pd.Categorical(X[col], categories=category_levels[col])
            else:
                X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
    return X


def _build_recency_sample_weight(
    feat_df: pd.DataFrame,
    min_weight: float = 1.0,
    max_weight: float = 2.0,
) -> np.ndarray:
    dates = np.array(sorted(feat_df[DATE_COL].dropna().unique()))
    if len(dates) <= 1:
        return np.ones(len(feat_df), dtype=float)

    min_weight = float(min_weight)
    max_weight = float(max_weight)
    if max_weight < min_weight:
        max_weight = min_weight

    date_to_rank = {pd.Timestamp(dt): i for i, dt in enumerate(dates)}
    ranks = feat_df[DATE_COL].map(date_to_rank).fillna(0).astype(float).values
    denom = max(1.0, float(len(dates) - 1))
    scaled = ranks / denom
    weights = min_weight + (max_weight - min_weight) * scaled
    return np.asarray(weights, dtype=float)


def _fit_global_lgbm_regressor(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    categorical_feature_cols: Optional[List[str]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
    cap_q: float = 0.995,
    recent_weight_max: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    _require_lightgbm()
    categorical_feature_cols = categorical_feature_cols or []

    if reg_params is None:
        reg_params = {
            "objective": "regression",
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "verbose": -1,
        }

    category_levels = {
        col: train_feat[col].astype("category").cat.categories.tolist()
        for col in categorical_feature_cols
        if col in feature_cols
    }
    X_train = _prepare_model_matrix(
        train_feat,
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        category_levels=category_levels,
    )
    X_test = _prepare_model_matrix(
        test_feat,
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        category_levels=category_levels,
    )
    y_train = train_feat["y"].astype(float).values
    y_train_target = np.log1p(np.clip(y_train, 0.0, None))
    sample_weight = _build_recency_sample_weight(
        train_feat,
        min_weight=1.0,
        max_weight=recent_weight_max,
    )

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(
        X_train,
        y_train_target,
        sample_weight=sample_weight,
        categorical_feature=[c for c in categorical_feature_cols if c in feature_cols],
    )

    pred_log = reg.predict(X_test)
    pred = np.expm1(np.asarray(pred_log, dtype=float))
    pos = train_feat.loc[train_feat["y"] > 0, "y"].astype(float).values
    cap_value = float(np.quantile(pos, cap_q)) if len(pos) > 0 else np.inf
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, cap_value)
    return pred


def _compute_global_tau(train_feat: pd.DataFrame, tau_quantile: float) -> float:
    pos = train_feat.loc[train_feat["y"] > 0, "y"].astype(float).values
    if len(pos) == 0:
        return 0.0
    return float(np.quantile(pos, float(tau_quantile)))


def _compute_sku_tau_map(train_feat: pd.DataFrame, tau_quantile: float) -> Dict[Any, float]:
    tau_map: Dict[Any, float] = {}
    global_tau = _compute_global_tau(train_feat=train_feat, tau_quantile=tau_quantile)
    for sku, grp in train_feat.groupby(SKU_COL):
        pos = grp.loc[grp["y"] > 0, "y"].astype(float).values
        tau_map[sku] = float(np.quantile(pos, float(tau_quantile))) if len(pos) > 0 else global_tau
    return tau_map


def _apply_prediction_threshold(
    pred: np.ndarray,
    feat_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    threshold_mode: str,
    tau_quantile: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred = np.asarray(pred, dtype=float).copy()
    threshold_mode = str(threshold_mode).lower()
    tau_quantile = float(tau_quantile)

    if threshold_mode == "global":
        tau = _compute_global_tau(train_feat=train_feat, tau_quantile=tau_quantile)
        pred[pred < tau] = 0.0
        return pred, {"threshold_mode": threshold_mode, "tau": float(tau), "tau_quantile": tau_quantile}

    if threshold_mode == "sku":
        tau_map = _compute_sku_tau_map(train_feat=train_feat, tau_quantile=tau_quantile)
        sku_values = feat_df[SKU_COL].astype(object).values
        tau_vec = np.asarray([tau_map.get(sku, 0.0) for sku in sku_values], dtype=float)
        pred[pred < tau_vec] = 0.0
        return pred, {"threshold_mode": threshold_mode, "tau_quantile": tau_quantile}

    raise ValueError("threshold_mode must be one of: global, sku")


def _search_global_lgbm_params(
    train_feat: pd.DataFrame,
    feature_cols: List[str],
    categorical_feature_cols: List[str],
    eps_mape: float = 1.0,
    n_iter: int = 12,
    param_space: Optional[Dict[str, List[Any]]] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    _require_lightgbm()
    tr, va = _split_train_val_by_time(train_feat, train_ratio=0.8)
    y_va = va["y"].astype(float).values
    if param_space is None:
        param_space = default_random_search_space()
    trial_params = sample_random_param_sets(param_space=param_space, n_iter=n_iter, random_state=random_state)

    trial_rows: List[Dict[str, Any]] = []
    best_score = np.inf
    best_cfg: Optional[Dict[str, Any]] = None

    for sampled_params in trial_params:
        reg_params = {
            "objective": sampled_params["objective"],
            "n_estimators": sampled_params["n_estimators"],
            "learning_rate": sampled_params["learning_rate"],
            "num_leaves": sampled_params["num_leaves"],
            "min_child_samples": sampled_params["min_child_samples"],
            "subsample": sampled_params["subsample"],
            "colsample_bytree": sampled_params["colsample_bytree"],
            "reg_alpha": sampled_params["reg_alpha"],
            "reg_lambda": sampled_params["reg_lambda"],
            "random_state": random_state,
            "verbose": -1,
        }
        if sampled_params["objective"] == "tweedie":
            reg_params["tweedie_variance_power"] = float(sampled_params["tweedie_variance_power"])
        cap_q = float(sampled_params["cap_q"])
        recent_weight_max = float(sampled_params["recent_weight_max"])
        raw_pred = _fit_global_lgbm_regressor(
            train_feat=tr,
            test_feat=va,
            feature_cols=feature_cols,
            categorical_feature_cols=categorical_feature_cols,
            reg_params=reg_params,
            cap_q=cap_q,
            recent_weight_max=recent_weight_max,
            random_state=random_state,
        )
        pred, threshold_info = _apply_prediction_threshold(
            pred=raw_pred,
            feat_df=va,
            train_feat=tr,
            threshold_mode=str(sampled_params["threshold_mode"]),
            tau_quantile=float(sampled_params["tau_quantile"]),
        )
        metric_bundle = compute_overall_metric_bundle(
            y_true=y_va,
            y_pred=pred,
            eps=eps_mape,
            metric_name="bounded_mape",
        )
        score = float(
            0.6 * metric_bundle["WMAPE_0_100"]
            + 0.2 * metric_bundle["POSITIVE_ONLY_MAPE_PCT"]
            + 0.2 * metric_bundle["ZERO_DAY_FPR"]
        )
        row = {
            "score": score,
            "metric_mape": float(metric_bundle["MAPE_0_100"]),
            "metric_epsilon_mape": float(metric_bundle["EPSILON_MAPE_PCT"]),
            "metric_wmape": float(metric_bundle["WMAPE_0_100"]),
            "metric_positive_only_mape": float(metric_bundle["POSITIVE_ONLY_MAPE_PCT"]),
            "metric_zero_day_fpr": float(metric_bundle["ZERO_DAY_FPR"]),
            **sampled_params,
            **threshold_info,
            "reg_params": reg_params,
            "cap_q": cap_q,
        }
        trial_rows.append(row)
        if score < best_score:
            best_score = score
            best_cfg = {
                "reg_params": reg_params,
                "cap_q": cap_q,
                "threshold_mode": str(sampled_params["threshold_mode"]),
                "tau_quantile": float(sampled_params["tau_quantile"]),
                "recent_weight_max": recent_weight_max,
                "selection_metric": "0.6*WMAPE_0_100 + 0.2*POSITIVE_ONLY_MAPE_PCT + 0.2*ZERO_DAY_FPR",
                "selection_score": score,
            }

    trials_df = pd.DataFrame(trial_rows).sort_values("score").reset_index(drop=True)
    if best_cfg is None:
        fallback_trial = trial_params[0] if len(trial_params) > 0 else None
        fallback_reg_params = {
            "objective": "regression",
            "n_estimators": 350,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "verbose": -1,
        }
        best_cfg = {
            "reg_params": fallback_reg_params,
            "cap_q": float(fallback_trial["cap_q"]) if fallback_trial is not None else 0.995,
            "threshold_mode": str(fallback_trial["threshold_mode"]) if fallback_trial is not None else "global",
            "tau_quantile": float(fallback_trial["tau_quantile"]) if fallback_trial is not None else 0.0,
            "recent_weight_max": float(fallback_trial["recent_weight_max"]) if fallback_trial is not None else 2.0,
            "selection_metric": "0.6*WMAPE_0_100 + 0.2*POSITIVE_ONLY_MAPE_PCT + 0.2*ZERO_DAY_FPR",
            "selection_score": float("nan"),
        }
    return best_cfg, trials_df


def run_c2_m2_pipeline(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 2,
    feature_cols: Optional[List[str]] = None,
    categorical_feature_cols: Optional[List[str]] = None,
    tune: bool = True,
    eps_mape: float = 1.0,
    random_search_iter: int = 12,
    random_search_space: Optional[Dict[str, List[Any]]] = None,
    random_state: int = 42,
    prediction_output_path: str | Path = "forecasting/c2_m2_prediction.parquet",
) -> C2M2Artifacts:
    _require_lightgbm()
    if feature_cols is None:
        feature_cols = list(DEFAULT_FEATURE_COLS)
    if categorical_feature_cols is None:
        categorical_feature_cols = [c for c in DEFAULT_CATEGORICAL_FEATURE_COLS if c in feature_cols]

    train_raw, test_raw = _load_daily(train_path=train_path, test_path=test_path, cluster_id=cluster_id)
    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test.")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_feat, test_feat = _build_features(train_panel, test_panel)

    tuning_best_config: Optional[Dict[str, Any]] = None
    tuning_trials: Optional[pd.DataFrame] = None

    if tune:
        tuning_best_config, tuning_trials = _search_global_lgbm_params(
            train_feat=train_feat,
            feature_cols=feature_cols,
            categorical_feature_cols=categorical_feature_cols,
            eps_mape=eps_mape,
            n_iter=random_search_iter,
            param_space=random_search_space,
            random_state=random_state,
        )
        raw_pred_final = _fit_global_lgbm_regressor(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            categorical_feature_cols=categorical_feature_cols,
            reg_params=tuning_best_config["reg_params"],
            cap_q=tuning_best_config["cap_q"],
            recent_weight_max=tuning_best_config["recent_weight_max"],
            random_state=random_state,
        )
        pred_final, _ = _apply_prediction_threshold(
            pred=raw_pred_final,
            feat_df=test_feat,
            train_feat=train_feat,
            threshold_mode=tuning_best_config["threshold_mode"],
            tau_quantile=tuning_best_config["tau_quantile"],
        )
    else:
        raw_pred_final = _fit_global_lgbm_regressor(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            categorical_feature_cols=categorical_feature_cols,
            recent_weight_max=2.0,
            random_state=random_state,
        )
        pred_final, _ = _apply_prediction_threshold(
            pred=raw_pred_final,
            feat_df=test_feat,
            train_feat=train_feat,
            threshold_mode="global",
            tau_quantile=0.0,
        )

    pred_df = test_feat[[DATE_COL, SKU_COL, CLUSTER_COL, "y"]].copy()
    pred_df["pred_global_lgbm_raw"] = raw_pred_final
    pred_df["pred_global_lgbm"] = pred_final

    sku_actual_df = pred_df.pivot(index=DATE_COL, columns=SKU_COL, values="y").sort_index(axis=1)
    sku_pred_df = pred_df.pivot(index=DATE_COL, columns=SKU_COL, values="pred_global_lgbm").sort_index(axis=1)

    prediction_output_path = str(prediction_output_path)
    Path(prediction_output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(prediction_output_path, index=False)

    return C2M2Artifacts(
        train_panel=train_panel,
        test_panel=test_panel,
        train_feat=train_feat,
        test_feat=test_feat,
        pred_df=pred_df,
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        model_feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        tuning_best_config=tuning_best_config,
        tuning_trials=tuning_trials,
        prediction_output_path=prediction_output_path,
    )
