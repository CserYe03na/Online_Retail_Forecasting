from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


DEFAULT_FEATURE_COLS = [
    "dow",
    "dom",
    "weekofyear",
    "month",
    "quarter",
    "is_weekend",
    "is_q4",
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
]


@dataclass
class C1Artifacts:
    train_panel: pd.DataFrame
    test_panel: pd.DataFrame
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    pred_df: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_period: pd.DataFrame
    ape_box_df: pd.DataFrame
    ape_box_df_positive: Optional[pd.DataFrame] = None
    ape_box_df_trimmed: Optional[pd.DataFrame] = None
    error_quantiles: Optional[pd.DataFrame] = None
    tuning_trials: Optional[pd.DataFrame] = None
    tuning_best_config: Optional[Dict[str, Any]] = None
    prediction_output_path: Optional[str] = None


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return eps


def _prepare_metric_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    valid = np.isfinite(yt) & np.isfinite(yp)
    return yt, yp, valid


def pointwise_safe_ape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> np.ndarray:
    eps = _validate_eps(eps)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), eps)
    return np.abs(yt - yp) / denom * 100.0


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    eps = _validate_eps(eps)
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if int(np.sum(valid)) == 0:
        return float("nan")
    ape = pointwise_safe_ape(yt[valid], yp[valid], eps=eps)
    return float(np.mean(ape))


def bounded_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0,
    ape_cap: float = 100.0,
) -> float:
    eps = _validate_eps(eps)
    ape_cap = float(ape_cap)
    if ape_cap <= 0:
        raise ValueError("ape_cap must be > 0")
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if int(np.sum(valid)) == 0:
        return float("nan")
    ape = pointwise_safe_ape(yt[valid], yp[valid], eps=eps)
    return float(np.mean(np.clip(ape, 0.0, ape_cap)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    eps = _validate_eps(eps)
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if int(np.sum(valid)) == 0:
        return float("nan")
    yt = yt[valid]
    yp = yp[valid]
    numerator = np.sum(np.abs(yt - yp))
    denominator = max(float(np.sum(np.abs(yt))), eps)
    return float(numerator / denominator * 100.0)


def positive_only_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    yt = yt[valid]
    yp = yp[valid]
    mask = yt > 0
    if mask.sum() == 0:
        return float("nan")
    denom = np.maximum(yt[mask], eps)
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom) * 100.0)


def occurrence_f1(y_true_sale: np.ndarray, y_pred_sale: np.ndarray) -> float:
    y_true_sale = y_true_sale.astype(int)
    y_pred_sale = y_pred_sale.astype(int)
    tp = int(np.sum((y_true_sale == 1) & (y_pred_sale == 1)))
    fp = int(np.sum((y_true_sale == 0) & (y_pred_sale == 1)))
    fn = int(np.sum((y_true_sale == 1) & (y_pred_sale == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def zero_day_fpr(y_true_sale: np.ndarray, y_pred_sale: np.ndarray) -> float:
    y_true_sale = y_true_sale.astype(int)
    y_pred_sale = y_pred_sale.astype(int)
    zero_mask = y_true_sale == 0
    if int(np.sum(zero_mask)) == 0:
        return float("nan")
    fp = np.sum((y_pred_sale == 1) & zero_mask)
    return float(fp / np.sum(zero_mask))


def compute_metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_sale: np.ndarray,
    metric_name: str,
    eps: float,
) -> Dict[str, float]:
    eps = _validate_eps(eps)
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    ys = np.asarray(y_true_sale).reshape(-1)
    if ys.shape[0] != yt.shape[0]:
        raise ValueError("y_true_sale must have the same length as y_true/y_pred")
    ys = ys[valid].astype(int)
    yt = yt[valid]
    yp = yp[valid]

    epsilon_mape = safe_mape(yt, yp, eps=eps)
    cap_mape = bounded_mape(yt, yp, eps=eps, ape_cap=100.0)

    if metric_name == "bounded_mape":
        main_mape = cap_mape
    elif metric_name == "safe_mape":
        main_mape = epsilon_mape
    else:
        raise ValueError("metric_name must be one of: bounded_mape, safe_mape")

    y_pred_sale = (yp > 0).astype(int)
    return {
        "MAPE_0_100": float(main_mape),
        "EPSILON_MAPE_PCT": float(epsilon_mape),
        "CAP_MAPE_0_100": float(cap_mape),
        "POSITIVE_ONLY_MAPE_PCT": float(positive_only_mape(yt, yp)),
        "WMAPE_0_100": float(wmape(yt, yp, eps=eps)),
        "OCCURRENCE_F1": float(occurrence_f1(ys, y_pred_sale)),
        "ZERO_DAY_FPR": float(zero_day_fpr(ys, y_pred_sale)),
    }


def _load_daily(train_path: str | Path, test_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _build_zero_filled_panel(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sku_map = train_raw[["product_family_name", "cluster"]].drop_duplicates()
    sku_list = sku_map["product_family_name"].tolist()

    train_dates = pd.date_range(train_raw["date"].min(), train_raw["date"].max(), freq="D")
    test_dates = pd.date_range(test_raw["date"].min(), test_raw["date"].max(), freq="D")

    train_grid = pd.MultiIndex.from_product(
        [sku_list, train_dates], names=["product_family_name", "date"]
    ).to_frame(index=False)
    test_grid = pd.MultiIndex.from_product(
        [sku_list, test_dates], names=["product_family_name", "date"]
    ).to_frame(index=False)

    train_grid = train_grid.merge(sku_map, on="product_family_name", how="left")
    test_grid = test_grid.merge(sku_map, on="product_family_name", how="left")

    train_panel = train_grid.merge(
        train_raw, on=["date", "product_family_name", "cluster"], how="left"
    )
    test_panel = test_grid.merge(
        test_raw, on=["date", "product_family_name", "cluster"], how="left"
    )

    train_panel["total_sales"] = train_panel["total_sales"].fillna(0.0)
    test_panel["total_sales"] = test_panel["total_sales"].fillna(0.0)
    return train_panel, test_panel


def _build_features(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(
        [train_panel.assign(split="train"), test_panel.assign(split="test")], ignore_index=True
    ).sort_values(["product_family_name", "date"])

    all_df["y"] = all_df["total_sales"].astype(float)
    all_df["is_sale"] = (all_df["y"] > 0).astype(int)

    all_df["dow"] = all_df["date"].dt.dayofweek
    all_df["dom"] = all_df["date"].dt.day
    all_df["weekofyear"] = all_df["date"].dt.isocalendar().week.astype(int)
    all_df["month"] = all_df["date"].dt.month
    all_df["quarter"] = all_df["date"].dt.quarter
    all_df["is_weekend"] = all_df["dow"].isin([5, 6]).astype(int)
    all_df["is_q4"] = (all_df["quarter"] == 4).astype(int)

    g = all_df.groupby("product_family_name", group_keys=False)
    for lag in [1, 7, 14, 28]:
        all_df[f"lag_{lag}"] = g["y"].shift(lag)

    for w in [7, 14, 28]:
        all_df[f"roll_mean_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).mean()
        all_df[f"roll_std_{w}"] = g["y"].shift(1).rolling(w, min_periods=1).std()

    # Robust implementation to avoid groupby.apply dropping group columns in newer pandas
    last_sale_date = all_df["date"].where(all_df["y"] > 0)
    all_df["last_sale_date"] = (
        last_sale_date.groupby(all_df["product_family_name"]).ffill()
    )
    all_df["days_since_last_sale"] = (
        all_df["date"] - all_df["last_sale_date"]
    ).dt.days
    all_df["days_since_last_sale"] = all_df["days_since_last_sale"].fillna(999).astype(int)
    all_df = all_df.drop(columns=["last_sale_date"])

    lag_roll_cols = [c for c in all_df.columns if c.startswith("lag_") or c.startswith("roll_")]
    all_df[lag_roll_cols] = all_df[lag_roll_cols].fillna(0.0)

    train_feat = all_df[all_df["split"] == "train"].copy()
    test_feat = all_df[all_df["split"] == "test"].copy()
    return train_feat, test_feat


def _seasonal_naive_7(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> pd.DataFrame:
    pred_list: List[pd.DataFrame] = []
    for sku, tr in train_panel.groupby("product_family_name"):
        te = test_panel[test_panel["product_family_name"] == sku].copy()
        if te.empty:
            continue

        full = pd.concat(
            [
                tr[["date", "total_sales"]].assign(split="train"),
                te[["date", "total_sales"]].assign(split="test"),
            ],
            ignore_index=True,
        ).sort_values("date")
        full["pred_snaive7"] = full["total_sales"].shift(7).fillna(0.0)

        out = full[full["split"] == "test"][["date", "pred_snaive7"]].copy()
        out["product_family_name"] = sku
        pred_list.append(out)

    return pd.concat(pred_list, ignore_index=True)


def _fit_two_stage_hgb(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    cls_params: Optional[Dict[str, Any]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
    tau: float = 0.0,
    alpha: float = 1.0,
    cap_value: float = np.inf,
    peak_prob_threshold: float = 1.0,
    peak_mult: float = 1.0,
    smooth_gamma: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = train_feat[feature_cols].astype(float)
    y_cls = train_feat["is_sale"].astype(int).values
    X_test = test_feat[feature_cols].astype(float)

    if cls_params is None:
        cls_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_iter": 300,
            "min_samples_leaf": 50,
        }
    if reg_params is None:
        reg_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_iter": 400,
            "min_samples_leaf": 30,
        }

    clf = HistGradientBoostingClassifier(random_state=random_state, **cls_params)
    clf.fit(X_train, y_cls)
    p_sale = clf.predict_proba(X_test)[:, 1]

    reg_train = train_feat[train_feat["is_sale"] == 1].copy()
    X_reg = reg_train[feature_cols].astype(float)
    y_reg_pos = np.log1p(reg_train["y"].astype(float).values)

    reg = HistGradientBoostingRegressor(random_state=random_state, **reg_params)
    reg.fit(X_reg, y_reg_pos)

    pred_log = reg.predict(X_test)
    pred_pos = np.expm1(pred_log).clip(min=0.0)
    pred_final = p_sale * pred_pos
    if tau > 0:
        pred_final = np.where(p_sale < tau, 0.0, pred_final)
    pred_final = alpha * pred_final
    if peak_mult > 1.0 and peak_prob_threshold < 1.0:
        pred_final = np.where(p_sale >= peak_prob_threshold, pred_final * peak_mult, pred_final)
    if smooth_gamma > 0.0:
        # Smooth prediction toward recent local level to mitigate excessive volatility
        local_level = test_feat["roll_mean_14"].astype(float).values
        pred_final = (1.0 - smooth_gamma) * pred_final + smooth_gamma * local_level
    pred_final = np.clip(pred_final, 0.0, cap_value)
    return p_sale, pred_pos, pred_final


def _peak_underprediction_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float = 0.9,
    eps: float = 1.0,
) -> float:
    pos = y_true[y_true > 0]
    if len(pos) == 0:
        return 0.0
    threshold = np.quantile(pos, q)
    mask = y_true >= threshold
    if int(np.sum(mask)) == 0:
        return 0.0
    under = np.maximum(y_true[mask] - y_pred[mask], 0.0)
    return float(np.mean(under / np.maximum(y_true[mask], eps)) * 100.0)


def _volatility_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_std_ratio: float = 1.15,
    eps: float = 1.0,
) -> float:
    # Penalize only when predicted volatility is materially higher than actual volatility
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    ratio = std_pred / max(std_true, eps)
    excess = max(0.0, ratio - max_std_ratio)
    return excess * 100.0


def _split_train_val_by_time(train_feat: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train_feat["date"].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * train_ratio))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_feat[train_feat["date"] <= cut_date].copy()
    va = train_feat[train_feat["date"] > cut_date].copy()
    return tr, va


def _objective_from_bundle(metric_bundle: Dict[str, float], tuning_objective: str) -> float:
    if tuning_objective == "wmape":
        return float(metric_bundle["WMAPE_0_100"])
    if tuning_objective == "mape":
        return float(metric_bundle["MAPE_0_100"])
    if tuning_objective == "hybrid":
        pos = metric_bundle["POSITIVE_ONLY_MAPE_PCT"]
        pos = metric_bundle["MAPE_0_100"] if np.isnan(pos) else pos
        return float(0.70 * metric_bundle["WMAPE_0_100"] + 0.30 * pos)
    raise ValueError("tuning_objective must be one of: wmape, mape, hybrid")


def _search_two_stage_params(
    train_feat: pd.DataFrame,
    feature_cols: List[str],
    metric_name: str,
    eps_mape: float,
    tuning_objective: str = "wmape",
    max_trials_per_model_pair: int = 500,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    tr, va = _split_train_val_by_time(train_feat, train_ratio=0.8)
    X_va = va[feature_cols].astype(float)
    y_va = va["y"].astype(float).values

    cls_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 60},
        {"max_depth": 5, "learning_rate": 0.04, "max_iter": 350, "min_samples_leaf": 50},
        {"max_depth": 6, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 80},
        {"max_depth": 7, "learning_rate": 0.02, "max_iter": 650, "min_samples_leaf": 100},
    ]
    reg_grid = [
        {"loss": "squared_error", "max_depth": 4, "learning_rate": 0.05, "max_iter": 400, "min_samples_leaf": 40},
        {"loss": "squared_error", "max_depth": 6, "learning_rate": 0.04, "max_iter": 500, "min_samples_leaf": 30},
        {"loss": "quantile", "quantile": 0.60, "max_depth": 6, "learning_rate": 0.03, "max_iter": 650, "min_samples_leaf": 50},
        {"loss": "quantile", "quantile": 0.70, "max_depth": 6, "learning_rate": 0.03, "max_iter": 650, "min_samples_leaf": 50},
        {"loss": "quantile", "quantile": 0.80, "max_depth": 7, "learning_rate": 0.02, "max_iter": 750, "min_samples_leaf": 60},
    ]
    tau_grid = [0.00, 0.05, 0.10, 0.15, 0.20]
    alpha_grid = [0.85, 1.0, 1.15, 1.30]
    cap_q_grid = [0.995, 0.999, 1.0]
    peak_prob_threshold_grid = [1.0, 0.80, 0.85, 0.90]
    peak_mult_grid = [1.0, 1.05, 1.10, 1.15]
    smooth_gamma_grid = [0.0, 0.05, 0.10, 0.20, 0.30]
    rng = np.random.RandomState(random_state)

    tr_positive = tr[tr["y"] > 0]["y"].values
    if len(tr_positive) == 0:
        default_cfg = {
            "cls_params": cls_grid[1],
            "reg_params": reg_grid[1],
            "tau": 0.0,
            "alpha": 1.0,
            "cap_q": 1.0,
            "cap_value": np.inf,
            "peak_prob_threshold": 1.0,
            "peak_mult": 1.0,
            "smooth_gamma": 0.0,
        }
        return default_cfg, pd.DataFrame([{"score": np.nan, **default_cfg}])

    trial_rows: List[Dict[str, Any]] = []
    best_score = np.inf
    best_cfg: Optional[Dict[str, Any]] = None

    for cls_params in cls_grid:
        clf = HistGradientBoostingClassifier(random_state=random_state, **cls_params)
        X_tr = tr[feature_cols].astype(float)
        y_tr_cls = tr["is_sale"].astype(int).values
        clf.fit(X_tr, y_tr_cls)
        p_va = clf.predict_proba(X_va)[:, 1]

        for reg_params in reg_grid:
            reg_train = tr[tr["is_sale"] == 1].copy()
            X_reg = reg_train[feature_cols].astype(float)
            y_reg_pos = np.log1p(reg_train["y"].astype(float).values)
            reg = HistGradientBoostingRegressor(random_state=random_state, **reg_params)
            reg.fit(X_reg, y_reg_pos)
            pred_pos = np.expm1(reg.predict(X_va)).clip(min=0.0)
            raw_pred = p_va * pred_pos

            param_combos = list(
                product(cap_q_grid, tau_grid, alpha_grid, peak_prob_threshold_grid, peak_mult_grid, smooth_gamma_grid)
            )
            if len(param_combos) > max_trials_per_model_pair:
                selected_idx = rng.choice(len(param_combos), size=max_trials_per_model_pair, replace=False)
                selected_combos = [param_combos[i] for i in selected_idx]
            else:
                selected_combos = param_combos

            va_local_level = va["roll_mean_14"].astype(float).values
            for cap_q, tau, alpha, peak_prob_threshold, peak_mult, smooth_gamma in selected_combos:
                cap_value = np.quantile(tr_positive, cap_q) if cap_q < 1.0 else np.inf
                pred2 = np.where(p_va < tau, 0.0, raw_pred)
                pred2 = alpha * pred2
                if peak_mult > 1.0 and peak_prob_threshold < 1.0:
                    pred2 = np.where(p_va >= peak_prob_threshold, pred2 * peak_mult, pred2)
                if smooth_gamma > 0.0:
                    pred2 = (1.0 - smooth_gamma) * pred2 + smooth_gamma * va_local_level
                pred2 = np.clip(pred2, 0.0, cap_value)

                metric_bundle = compute_metric_bundle(
                    y_true=y_va,
                    y_pred=pred2,
                    y_true_sale=(y_va > 0).astype(int),
                    metric_name=metric_name,
                    eps=eps_mape,
                )
                objective_value = _objective_from_bundle(metric_bundle, tuning_objective=tuning_objective)
                peak_pen = _peak_underprediction_penalty(
                    y_true=y_va,
                    y_pred=pred2,
                    q=0.9,
                    eps=eps_mape,
                )
                vol_pen = _volatility_penalty(
                    y_true=y_va,
                    y_pred=pred2,
                    max_std_ratio=1.15,
                    eps=eps_mape,
                )
                # Optimize closeness via WMAPE-centered objective with peak/volatility regularization.
                score = objective_value + 0.12 * peak_pen + 0.35 * vol_pen
                row = {
                    "score": score,
                    "objective_value": objective_value,
                    "objective_name": tuning_objective,
                    "metric_mape": metric_bundle["MAPE_0_100"],
                    "metric_wmape": metric_bundle["WMAPE_0_100"],
                    "metric_pos_mape": metric_bundle["POSITIVE_ONLY_MAPE_PCT"],
                    "peak_under_penalty": peak_pen,
                    "volatility_penalty": vol_pen,
                    "cls_params": cls_params,
                    "reg_params": reg_params,
                    "tau": tau,
                    "alpha": alpha,
                    "cap_q": cap_q,
                    "cap_value": cap_value,
                    "peak_prob_threshold": peak_prob_threshold,
                    "peak_mult": peak_mult,
                    "smooth_gamma": smooth_gamma,
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
                        "peak_prob_threshold": peak_prob_threshold,
                        "peak_mult": peak_mult,
                        "smooth_gamma": smooth_gamma,
                    }

    trials_df = pd.DataFrame(trial_rows).sort_values("score").reset_index(drop=True)
    if best_cfg is None:
        best_cfg = {
            "cls_params": cls_grid[1],
            "reg_params": reg_grid[1],
            "tau": 0.0,
            "alpha": 1.0,
            "cap_q": 1.0,
            "cap_value": np.inf,
            "peak_prob_threshold": 1.0,
            "peak_mult": 1.0,
            "smooth_gamma": 0.0,
        }
    return best_cfg, trials_df


def _periodize_test(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    out = df.copy()
    unique_dates = np.array(sorted(out["date"].dropna().unique()))
    chunks = np.array_split(unique_dates, n_periods)
    mapper: Dict[pd.Timestamp, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        for d in chunk:
            mapper[pd.Timestamp(d)] = f"P{i}"
    out["period"] = out["date"].map(mapper)
    return out


def _build_error_quantiles(ape_box_df: pd.DataFrame) -> pd.DataFrame:
    if ape_box_df.empty:
        return pd.DataFrame(columns=["method", "period", "count", "q50", "q75", "q90", "q95", "q99", "mean"])
    rows: List[Dict[str, Any]] = []
    for (m, p), grp in ape_box_df.groupby(["method", "period"]):
        arr = grp["APE_0_100"].values.astype(float)
        rows.append(
            {
                "method": m,
                "period": p,
                "count": int(len(arr)),
                "q50": float(np.quantile(arr, 0.50)),
                "q75": float(np.quantile(arr, 0.75)),
                "q90": float(np.quantile(arr, 0.90)),
                "q95": float(np.quantile(arr, 0.95)),
                "q99": float(np.quantile(arr, 0.99)),
                "mean": float(np.mean(arr)),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "period"]).reset_index(drop=True)


def run_c1_pipeline(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 1,
    n_periods: int = 4,
    eps_mape: float = 1.0,
    metric_name: str = "bounded_mape",
    tune: bool = True,
    tuning_objective: str = "wmape",
    max_trials_per_model_pair: int = 500,
    feature_cols: List[str] | None = None,
    random_state: int = 42,
    prediction_output_path: str | Path = "forecasting/c1_prediction.parquet",
) -> C1Artifacts:
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    train_daily, test_daily = _load_daily(train_path, test_path)
    train_raw = train_daily[train_daily["cluster"] == cluster_id].copy()
    test_raw = test_daily[test_daily["cluster"] == cluster_id].copy()

    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test.")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_feat, test_feat = _build_features(train_panel, test_panel)

    # Baseline: Seasonal Naive(s=7)
    snaive_pred = _seasonal_naive_7(train_panel, test_panel)

    tuning_trials: Optional[pd.DataFrame] = None
    tuning_best_config: Optional[Dict[str, Any]] = None

    # Champion: two-stage HGB, optionally tuned on time-based pseudo validation split
    if tune:
        tuning_best_config, tuning_trials = _search_two_stage_params(
            train_feat=train_feat,
            feature_cols=feature_cols,
            metric_name=metric_name,
            eps_mape=eps_mape,
            tuning_objective=tuning_objective,
            max_trials_per_model_pair=max_trials_per_model_pair,
            random_state=random_state,
        )
        p_sale, pred_pos, pred_two_stage = _fit_two_stage_hgb(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            cls_params=tuning_best_config["cls_params"],
            reg_params=tuning_best_config["reg_params"],
            tau=tuning_best_config["tau"],
            alpha=tuning_best_config["alpha"],
            cap_value=tuning_best_config["cap_value"],
            peak_prob_threshold=tuning_best_config["peak_prob_threshold"],
            peak_mult=tuning_best_config["peak_mult"],
            smooth_gamma=tuning_best_config["smooth_gamma"],
            random_state=random_state,
        )
    else:
        p_sale, pred_pos, pred_two_stage = _fit_two_stage_hgb(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            random_state=random_state,
        )

    pred_df = test_feat[
        ["date", "product_family_name", "cluster", "y", "is_sale"]
    ].copy()
    pred_df = pred_df.merge(
        snaive_pred, on=["date", "product_family_name"], how="left"
    )
    pred_df["pred_snaive7"] = pred_df["pred_snaive7"].fillna(0.0)
    pred_df["p_sale"] = p_sale
    pred_df["pred_pos_if_sale"] = pred_pos
    pred_df["pred_two_stage"] = pred_two_stage

    # Overall metrics
    overall_rows = []
    for method, col in [("snaive7", "pred_snaive7"), ("two_stage_hgb", "pred_two_stage")]:
        metric_bundle = compute_metric_bundle(
            y_true=pred_df["y"].values,
            y_pred=pred_df[col].values,
            y_true_sale=pred_df["is_sale"].values,
            metric_name=metric_name,
            eps=eps_mape,
        )
        overall_rows.append({"method": method, **metric_bundle})
    metrics_overall = pd.DataFrame(overall_rows).sort_values("MAPE_0_100")

    # Period metrics + boxplot data
    pred_df = _periodize_test(pred_df, n_periods=n_periods)

    period_rows = []
    ape_rows = []
    for method, col in [("snaive7", "pred_snaive7"), ("two_stage_hgb", "pred_two_stage")]:
        ape_eps = pointwise_safe_ape(pred_df["y"].values, pred_df[col].values, eps=eps_mape)
        ape_cap = np.clip(ape_eps, 0.0, 100.0)
        tmp = pred_df[["date", "period", "product_family_name"]].copy()
        tmp["method"] = method
        tmp["APE_EPS_PCT"] = ape_eps
        tmp["APE_CAP_0_100"] = ape_cap
        tmp["APE_0_100"] = ape_cap
        ape_rows.append(tmp)
        for period, grp in pred_df.groupby("period"):
            metric_bundle = compute_metric_bundle(
                y_true=grp["y"].values,
                y_pred=grp[col].values,
                y_true_sale=grp["is_sale"].values,
                metric_name=metric_name,
                eps=eps_mape,
            )
            period_rows.append({"method": method, "period": period, **metric_bundle})

    ape_box_df = pd.concat(ape_rows, ignore_index=True)
    ape_box_df_positive = pd.DataFrame(
        columns=[
            "date",
            "period",
            "product_family_name",
            "method",
            "APE_EPS_PCT",
            "APE_CAP_0_100",
            "APE_0_100",
        ]
    )
    ape_box_df_trimmed = ape_box_df.copy()
    if not ape_box_df_trimmed.empty:
        # Keep error distribution informative by trimming per method-period at 99th percentile
        keep_idx = []
        for (m, p), grp in ape_box_df_trimmed.groupby(["method", "period"]):
            cap = np.quantile(grp["APE_0_100"].values, 0.99)
            keep_idx.extend(grp.index[grp["APE_0_100"] <= cap].tolist())
        ape_box_df_trimmed = ape_box_df_trimmed.loc[keep_idx].copy()

    # Build positive-only boxplot data with same APE definition
    for method, col in [("snaive7", "pred_snaive7"), ("two_stage_hgb", "pred_two_stage")]:
        grp = pred_df[pred_df["y"] > 0][["date", "period", "product_family_name", "y"]].copy()
        if grp.empty:
            continue
        ape_pos = pointwise_safe_ape(grp["y"].values, pred_df.loc[grp.index, col].values, eps=1e-12)
        grp["method"] = method
        grp["APE_EPS_PCT"] = ape_pos
        grp["APE_CAP_0_100"] = np.clip(ape_pos, 0.0, 100.0)
        grp["APE_0_100"] = grp["APE_CAP_0_100"]
        ape_box_df_positive = pd.concat(
            [ape_box_df_positive, grp[["date", "period", "product_family_name", "method", "APE_EPS_PCT", "APE_CAP_0_100", "APE_0_100"]]],
            ignore_index=True,
        )

    error_quantiles = _build_error_quantiles(ape_box_df)
    metrics_by_period = (
        pd.DataFrame(period_rows)
        .sort_values(["method", "period"])
        .reset_index(drop=True)
    )

    # Persist prediction output for downstream analysis/reporting
    prediction_output_path = str(prediction_output_path)
    Path(prediction_output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(prediction_output_path, index=False)

    return C1Artifacts(
        train_panel=train_panel,
        test_panel=test_panel,
        train_feat=train_feat,
        test_feat=test_feat,
        pred_df=pred_df,
        metrics_overall=metrics_overall,
        metrics_by_period=metrics_by_period,
        ape_box_df=ape_box_df,
        ape_box_df_positive=ape_box_df_positive,
        ape_box_df_trimmed=ape_box_df_trimmed,
        error_quantiles=error_quantiles,
        tuning_trials=tuning_trials,
        tuning_best_config=tuning_best_config,
        prediction_output_path=prediction_output_path,
    )
