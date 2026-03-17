from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from c1_forecasting import (
    bounded_mape,
    DEFAULT_FEATURE_COLS,
    compute_metric_bundle,
    pointwise_safe_ape,
    safe_mape,
)


@dataclass
class C3Artifacts:
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
    baseline_comparison: Optional[pd.DataFrame] = None
    selected_baseline_name: Optional[str] = None


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


def _tsb_baseline(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    alpha: float = 0.1,
    beta: float = 0.1,
    sparse_min_nonzero: int = 3,
    sparse_nonzero_rate: float = 0.01,
) -> pd.DataFrame:
    pred_list: List[pd.DataFrame] = []
    for sku, tr in train_panel.groupby("product_family_name"):
        te = test_panel[test_panel["product_family_name"] == sku].copy()
        if te.empty:
            continue

        y = tr.sort_values("date")["total_sales"].values.astype(float)
        if len(y) == 0:
            continue

        nonzero_cnt = int(np.sum(y > 0))
        nonzero_rate = float(np.mean(y > 0))
        if nonzero_cnt <= sparse_min_nonzero or nonzero_rate <= sparse_nonzero_rate:
            f = 0.0
            out = te[["date", "product_family_name"]].copy()
            out["pred_tsb"] = f
            pred_list.append(out)
            continue

        p = nonzero_rate
        pos = y[y > 0]
        z = float(np.mean(pos)) if len(pos) > 0 else 0.0

        for yt in y:
            occ = 1.0 if yt > 0 else 0.0
            p = p + alpha * (occ - p)
            if yt > 0:
                z = z + beta * (yt - z)

        f = max(0.0, p * z)
        out = te[["date", "product_family_name"]].copy()
        out["pred_tsb"] = f
        pred_list.append(out)

    return pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame(
        columns=["date", "product_family_name", "pred_tsb"]
    )


def _sba_baseline(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    alpha: float = 0.1,
    sparse_min_nonzero: int = 3,
    sparse_nonzero_rate: float = 0.01,
) -> pd.DataFrame:
    pred_list: List[pd.DataFrame] = []
    for sku, tr in train_panel.groupby("product_family_name"):
        te = test_panel[test_panel["product_family_name"] == sku].copy()
        if te.empty:
            continue

        y = tr.sort_values("date")["total_sales"].values.astype(float)
        if len(y) == 0:
            continue
        nonzero_idx = np.where(y > 0)[0]
        nonzero_cnt = len(nonzero_idx)
        nonzero_rate = float(np.mean(y > 0))

        if nonzero_cnt <= sparse_min_nonzero or nonzero_rate <= sparse_nonzero_rate:
            f = 0.0
        else:
            z = float(np.mean(y[y > 0]))
            if nonzero_cnt >= 2:
                intervals = np.diff(nonzero_idx)
                p = float(np.mean(intervals))
            else:
                p = float(len(y))
            if p <= 0:
                f = 0.0
            else:
                # SBA correction term over Croston-style ratio
                f = max(0.0, (1.0 - alpha / 2.0) * (z / p))

        out = te[["date", "product_family_name"]].copy()
        out["pred_sba"] = f
        pred_list.append(out)

    return pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame(
        columns=["date", "product_family_name", "pred_sba"]
    )


def _adida_baseline(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    agg_period: int = 7,
    sparse_min_nonzero: int = 3,
    sparse_nonzero_rate: float = 0.01,
) -> pd.DataFrame:
    pred_list: List[pd.DataFrame] = []
    for sku, tr in train_panel.groupby("product_family_name"):
        te = test_panel[test_panel["product_family_name"] == sku].copy()
        if te.empty:
            continue

        series = tr.sort_values("date")[["date", "total_sales"]].copy()
        y = series["total_sales"].values.astype(float)
        if len(y) == 0:
            continue

        nonzero_cnt = int(np.sum(y > 0))
        nonzero_rate = float(np.mean(y > 0))
        if nonzero_cnt <= sparse_min_nonzero or nonzero_rate <= sparse_nonzero_rate:
            daily_f = 0.0
        else:
            # Aggregate to coarse buckets, forecast at aggregate level, then disaggregate
            arr = y
            pad = (-len(arr)) % agg_period
            if pad > 0:
                arr = np.concatenate([arr, np.zeros(pad)])
            buckets = arr.reshape(-1, agg_period).sum(axis=1)
            if len(buckets) == 0:
                daily_f = 0.0
            elif len(buckets) == 1:
                daily_f = max(0.0, float(buckets[-1]) / agg_period)
            else:
                # Naive(1) on aggregated series for out-of-sample baseline
                daily_f = max(0.0, float(buckets[-1]) / agg_period)

        out = te[["date", "product_family_name"]].copy()
        out["pred_adida"] = daily_f
        pred_list.append(out)

    return pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame(
        columns=["date", "product_family_name", "pred_adida"]
    )


def _fit_two_stage_hgb(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: List[str],
    cls_params: Optional[Dict[str, Any]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
    tau: float = 0.0,
    alpha: float = 1.0,
    cap_value: float = np.inf,
    y_floor: float = 0.0,
    peak_prob_threshold: float = 1.0,
    peak_mult: float = 1.0,
    vol_scale: float = 0.0,
    vol_clip: float = 3.0,
    sparse_skus: Optional[set[str]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = train_feat[feature_cols].astype(float)
    y_cls = train_feat["is_sale"].astype(int).values
    X_test = test_feat[feature_cols].astype(float)

    if cls_params is None:
        cls_params = {
            "max_depth": 4,
            "learning_rate": 0.03,
            "max_iter": 250,
            "min_samples_leaf": 100,
        }
    if reg_params is None:
        reg_params = {
            "max_depth": 4,
            "learning_rate": 0.03,
            "max_iter": 300,
            "min_samples_leaf": 80,
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
    if vol_scale > 0.0:
        roll_mean = test_feat["roll_mean_14"].astype(float).values
        roll_std = test_feat["roll_std_14"].astype(float).values
        rel_vol = np.clip(roll_std / np.maximum(roll_mean, 1.0), 0.0, vol_clip)
        pred_final = pred_final * (1.0 + vol_scale * rel_vol)
    pred_final = np.clip(pred_final, 0.0, cap_value)
    if y_floor > 0:
        pred_final = np.where(pred_final < y_floor, 0.0, pred_final)

    if sparse_skus:
        sku_test = test_feat["product_family_name"].astype(str).values
        sparse_mask = np.array([s in sparse_skus for s in sku_test], dtype=bool)
        pred_final[sparse_mask] = 0.0

    return p_sale, pred_pos, pred_final


def _split_train_val_by_time(train_feat: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train_feat["date"].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * train_ratio))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_feat[train_feat["date"] <= cut_date].copy()
    va = train_feat[train_feat["date"] > cut_date].copy()
    return tr, va


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


def _volatility_under_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_std_ratio: float = 0.65,
    eps: float = 1.0,
) -> float:
    # Penalize under-volatile predictions for sparse series with bursty peaks.
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    ratio = std_pred / max(std_true, eps)
    shortfall = max(0.0, min_std_ratio - ratio)
    return shortfall * 100.0


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
        {"max_depth": 3, "learning_rate": 0.03, "max_iter": 250, "min_samples_leaf": 120},
        {"max_depth": 4, "learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 90},
        {"max_depth": 5, "learning_rate": 0.03, "max_iter": 350, "min_samples_leaf": 70},
        {"max_depth": 6, "learning_rate": 0.02, "max_iter": 450, "min_samples_leaf": 60},
    ]
    reg_grid = [
        {"loss": "squared_error", "max_depth": 3, "learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 120},
        {"loss": "squared_error", "max_depth": 4, "learning_rate": 0.03, "max_iter": 350, "min_samples_leaf": 80},
        {"loss": "quantile", "quantile": 0.70, "max_depth": 5, "learning_rate": 0.03, "max_iter": 400, "min_samples_leaf": 60},
        {"loss": "quantile", "quantile": 0.80, "max_depth": 5, "learning_rate": 0.03, "max_iter": 450, "min_samples_leaf": 50},
        {"loss": "quantile", "quantile": 0.90, "max_depth": 6, "learning_rate": 0.02, "max_iter": 500, "min_samples_leaf": 40},
    ]
    tau_grid = [0.00, 0.05, 0.10, 0.15]
    alpha_grid = [0.9, 1.1, 1.3, 1.5]
    cap_q_grid = [0.995, 0.999, 1.0]
    y_floor_grid = [0.0, 0.25, 0.5]
    peak_prob_threshold_grid = [1.0, 0.70, 0.80, 0.90]
    peak_mult_grid = [1.0, 1.20, 1.40]
    vol_scale_grid = [0.0, 0.20, 0.40]
    rng = np.random.RandomState(random_state)

    tr_positive = tr[tr["y"] > 0]["y"].values
    if len(tr_positive) == 0:
        default_cfg = {
            "cls_params": cls_grid[1],
            "reg_params": reg_grid[1],
            "tau": 0.2,
            "alpha": 1.0,
            "cap_q": 1.0,
            "cap_value": np.inf,
            "y_floor": 0.0,
            "peak_prob_threshold": 1.0,
            "peak_mult": 1.0,
            "vol_scale": 0.0,
            "vol_clip": 3.0,
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
                product(
                    cap_q_grid,
                    tau_grid,
                    alpha_grid,
                    y_floor_grid,
                    peak_prob_threshold_grid,
                    peak_mult_grid,
                    vol_scale_grid,
                )
            )
            if len(param_combos) > max_trials_per_model_pair:
                selected_idx = rng.choice(len(param_combos), size=max_trials_per_model_pair, replace=False)
                selected_combos = [param_combos[i] for i in selected_idx]
            else:
                selected_combos = param_combos

            va_mean = va["roll_mean_14"].astype(float).values
            va_std = va["roll_std_14"].astype(float).values
            base_rel_vol = np.clip(va_std / np.maximum(va_mean, 1.0), 0.0, 3.0)

            for cap_q, tau, alpha, y_floor, peak_prob_threshold, peak_mult, vol_scale in selected_combos:
                cap_value = np.quantile(tr_positive, cap_q) if cap_q < 1.0 else np.inf

                pred = np.where(p_va < tau, 0.0, raw_pred)
                pred = alpha * pred
                if peak_mult > 1.0 and peak_prob_threshold < 1.0:
                    pred = np.where(p_va >= peak_prob_threshold, pred * peak_mult, pred)
                if vol_scale > 0.0:
                    pred = pred * (1.0 + vol_scale * base_rel_vol)
                pred = np.clip(pred, 0.0, cap_value)
                if y_floor > 0:
                    pred = np.where(pred < y_floor, 0.0, pred)

                metric_bundle = compute_metric_bundle(
                    y_true=y_va,
                    y_pred=pred,
                    y_true_sale=(y_va > 0).astype(int),
                    metric_name=metric_name,
                    eps=eps_mape,
                )
                metric_score = metric_bundle["MAPE_0_100"]
                metric_wmape = metric_bundle["WMAPE_0_100"]
                pos_mape = metric_bundle["POSITIVE_ONLY_MAPE_PCT"]
                objective_value = _objective_from_bundle(metric_bundle, tuning_objective=tuning_objective)
                peak_pen = _peak_underprediction_penalty(
                    y_true=y_va,
                    y_pred=pred,
                    q=0.9,
                    eps=eps_mape,
                )
                vol_pen = _volatility_under_penalty(
                    y_true=y_va,
                    y_pred=pred,
                    min_std_ratio=0.65,
                    eps=eps_mape,
                )
                # Optimize closeness via WMAPE-centered objective with sparse-peak regularization.
                score = objective_value + 0.15 * peak_pen + 0.10 * vol_pen
                row = {
                    "score": score,
                    "objective_value": objective_value,
                    "objective_name": tuning_objective,
                    "metric_mape": metric_score,
                    "metric_wmape": metric_wmape,
                    "positive_only_mape": pos_mape,
                    "peak_under_penalty": peak_pen,
                    "volatility_under_penalty": vol_pen,
                    "cls_params": cls_params,
                    "reg_params": reg_params,
                    "tau": tau,
                    "alpha": alpha,
                    "cap_q": cap_q,
                    "cap_value": cap_value,
                    "y_floor": y_floor,
                    "peak_prob_threshold": peak_prob_threshold,
                    "peak_mult": peak_mult,
                    "vol_scale": vol_scale,
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
                        "y_floor": y_floor,
                        "peak_prob_threshold": peak_prob_threshold,
                        "peak_mult": peak_mult,
                        "vol_scale": vol_scale,
                        "vol_clip": 3.0,
                    }

    trials_df = pd.DataFrame(trial_rows).sort_values("score").reset_index(drop=True)
    if best_cfg is None:
        best_cfg = {
            "cls_params": cls_grid[1],
            "reg_params": reg_grid[1],
            "tau": 0.2,
            "alpha": 1.0,
            "cap_q": 1.0,
            "cap_value": np.inf,
            "y_floor": 0.0,
            "peak_prob_threshold": 1.0,
            "peak_mult": 1.0,
            "vol_scale": 0.0,
            "vol_clip": 3.0,
        }
    return best_cfg, trials_df


def _score_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    eps_mape: float,
) -> float:
    if metric_name == "bounded_mape":
        return float(bounded_mape(y_true, y_pred, eps=eps_mape, ape_cap=100.0))
    if metric_name == "safe_mape":
        return float(safe_mape(y_true, y_pred, eps=eps_mape))
    raise ValueError("metric_name must be one of: bounded_mape, safe_mape")


def _choose_best_sparse_baseline(
    train_panel: pd.DataFrame,
    metric_name: str,
    eps_mape: float,
) -> Tuple[str, pd.DataFrame]:
    # Time-based split inside train to select baseline model class
    unique_dates = np.array(sorted(train_panel["date"].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * 0.8))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_panel[train_panel["date"] <= cut_date].copy()
    va = train_panel[train_panel["date"] > cut_date].copy()
    y_va = va["total_sales"].values.astype(float)

    tsb = _tsb_baseline(tr, va, alpha=0.1, beta=0.1)
    sba = _sba_baseline(tr, va, alpha=0.1)
    adida = _adida_baseline(tr, va, agg_period=7)

    merged = va[["date", "product_family_name", "total_sales"]].copy()
    merged = merged.merge(tsb, on=["date", "product_family_name"], how="left")
    merged = merged.merge(sba, on=["date", "product_family_name"], how="left")
    merged = merged.merge(adida, on=["date", "product_family_name"], how="left")
    merged = merged.fillna(0.0)

    baseline_scores = [
        {"baseline": "tsb", "score": _score_array(y_va, merged["pred_tsb"].values, metric_name, eps_mape)},
        {"baseline": "sba", "score": _score_array(y_va, merged["pred_sba"].values, metric_name, eps_mape)},
        {"baseline": "adida", "score": _score_array(y_va, merged["pred_adida"].values, metric_name, eps_mape)},
    ]
    baseline_df = pd.DataFrame(baseline_scores).sort_values("score").reset_index(drop=True)
    best_name = str(baseline_df.iloc[0]["baseline"])
    return best_name, baseline_df


def _find_sparse_skus(
    train_panel: pd.DataFrame,
    sparse_min_nonzero: int = 3,
    sparse_nonzero_rate: float = 0.01,
) -> set[str]:
    stats = train_panel.groupby("product_family_name").agg(
        n=("total_sales", "size"),
        nnz=("total_sales", lambda x: int(np.sum(np.asarray(x) > 0))),
    )
    stats["nnz_rate"] = stats["nnz"] / stats["n"]
    sparse = stats[(stats["nnz"] <= sparse_min_nonzero) | (stats["nnz_rate"] <= sparse_nonzero_rate)]
    return set(sparse.index.astype(str).tolist())


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


def run_c3_pipeline(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 3,
    n_periods: int = 4,
    eps_mape: float = 1.0,
    metric_name: str = "bounded_mape",
    tune: bool = True,
    tuning_objective: str = "wmape",
    max_trials_per_model_pair: int = 500,
    feature_cols: List[str] | None = None,
    random_state: int = 42,
    prediction_output_path: str | Path = "forecasting/c3_prediction.parquet",
) -> C3Artifacts:
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    train_daily, test_daily = _load_daily(train_path, test_path)
    train_raw = train_daily[train_daily["cluster"] == cluster_id].copy()
    test_raw = test_daily[test_daily["cluster"] == cluster_id].copy()

    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test.")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_feat, test_feat = _build_features(train_panel, test_panel)

    # Baseline family for sparse intermittent demand: TSB / SBA / ADIDA
    selected_baseline_name, baseline_comparison = _choose_best_sparse_baseline(
        train_panel=train_panel,
        metric_name=metric_name,
        eps_mape=eps_mape,
    )
    tsb_pred = _tsb_baseline(train_panel, test_panel, alpha=0.1, beta=0.1)
    sba_pred = _sba_baseline(train_panel, test_panel, alpha=0.1)
    adida_pred = _adida_baseline(train_panel, test_panel, agg_period=7)
    sparse_skus = _find_sparse_skus(train_panel)

    tuning_trials: Optional[pd.DataFrame] = None
    tuning_best_config: Optional[Dict[str, Any]] = None

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
            y_floor=tuning_best_config["y_floor"],
            peak_prob_threshold=tuning_best_config["peak_prob_threshold"],
            peak_mult=tuning_best_config["peak_mult"],
            vol_scale=tuning_best_config["vol_scale"],
            vol_clip=tuning_best_config["vol_clip"],
            sparse_skus=sparse_skus,
            random_state=random_state,
        )
    else:
        p_sale, pred_pos, pred_two_stage = _fit_two_stage_hgb(
            train_feat=train_feat,
            test_feat=test_feat,
            feature_cols=feature_cols,
            tau=0.2,
            y_floor=1.0,
            peak_prob_threshold=0.8,
            peak_mult=1.1,
            vol_scale=0.2,
            vol_clip=3.0,
            sparse_skus=sparse_skus,
            random_state=random_state,
        )

    pred_df = test_feat[
        ["date", "product_family_name", "cluster", "y", "is_sale"]
    ].copy()
    pred_df = pred_df.merge(tsb_pred, on=["date", "product_family_name"], how="left")
    pred_df = pred_df.merge(sba_pred, on=["date", "product_family_name"], how="left")
    pred_df = pred_df.merge(adida_pred, on=["date", "product_family_name"], how="left")
    pred_df["pred_tsb"] = pred_df["pred_tsb"].fillna(0.0)
    pred_df["pred_sba"] = pred_df["pred_sba"].fillna(0.0)
    pred_df["pred_adida"] = pred_df["pred_adida"].fillna(0.0)
    pred_df["p_sale"] = p_sale
    pred_df["pred_pos_if_sale"] = pred_pos
    pred_df["pred_two_stage"] = pred_two_stage

    baseline_col_map = {
        "tsb": "pred_tsb",
        "sba": "pred_sba",
        "adida": "pred_adida",
    }
    selected_baseline_col = baseline_col_map[selected_baseline_name]

    overall_rows = []
    for method, col in [(selected_baseline_name, selected_baseline_col), ("two_stage_hgb", "pred_two_stage")]:
        metric_bundle = compute_metric_bundle(
            y_true=pred_df["y"].values,
            y_pred=pred_df[col].values,
            y_true_sale=pred_df["is_sale"].values,
            metric_name=metric_name,
            eps=eps_mape,
        )
        overall_rows.append({"method": method, **metric_bundle})
    metrics_overall = pd.DataFrame(overall_rows).sort_values("MAPE_0_100")

    pred_df = _periodize_test(pred_df, n_periods=n_periods)

    period_rows = []
    ape_rows = []
    for method, col in [(selected_baseline_name, selected_baseline_col), ("two_stage_hgb", "pred_two_stage")]:
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
    for method, col in [(selected_baseline_name, selected_baseline_col), ("two_stage_hgb", "pred_two_stage")]:
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

    ape_box_df_trimmed = ape_box_df.copy()
    if not ape_box_df_trimmed.empty:
        keep_idx = []
        for (m, p), grp in ape_box_df_trimmed.groupby(["method", "period"]):
            cap = np.quantile(grp["APE_0_100"].values, 0.99)
            keep_idx.extend(grp.index[grp["APE_0_100"] <= cap].tolist())
        ape_box_df_trimmed = ape_box_df_trimmed.loc[keep_idx].copy()

    error_quantiles = _build_error_quantiles(ape_box_df)
    metrics_by_period = (
        pd.DataFrame(period_rows)
        .sort_values(["method", "period"])
        .reset_index(drop=True)
    )

    prediction_output_path = str(prediction_output_path)
    Path(prediction_output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(prediction_output_path, index=False)

    return C3Artifacts(
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
        baseline_comparison=baseline_comparison,
        selected_baseline_name=selected_baseline_name,
    )
