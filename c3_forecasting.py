from __future__ import annotations

"""C3 forecasting pipeline with naive7 baseline vs two-stage HGB model.

This module exposes the final pairwise Cluster 3 comparison requested by the
project while preserving the leak-safe recursive forecasting workflow.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


FEATURE_COLS: List[str] = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_month_start",
    "is_quarter_end_month",
    "is_nov_dec",
    "is_year_end_week",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_nonzero_7",
    "rolling_nonzero_14",
    "log1p_rolling_mean_7",
    "days_since_last_sale",
]

FIT_SEARCH_GRID: List[Dict[str, Any]] = [
    {
        "clf_max_depth": 5,
        "clf_l2": 1.0,
        "reg_max_depth": 5,
        "reg_l2": 1.0,
        "min_samples_leaf": min_samples_leaf,
        "lognormal_correction": lognormal_correction,
    }
    for min_samples_leaf in [20]
    for lognormal_correction in [True]
]

DECISION_SEARCH_GRID: List[Dict[str, Any]] = [
    {
        "occurrence_threshold": occurrence_threshold,
        "prediction_floor": prediction_floor,
        "seasonal_anchor_weight": seasonal_anchor_weight,
    }
    for occurrence_threshold in [0.15, 0.20]
    for prediction_floor in [0.00]
    for seasonal_anchor_weight in [0.35, 0.40]
]
DEFAULT_TUNING_OBJECTIVE = "wmape"


@dataclass
class C3Artifact:
    pred_df: pd.DataFrame
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_period: pd.DataFrame
    error_quantiles: pd.DataFrame
    ape_box_df: pd.DataFrame
    tuning_best_config: Dict[str, Any]
    tuning_trials: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    prediction_output_path: Optional[str] = None
    metrics_output_path: Optional[str] = None


def _load_cluster(path: str | Path, cluster_id: int) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df.columns = [c.lower().strip() for c in df.columns]

    rename_map = {}
    if "total_sales" in df.columns:
        rename_map["total_sales"] = "y"
    elif "quantity" in df.columns:
        rename_map["quantity"] = "y"
    elif "sales" in df.columns:
        rename_map["sales"] = "y"

    if "product_family_name" in df.columns:
        rename_map["product_family_name"] = "sku_id"
    elif "stockcode" in df.columns:
        rename_map["stockcode"] = "sku_id"
    elif "stock_code" in df.columns:
        rename_map["stock_code"] = "sku_id"

    if "cluster_kmeans" in df.columns and "cluster" not in df.columns:
        rename_map["cluster_kmeans"] = "cluster"

    df = df.rename(columns=rename_map)

    for col in ["date", "sku_id", "cluster", "y"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}'. Available columns: {df.columns.tolist()}"
            )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sku_id"] = df["sku_id"].astype("string").str.strip()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = df[df["cluster"] == cluster_id].copy()
    df = (
        df.groupby(["date", "sku_id", "cluster"], as_index=False)["y"]
        .sum()
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )
    return df


def _build_zero_filled_panel(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sku_map = (
        pd.concat(
            [
                train_raw[["sku_id", "cluster"]],
                test_raw[["sku_id", "cluster"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("sku_id")
        .reset_index(drop=True)
    )
    sku_list = sku_map["sku_id"].tolist()

    train_dates = pd.date_range(train_raw["date"].min(), train_raw["date"].max(), freq="D")
    test_dates = pd.date_range(test_raw["date"].min(), test_raw["date"].max(), freq="D")

    train_grid = pd.MultiIndex.from_product(
        [sku_list, train_dates],
        names=["sku_id", "date"],
    ).to_frame(index=False)
    test_grid = pd.MultiIndex.from_product(
        [sku_list, test_dates],
        names=["sku_id", "date"],
    ).to_frame(index=False)

    train_grid = train_grid.merge(sku_map, on="sku_id", how="left")
    test_grid = test_grid.merge(sku_map, on="sku_id", how="left")

    train_panel = train_grid.merge(train_raw, on=["date", "sku_id", "cluster"], how="left")
    test_panel = test_grid.merge(test_raw, on=["date", "sku_id", "cluster"], how="left")
    train_panel["y"] = train_panel["y"].fillna(0.0)
    test_panel["y"] = test_panel["y"].fillna(0.0)
    return train_panel, test_panel


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out["date"]
    out["day_of_week"] = d.dt.dayofweek.astype(np.int8)
    out["day_of_month"] = d.dt.day.astype(np.int8)
    out["month"] = d.dt.month.astype(np.int8)
    out["week_of_year"] = d.dt.isocalendar().week.astype(np.int16)
    out["is_month_start"] = (d.dt.day <= 5).astype(np.int8)
    out["is_quarter_end_month"] = d.dt.month.isin([3, 6, 9, 12]).astype(np.int8)
    out["is_nov_dec"] = d.dt.month.isin([11, 12]).astype(np.int8)
    out["is_year_end_week"] = (out["week_of_year"] >= 50).astype(np.int8)
    return out


def _dsls_vectorized(sale_flag: np.ndarray) -> np.ndarray:
    n = len(sale_flag)
    idx = np.arange(n)
    last_sale_pos = np.where(sale_flag.astype(bool), idx, -1)
    last_sale_cummax = np.maximum.accumulate(last_sale_pos)
    dsls = np.where(last_sale_cummax < 0, 999.0, (idx - last_sale_cummax).astype(float))
    return dsls.astype(np.float32)


def _sku_lag_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("date").copy()
    y_s1 = grp["y"].shift(1).fillna(0.0).clip(lower=0.0)

    grp["lag_1"] = grp["y"].shift(1).fillna(0.0).clip(lower=0.0)
    for lag in (7, 14, 28):
        grp[f"lag_{lag}"] = grp["y"].shift(lag).fillna(0.0).clip(lower=0.0)

    for window in (7, 14):
        grp[f"rolling_mean_{window}"] = (
            y_s1.rolling(window, min_periods=1).mean().fillna(0.0)
        )
        grp[f"rolling_nonzero_{window}"] = (
            (y_s1 > 0).rolling(window, min_periods=1).mean().fillna(0.0)
        )

    grp["log1p_rolling_mean_7"] = np.log1p(grp["rolling_mean_7"])
    grp["days_since_last_sale"] = _dsls_vectorized((y_s1.values > 0).astype(int))
    return grp


def _build_feature_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_calendar(df.copy())
    out = out.sort_values(["sku_id", "date"]).reset_index(drop=True)
    parts = []
    for _, grp in out.groupby("sku_id", sort=False):
        parts.append(_sku_lag_features(grp))
    out = pd.concat(parts, ignore_index=True)
    for col in FEATURE_COLS:
        out[col] = out[col].fillna(0.0)
    return out


def _build_diagnostic_features(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_panel.copy()
    test_df = test_panel.copy()
    train_df["_split"] = 0
    test_df["_split"] = 1
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined_feat = _build_feature_panel(combined)
    train_out = combined_feat[combined_feat["_split"] == 0].drop(columns=["_split"]).copy()
    test_out = combined_feat[combined_feat["_split"] == 1].drop(columns=["_split"]).copy()
    return train_out, test_out


def _fit_model(
    train_feat: pd.DataFrame,
    clf_params: Dict[str, Any],
    reg_params: Dict[str, Any],
    lognormal_correction: bool,
) -> Dict[str, Any]:
    feat_cols = [c for c in FEATURE_COLS if c in train_feat.columns]
    X_tr = train_feat[feat_cols].to_numpy(dtype=np.float32)
    y_tr = train_feat["y"].to_numpy(dtype=np.float64)
    y_occ = (y_tr > 0).astype(int)

    if len(np.unique(y_occ)) == 1:
        clf = None
        p_sale_const = float(y_occ[0])
    else:
        clf = HistGradientBoostingClassifier(**clf_params)
        clf.fit(X_tr, y_occ)
        p_sale_const = None

    pos_mask = y_tr > 0
    if int(pos_mask.sum()) == 0:
        reg = None
        sigma2 = 0.0
    else:
        X_pos = X_tr[pos_mask]
        log_y_pos = np.log1p(y_tr[pos_mask])
        reg = HistGradientBoostingRegressor(**reg_params)
        reg.fit(X_pos, log_y_pos)
        if lognormal_correction and int(pos_mask.sum()) > 20:
            log_resid = log_y_pos - reg.predict(X_pos)
            sigma2 = float(np.var(log_resid, ddof=1))
        else:
            sigma2 = 0.0

    return {
        "feat_cols": feat_cols,
        "clf": clf,
        "reg": reg,
        "p_sale_const": p_sale_const,
        "sigma2": float(max(sigma2, 0.0)),
    }


def _predict_from_model(
    fit_obj: Dict[str, Any],
    feat_df: pd.DataFrame,
    occurrence_threshold: float = 0.50,
    prediction_floor: float = 0.0,
    seasonal_anchor_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    X = feat_df[fit_obj["feat_cols"]].to_numpy(dtype=np.float32)

    if fit_obj["clf"] is None:
        p_sale = np.full(len(feat_df), float(fit_obj["p_sale_const"]), dtype=np.float32)
    else:
        p_sale = fit_obj["clf"].predict_proba(X)[:, 1].astype(np.float32)

    if fit_obj["reg"] is None:
        qty_pred = np.zeros(len(feat_df), dtype=np.float64)
    else:
        log_qty_pred = fit_obj["reg"].predict(X).astype(np.float64)
        qty_pred = np.expm1(log_qty_pred + 0.5 * float(fit_obj["sigma2"])).clip(min=0.0)

    pred = (p_sale.astype(np.float64) * qty_pred).clip(min=0.0)
    if seasonal_anchor_weight > 0:
        lag7_anchor = feat_df["lag_7"].to_numpy(dtype=np.float64)
        pred = (1.0 - seasonal_anchor_weight) * pred + seasonal_anchor_weight * lag7_anchor
    pred = np.where(p_sale >= occurrence_threshold, pred, 0.0)
    if prediction_floor > 0:
        pred = np.where(pred >= prediction_floor, pred, 0.0)
    pred = pred.astype(np.float32)
    return pred, p_sale


def _tail_mean(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values[-min(window, len(values)):], dtype=float)
    return float(np.mean(arr))


def _tail_nonzero(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values[-min(window, len(values)):], dtype=float)
    return float(np.mean(arr > 0))


def _days_since_last_sale_from_occ(occ_hist: List[int]) -> float:
    if not occ_hist or not any(occ_hist):
        return 999.0
    last_idx = max(i for i, flag in enumerate(occ_hist) if flag > 0)
    return float((len(occ_hist) - 1) - last_idx)


def _history_feature_row(
    current_date: pd.Timestamp,
    sales_hist: List[float],
    occ_hist: List[int],
) -> Dict[str, float]:
    day_of_week = int(current_date.dayofweek)
    day_of_month = int(current_date.day)
    month = int(current_date.month)
    week_of_year = int(current_date.isocalendar().week)

    row: Dict[str, float] = {
        "day_of_week": float(day_of_week),
        "day_of_month": float(day_of_month),
        "month": float(month),
        "week_of_year": float(week_of_year),
        "is_month_start": float(day_of_month <= 5),
        "is_quarter_end_month": float(month in [3, 6, 9, 12]),
        "is_nov_dec": float(month in [11, 12]),
        "is_year_end_week": float(week_of_year >= 50),
        "lag_1": float(sales_hist[-1]) if len(sales_hist) >= 1 else 0.0,
        "lag_7": float(sales_hist[-7]) if len(sales_hist) >= 7 else 0.0,
        "lag_14": float(sales_hist[-14]) if len(sales_hist) >= 14 else 0.0,
        "lag_28": float(sales_hist[-28]) if len(sales_hist) >= 28 else 0.0,
        "rolling_mean_7": _tail_mean(sales_hist, 7),
        "rolling_mean_14": _tail_mean(sales_hist, 14),
        "rolling_nonzero_7": _tail_nonzero(sales_hist, 7),
        "rolling_nonzero_14": _tail_nonzero(sales_hist, 14),
        "days_since_last_sale": _days_since_last_sale_from_occ(occ_hist),
    }
    row["log1p_rolling_mean_7"] = float(np.log1p(row["rolling_mean_7"]))
    return row


def _recursive_forecast(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    fit_obj: Dict[str, Any],
    occurrence_threshold: float = 0.50,
    prediction_floor: float = 0.0,
    seasonal_anchor_weight: float = 0.0,
) -> pd.DataFrame:
    cluster_map = (
        train_panel[["sku_id", "cluster"]]
        .drop_duplicates()
        .set_index("sku_id")["cluster"]
        .to_dict()
    )
    sku_list = sorted(cluster_map.keys())

    history_sales: Dict[str, List[float]] = {
        sku: train_panel.loc[train_panel["sku_id"] == sku, "y"].astype(float).tolist()
        for sku in sku_list
    }
    history_occ: Dict[str, List[int]] = {
        sku: [int(v > 0) for v in history_sales[sku]]
        for sku in sku_list
    }

    rows: List[pd.DataFrame] = []
    test_dates = sorted(pd.to_datetime(test_panel["date"].unique()))
    for current_date in test_dates:
        feat_rows: List[Dict[str, Any]] = []
        for sku in sku_list:
            row = _history_feature_row(current_date, history_sales[sku], history_occ[sku])
            row["date"] = current_date
            row["sku_id"] = sku
            row["cluster"] = cluster_map[sku]
            feat_rows.append(row)

        feat_df = pd.DataFrame(feat_rows)
        pred, p_sale = _predict_from_model(
            fit_obj,
            feat_df,
            occurrence_threshold=occurrence_threshold,
            prediction_floor=prediction_floor,
            seasonal_anchor_weight=seasonal_anchor_weight,
        )
        feat_df["pred_two_stage"] = pred
        feat_df["p_sale"] = p_sale
        rows.append(feat_df[["date", "sku_id", "cluster", "pred_two_stage", "p_sale"]])

        for sku, pred_value, p_value in zip(
            feat_df["sku_id"],
            pred.tolist(),
            p_sale.tolist(),
        ):
            history_sales[sku].append(float(pred_value))
            history_occ[sku].append(int((p_value >= occurrence_threshold) and (pred_value > 0)))

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df = pred_df.merge(
        test_panel[["date", "sku_id", "cluster", "y"]],
        on=["date", "sku_id", "cluster"],
        how="left",
    )
    pred_df["y"] = pred_df["y"].fillna(0.0).astype(float)
    return pred_df


def _build_naive7_predictions(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for sku, tr_sku in train_panel.groupby("sku_id", sort=False):
        te_sku = test_panel[test_panel["sku_id"] == sku].sort_values("date")
        horizon = len(te_sku)
        if horizon <= 0:
            continue
        train_series = tr_sku.sort_values("date")["y"].to_numpy(dtype=float)
        if train_series.size == 0:
            pred = np.zeros(horizon, dtype=np.float32)
        else:
            pattern_len = min(7, train_series.size)
            pattern = train_series[-pattern_len:]
            pred = np.tile(pattern, int(np.ceil(horizon / pattern_len)))[:horizon].astype(np.float32)
        part = te_sku[["date", "sku_id", "cluster"]].copy()
        part["pred_naive7"] = pred
        rows.append(part)

    return pd.concat(rows, ignore_index=True).sort_values(["sku_id", "date"]).reset_index(drop=True)


def _ape_eps(y: np.ndarray, yhat: np.ndarray, eps: float) -> np.ndarray:
    return np.abs(y - yhat) / np.maximum(np.abs(y), eps) * 100.0


def _wmape(y: np.ndarray, yhat: np.ndarray, eps: float) -> float:
    return float(100.0 * np.sum(np.abs(y - yhat)) / max(float(np.sum(np.abs(y))), eps))


def _non_zero_mape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0)
    if int(np.sum(mask)) == 0:
        return float("nan")
    denom = np.maximum(np.abs(y[mask]), eps)
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom) * 100.0)


def _score(y: np.ndarray, yhat: np.ndarray, eps: float, name: str) -> float:
    if name.lower() == "wmape":
        return _wmape(y, yhat, eps)
    ape = _ape_eps(y, yhat, eps)
    return float(np.mean(np.clip(ape, 0.0, 100.0)))


def _false_positive_rate_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    zero_mask = y <= 0
    if int(np.sum(zero_mask)) == 0:
        return 0.0
    fp = np.sum((yhat > 0) & zero_mask)
    return float(100.0 * fp / np.sum(zero_mask))


def _flatness_penalty(
    pred_df: pd.DataFrame,
    min_level_ratio: float = 0.45,
    min_diff_ratio: float = 0.50,
    eps: float = 1.0,
) -> float:
    """
    Penalize predictions that are too flat relative to actual daily aggregate variation.

    A straight-line forecast typically shows both:
    1) low day-level aggregate volatility, and
    2) low day-to-day movement.
    We penalize both shortfalls to avoid selecting overly smooth recursive paths.
    """
    daily = (
        pred_df.groupby("date", as_index=False)[["y", "pred_two_stage"]]
        .sum()
        .sort_values("date")
    )
    actual = daily["y"].to_numpy(dtype=float)
    pred = daily["pred_two_stage"].to_numpy(dtype=float)

    std_actual = float(np.std(actual))
    std_pred = float(np.std(pred))
    level_ratio = std_pred / max(std_actual, eps)
    level_shortfall = max(0.0, min_level_ratio - level_ratio)

    if len(actual) >= 2:
        diff_actual = np.diff(actual)
        diff_pred = np.diff(pred)
        diff_actual_std = float(np.std(diff_actual))
        diff_pred_std = float(np.std(diff_pred))
        diff_ratio = diff_pred_std / max(diff_actual_std, eps)
        diff_shortfall = max(0.0, min_diff_ratio - diff_ratio)
    else:
        diff_shortfall = 0.0

    return float(100.0 * (level_shortfall + 1.25 * diff_shortfall))


def _ensure_periods(df: pd.DataFrame, n_periods: int) -> pd.DataFrame:
    if "period" in df.columns:
        return df
    out = df.copy()
    dates = sorted(out["date"].unique())
    chunks = np.array_split(dates, n_periods)
    mapper: Dict[Any, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        for d in chunk:
            mapper[pd.Timestamp(d)] = f"P{i}"
    out["period"] = out["date"].map(mapper)
    return out


def _build_metrics(
    pred_df: pd.DataFrame,
    baseline_col: str,
    model_col: str,
    eps: float,
    n_periods: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_periods(pred_df.copy(), n_periods)
    methods = {"baseline": baseline_col, "model": model_col}
    periods = [f"P{i + 1}" for i in range(n_periods)]

    overall_rows = []
    for method_name, col in methods.items():
        y = df["y"].to_numpy(dtype=np.float64)
        yhat = df[col].to_numpy(dtype=np.float64)
        ape = _ape_eps(y, yhat, eps)
        overall_rows.append(
            {
                "method": method_name,
                "MAPE_0_100": float(np.mean(np.clip(ape, 0.0, 100.0))),
                "WMAPE_0_100": _wmape(y, yhat, eps),
                "EPSILON_MAPE_PCT": float(np.mean(ape)),
                "NON_ZERO_MAPE_PCT": _non_zero_mape(y, yhat),
                "MAE": float(np.mean(np.abs(y - yhat))),
                "pred_nonzero_rate": float(np.mean(yhat > 0)),
                "actual_nonzero_rate": float(np.mean(y > 0)),
                "nonzero_rate_gap": float(np.mean(yhat > 0) - np.mean(y > 0)),
            }
        )
    metrics_overall = pd.DataFrame(overall_rows).sort_values("MAPE_0_100").reset_index(drop=True)

    period_rows = []
    for method_name, col in methods.items():
        for period in periods:
            sub = df[df["period"] == period]
            if sub.empty:
                continue
            y = sub["y"].to_numpy(dtype=np.float64)
            yhat = sub[col].to_numpy(dtype=np.float64)
            ape = _ape_eps(y, yhat, eps)
            period_rows.append(
                {
                    "method": method_name,
                    "period": period,
                    "MAPE_0_100": float(np.mean(np.clip(ape, 0.0, 100.0))),
                    "WMAPE_0_100": _wmape(y, yhat, eps),
                    "EPSILON_MAPE_PCT": float(np.mean(ape)),
                    "NON_ZERO_MAPE_PCT": _non_zero_mape(y, yhat),
                    "MAE": float(np.mean(np.abs(y - yhat))),
                }
            )
    metrics_by_period = pd.DataFrame(period_rows).sort_values(["method", "period"]).reset_index(drop=True)

    quantile_rows = []
    for method_name, col in methods.items():
        ape = _ape_eps(df["y"].to_numpy(dtype=np.float64), df[col].to_numpy(dtype=np.float64), eps)
        quantile_rows.append(
            {
                "method": method_name,
                "period": "overall",
                "p25": float(np.percentile(ape, 25)),
                "p50": float(np.percentile(ape, 50)),
                "p75": float(np.percentile(ape, 75)),
                "p90": float(np.percentile(ape, 90)),
                "p95": float(np.percentile(ape, 95)),
            }
        )
    error_quantiles = pd.DataFrame(quantile_rows)

    box_rows = []
    for method_name, col in methods.items():
        for period in periods:
            sub = df[df["period"] == period]
            if sub.empty:
                continue
            ape = _ape_eps(sub["y"].to_numpy(dtype=np.float64), sub[col].to_numpy(dtype=np.float64), eps)
            for value in ape:
                box_rows.append(
                    {
                        "method": method_name,
                        "period": period,
                        "APE_EPS_PCT": float(value),
                    }
                )
    ape_box_df = pd.DataFrame(box_rows)

    return metrics_overall, metrics_by_period, error_quantiles, ape_box_df


def _default_config() -> Dict[str, Any]:
    return {
        "clf_max_depth": 5,
        "clf_l2": 1.0,
        "reg_max_depth": 5,
        "reg_l2": 1.0,
        "min_samples_leaf": 20,
        "lognormal_correction": True,
        "occurrence_threshold": 0.20,
        "prediction_floor": 0.0,
        "seasonal_anchor_weight": 0.40,
    }


def _config_to_sklearn_params(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    clf_params = {
        "max_iter": 150,
        "max_depth": cfg["clf_max_depth"],
        "l2_regularization": cfg["clf_l2"],
        "min_samples_leaf": int(cfg["min_samples_leaf"]),
        "random_state": 42,
        "verbose": 0,
    }
    reg_params = {
        "max_iter": 150,
        "max_depth": cfg["reg_max_depth"],
        "l2_regularization": cfg["reg_l2"],
        "min_samples_leaf": int(cfg["min_samples_leaf"]),
        "loss": "squared_error",
        "random_state": 42,
        "verbose": 0,
    }
    return clf_params, reg_params, bool(cfg.get("lognormal_correction", True))


def _split_panel_by_holdout(
    train_panel: pd.DataFrame,
    holdout_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(train_panel["date"].unique())))
    if len(dates) <= holdout_days + 14:
        holdout_days = max(7, len(dates) // 5)
    cutoff_date = pd.Timestamp(dates[-holdout_days])
    pseudo_tr = train_panel[train_panel["date"] < cutoff_date].copy()
    pseudo_te = train_panel[train_panel["date"] >= cutoff_date].copy()
    return pseudo_tr, pseudo_te


def _tune(
    train_panel: pd.DataFrame,
    eps: float,
    tuning_objective: str,
    holdout_days: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pseudo_tr_panel, pseudo_te_panel = _split_panel_by_holdout(train_panel, holdout_days)
    if pseudo_tr_panel.empty or pseudo_te_panel.empty:
        return _default_config(), pd.DataFrame()
    if int((pseudo_te_panel["y"] > 0).sum()) == 0:
        return _default_config(), pd.DataFrame()

    pseudo_tr_feat = _build_feature_panel(pseudo_tr_panel)
    rows = []
    for fit_cfg in FIT_SEARCH_GRID:
        clf_params, reg_params, use_lnc = _config_to_sklearn_params(fit_cfg)
        try:
            fit_obj = _fit_model(pseudo_tr_feat, clf_params, reg_params, lognormal_correction=use_lnc)
        except Exception as exc:  # pragma: no cover
            rows.append({**fit_cfg, "score": np.nan, "_error": str(exc)})
            continue

        for decision_cfg in DECISION_SEARCH_GRID:
            cfg = {**fit_cfg, **decision_cfg}
            try:
                pred_df = _recursive_forecast(
                    pseudo_tr_panel,
                    pseudo_te_panel,
                    fit_obj,
                    occurrence_threshold=float(cfg["occurrence_threshold"]),
                    prediction_floor=float(cfg["prediction_floor"]),
                    seasonal_anchor_weight=float(cfg["seasonal_anchor_weight"]),
                )
                base_score = _score(
                    pred_df["y"].to_numpy(dtype=np.float64),
                    pred_df["pred_two_stage"].to_numpy(dtype=np.float64),
                    eps,
                    tuning_objective,
                )
                fp_rate_pct = _false_positive_rate_pct(
                    pred_df["y"].to_numpy(dtype=np.float64),
                    pred_df["pred_two_stage"].to_numpy(dtype=np.float64),
                )
                flatness_penalty = _flatness_penalty(pred_df)
                fp_penalty_weight = 0.20 if tuning_objective.lower() == "wmape" else 0.35
                score = float(base_score + fp_penalty_weight * fp_rate_pct + 0.30 * flatness_penalty)
                rows.append(
                    {
                        **cfg,
                        "score": score,
                        "base_score": float(base_score),
                        "tuning_objective": tuning_objective,
                        "fp_rate_pct": float(fp_rate_pct),
                        "flatness_penalty": float(flatness_penalty),
                    }
                )
            except Exception as exc:  # pragma: no cover
                rows.append({**cfg, "score": np.nan, "_error": str(exc)})

    trials = pd.DataFrame(rows)
    valid = trials.dropna(subset=["score"])
    if valid.empty:
        return _default_config(), trials

    best_row = valid.loc[valid["score"].idxmin()].to_dict()
    best_cfg = {key: best_row[key] for key in _default_config().keys()}
    return best_cfg, trials.sort_values("score").reset_index(drop=True)


def run_c3_forecasting_2(
    train_path: str | Path,
    test_path: str | Path,
    cluster_id: int = 3,
    n_periods: int = 4,
    eps_mape: float = 1.0,
    metric_name: str = "bounded_mape",
    tuning_objective: str = DEFAULT_TUNING_OBJECTIVE,
    search_enabled: bool = False,
    search_holdout_days: int = 42,
    save_predictions: bool = True,
    prediction_output_path: Optional[str] = None,
    metrics_output_path: Optional[str] = None,
    lognormal_correction: bool = True,
) -> C3Artifact:
    train_raw = _load_cluster(train_path, cluster_id)
    test_raw = _load_cluster(test_path, cluster_id)
    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)

    train_feat = _build_feature_panel(train_panel)
    _, test_feat = _build_diagnostic_features(train_panel, test_panel)

    baseline_df = _build_naive7_predictions(train_panel, test_panel)

    if search_enabled:
        best_config, trials_df = _tune(
            train_panel=train_panel,
            eps=eps_mape,
            tuning_objective=tuning_objective,
            holdout_days=search_holdout_days,
        )
    else:
        best_config = _default_config()
        trials_df = None

    clf_params, reg_params, use_lnc = _config_to_sklearn_params(best_config)
    use_lnc = bool(use_lnc and lognormal_correction)
    fit_obj = _fit_model(train_feat, clf_params, reg_params, lognormal_correction=use_lnc)
    pred_df = _recursive_forecast(
        train_panel,
        test_panel,
        fit_obj,
        occurrence_threshold=float(best_config["occurrence_threshold"]),
        prediction_floor=float(best_config["prediction_floor"]),
        seasonal_anchor_weight=float(best_config["seasonal_anchor_weight"]),
    )
    pred_df = pred_df.merge(baseline_df, on=["date", "sku_id", "cluster"], how="left")
    pred_df["pred_naive7"] = pred_df["pred_naive7"].fillna(0.0)
    pred_df = pred_df[
        ["date", "sku_id", "cluster", "y", "pred_naive7", "pred_two_stage", "p_sale"]
    ].copy()

    metrics_overall, metrics_by_period, error_quantiles, ape_box_df = _build_metrics(
        pred_df=pred_df,
        baseline_col="pred_naive7",
        model_col="pred_two_stage",
        eps=eps_mape,
        n_periods=n_periods,
    )

    pred_out: Optional[str] = None
    if save_predictions and prediction_output_path:
        Path(prediction_output_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(prediction_output_path, index=False)
        pred_out = str(prediction_output_path)

    metadata: Dict[str, Any] = {
        "cluster_id": cluster_id,
        "n_train_skus": int(train_panel["sku_id"].nunique()),
        "n_test_skus": int(test_panel["sku_id"].nunique()),
        "train_rows": int(len(train_panel)),
        "test_rows": int(len(test_panel)),
        "train_start": str(train_panel["date"].min().date()),
        "train_end": str(train_panel["date"].max().date()),
        "test_start": str(test_panel["date"].min().date()),
        "test_end": str(test_panel["date"].max().date()),
        "train_nonzero_rate": float((train_panel["y"] > 0).mean()),
        "test_nonzero_rate": float((test_panel["y"] > 0).mean()),
        "feature_cols": ", ".join(FEATURE_COLS),
        "n_features": len(FEATURE_COLS),
        "log_sigma": float(np.sqrt(max(float(fit_obj["sigma2"]), 0.0))),
        "lognormal_correction": use_lnc,
        "occurrence_threshold": float(best_config["occurrence_threshold"]),
        "prediction_floor": float(best_config["prediction_floor"]),
        "seasonal_anchor_weight": float(best_config["seasonal_anchor_weight"]),
        "search_enabled": bool(search_enabled),
        "search_holdout_days": int(search_holdout_days) if search_enabled else None,
        "metric_name": metric_name,
        "tuning_objective": tuning_objective,
        "eps_mape": float(eps_mape),
        "baseline_col": "pred_naive7",
        "model_col": "pred_two_stage",
        "split_policy": (
            "strict parquet train/test for final reporting; recursive test forecasting "
            "with no test leakage"
        ),
    }

    return C3Artifact(
        pred_df=pred_df,
        train_feat=train_feat,
        test_feat=test_feat,
        metrics_overall=metrics_overall,
        metrics_by_period=metrics_by_period,
        error_quantiles=error_quantiles,
        ape_box_df=ape_box_df,
        tuning_best_config=best_config,
        tuning_trials=trials_df,
        metadata=metadata,
        prediction_output_path=pred_out,
        metrics_output_path=None,
    )
