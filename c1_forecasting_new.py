from __future__ import annotations

"""C1 forecasting comparison script built from scratch.

This module compares several leak-safe daily forecasting methods for cluster C1:
- Seasonal naive (weekly)
- Simple moving average
- Teunter-Syntetos-Babai (TSB)
- ADIDA
- iMAPA
- Zero-Inflated Poisson (ZIP)
- Zero-Inflated Negative Binomial (ZINB)

The implementation deliberately avoids any test leakage:
- train features are built from training history only
- test features for count models are generated recursively using prior predictions only
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# ----------------------------
# Configuration
# ----------------------------
TRAIN_DEFAULT = "data/forecasting/train_daily.parquet"
TEST_DEFAULT = "data/forecasting/test_daily.parquet"
PREDICTION_OUTPUT_DEFAULT = "forecasting/c1_prediction_new.parquet"
METRICS_OUTPUT_DEFAULT = "forecasting/c1_metrics_new.csv"

COUNT_FEATURE_COLS: List[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_28",
    "sale_rate_7",
    "sale_rate_28",
    "days_since_last_sale_clip56",
    "dow_sin",
    "dow_cos",
    "week_sin",
    "week_cos",
    "is_weekend",
    "is_q4",
]

INFLATION_FEATURE_COLS: List[str] = [
    "lag_is_sale_1",
    "lag_is_sale_7",
    "lag_is_sale_14",
    "sale_rate_7",
    "sale_rate_28",
    "days_since_last_sale_clip56",
    "dow_sin",
    "dow_cos",
    "week_sin",
    "week_cos",
    "is_weekend",
    "is_q4",
]

MODEL_COLS_ORDER: List[str] = [
    "pred_naive7",
    "pred_sma28",
    "pred_tsb",
    "pred_adida",
    "pred_imapa",
    "pred_zip",
    "pred_zinb",
]


@dataclass
class C1NewArtifacts:
    cluster_id: int
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_panel: pd.DataFrame
    test_panel: pd.DataFrame
    train_features: pd.DataFrame
    pred_df: pd.DataFrame
    metrics_table: pd.DataFrame
    model_cols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    prediction_output_path: Optional[str] = None
    metrics_output_path: Optional[str] = None


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            csv_path = path.with_suffix(".csv")
            pickle_path = path.with_suffix(".pkl")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            if pickle_path.exists():
                return pd.read_pickle(pickle_path)
            raise RuntimeError(
                f"Could not read {path}. pandas parquet support is unavailable and no CSV/PKL fallback was found."
            ) from exc
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type for {path}")


def _save_table(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return str(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return str(csv_path)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return str(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return str(path)
    raise ValueError(f"Unsupported output file type for {path}")


def _load_daily(train_path: str | Path, test_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_daily = _read_table(train_path)
    test_daily = _read_table(test_path)

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
    sku_map = train_raw[["product_family_name", "cluster"]].drop_duplicates().copy()
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


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek.astype(int)
    df["dom"] = df["date"].dt.day.astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)

    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 7.0)
    df["week_sin"] = np.sin(2.0 * np.pi * df["weekofyear"] / 52.0)
    df["week_cos"] = np.cos(2.0 * np.pi * df["weekofyear"] / 52.0)
    return df


def _build_train_features(train_panel: pd.DataFrame) -> pd.DataFrame:
    df = train_panel.copy().sort_values(["product_family_name", "date"]).reset_index(drop=True)
    df["y"] = df["total_sales"].astype(float)
    df["is_sale"] = (df["y"] > 0).astype(int)
    df = _add_calendar_features(df)

    g = df.groupby("product_family_name", group_keys=False)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = g["y"].shift(lag)
        df[f"lag_is_sale_{lag}"] = g["is_sale"].shift(lag)

    for window in [7, 28]:
        df[f"roll_mean_{window}"] = g["y"].shift(1).rolling(window, min_periods=1).mean()
        df[f"sale_rate_{window}"] = g["is_sale"].shift(1).rolling(window, min_periods=1).mean()

    last_sale_date = df["date"].where(df["is_sale"] == 1)
    last_sale_date = last_sale_date.groupby(df["product_family_name"]).ffill().groupby(df["product_family_name"]).shift(1)
    days_since = (df["date"] - last_sale_date).dt.days
    df["days_since_last_sale"] = days_since.fillna(999).astype(int)
    df["days_since_last_sale_clip56"] = np.minimum(df["days_since_last_sale"], 56)

    feature_fill_zero_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_mean_") or c.startswith("sale_rate_")]
    df[feature_fill_zero_cols] = df[feature_fill_zero_cols].fillna(0.0)
    return df


# ----------------------------
# Baseline models
# ----------------------------
def _forecast_naive7(series: np.ndarray, horizon: int) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if series.size == 0:
        return np.zeros(horizon, dtype=float)
    pattern_len = min(7, series.size)
    pattern = series[-pattern_len:]
    repeats = int(np.ceil(horizon / pattern_len))
    return np.tile(pattern, repeats)[:horizon].astype(float)


def _forecast_simple_moving_average(series: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if series.size == 0:
        return np.zeros(horizon, dtype=float)
    value = float(np.mean(series[-min(window, series.size) :]))
    return np.repeat(value, horizon).astype(float)


def _forecast_tsb(series: np.ndarray, horizon: int, alpha: float = 0.2, beta: float = 0.1) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if y.size == 0 or np.all(y <= 0):
        return np.zeros(horizon, dtype=float)

    occ = (y > 0).astype(float)
    pos = y[y > 0]
    p = float(np.clip(occ.mean(), 1e-6, 1.0))
    z = float(pos.mean()) if pos.size else 0.0

    for val in y:
        o = 1.0 if val > 0 else 0.0
        p = p + beta * (o - p)
        if o > 0:
            z = z + alpha * (float(val) - z)
    return np.repeat(max(p * z, 0.0), horizon).astype(float)


def _aggregate_series(series: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if y.size == 0:
        return np.zeros(0, dtype=float)
    pad = (-len(y)) % k
    if pad:
        y = np.concatenate([np.zeros(pad, dtype=float), y])
    return y.reshape(-1, k).sum(axis=1)


def _forecast_adida(series: np.ndarray, horizon: int, k: int = 7, avg_periods: int = 4) -> np.ndarray:
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    agg = _aggregate_series(np.asarray(series, dtype=float), k)
    if agg.size == 0:
        return np.zeros(horizon, dtype=float)
    use = agg[-min(avg_periods, agg.size) :]
    daily_level = float(np.mean(use) / max(k, 1))
    return np.repeat(max(daily_level, 0.0), horizon).astype(float)


def _forecast_imapa(
    series: np.ndarray,
    horizon: int,
    aggregation_levels: Sequence[int] = (7, 14, 28),
    avg_periods: int = 4,
) -> np.ndarray:
    forecasts = [_forecast_adida(series, horizon, k=k, avg_periods=avg_periods) for k in aggregation_levels]
    if not forecasts:
        return np.zeros(horizon, dtype=float)
    return np.mean(np.vstack(forecasts), axis=0)


def _build_univariate_baselines(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    test_dates = np.sort(test_panel["date"].unique())

    for sku, tr_sku in train_panel.groupby("product_family_name"):
        te_sku = test_panel[test_panel["product_family_name"] == sku].sort_values("date")
        horizon = len(te_sku)
        train_series = tr_sku.sort_values("date")["total_sales"].to_numpy(dtype=float)

        part = te_sku[["date", "product_family_name", "cluster", "total_sales"]].copy()
        part["y"] = part["total_sales"].astype(float)
        part["pred_naive7"] = _forecast_naive7(train_series, horizon)
        part["pred_sma28"] = _forecast_simple_moving_average(train_series, horizon, window=28)
        part["pred_tsb"] = _forecast_tsb(train_series, horizon)
        part["pred_adida"] = _forecast_adida(train_series, horizon, k=7)
        part["pred_imapa"] = _forecast_imapa(train_series, horizon)
        rows.append(part)

    pred_df = pd.concat(rows, ignore_index=True).sort_values(["product_family_name", "date"]).reset_index(drop=True)
    pred_df["is_sale"] = (pred_df["y"] > 0).astype(int)
    return pred_df


# ----------------------------
# Zero-inflated count models
# ----------------------------
def _clip_count_target(y: np.ndarray, q: float = 0.98, max_cap: int = 200) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)
    y_int = np.rint(y).astype(int)
    if y_int.size == 0:
        return y_int
    cap = int(np.quantile(y_int, q)) if np.any(y_int > 0) else 0
    if max_cap > 0:
        cap = min(cap, int(max_cap)) if cap > 0 else int(max_cap)
    if cap > 0:
        y_int = np.minimum(y_int, cap)
    return y_int


def _is_binary_or_cyclic_feature(col: str) -> bool:
    return (
        col.startswith("lag_is_sale_")
        or col.startswith("sale_rate_")
        or col.endswith("_sin")
        or col.endswith("_cos")
        or col in {"is_weekend", "is_q4", "dow", "month", "weekofyear"}
    )


def _stabilize_feature_frame(
    df: pd.DataFrame,
    cols: Sequence[str],
    center: Optional[pd.Series] = None,
    scale: Optional[pd.Series] = None,
    clip_value: float = 6.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = df[list(cols)].astype(float).copy()
    for col in cols:
        vals = x[col].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if not _is_binary_or_cyclic_feature(col):
            vals = np.log1p(np.clip(vals, 0.0, None))
        x[col] = vals

    if center is None:
        center = x.mean(axis=0)
    if scale is None:
        scale = x.std(axis=0, ddof=0)
        scale = scale.replace(0.0, 1.0).fillna(1.0)

    x = (x - center) / scale
    x = x.clip(-clip_value, clip_value)
    x = x.fillna(0.0)
    return x, center, scale


def _prepare_design_matrices(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    inflation_feature_cols: Sequence[str],
    feature_center: Optional[pd.Series] = None,
    feature_scale: Optional[pd.Series] = None,
    inflation_center: Optional[pd.Series] = None,
    inflation_scale: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    x_raw, feature_center, feature_scale = _stabilize_feature_frame(
        df, feature_cols, center=feature_center, scale=feature_scale
    )
    x_infl_raw, inflation_center, inflation_scale = _stabilize_feature_frame(
        df, inflation_feature_cols, center=inflation_center, scale=inflation_scale
    )
    x = add_constant(x_raw, prepend=True, has_constant="add")
    x_infl = add_constant(x_infl_raw, prepend=True, has_constant="add")
    design_info = {
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "inflation_center": inflation_center,
        "inflation_scale": inflation_scale,
    }
    return x, x_infl, design_info


def _fit_zero_inflated_count(
    train_feat: pd.DataFrame,
    distribution: str,
    feature_cols: Sequence[str] = COUNT_FEATURE_COLS,
    inflation_feature_cols: Sequence[str] = INFLATION_FEATURE_COLS,
    maxiter: int = 200,
) -> Dict[str, Any]:
    y = _clip_count_target(train_feat["y"].to_numpy(dtype=float))
    x, x_infl, design_info = _prepare_design_matrices(train_feat, feature_cols, inflation_feature_cols)

    if distribution == "zip":
        model = ZeroInflatedPoisson(endog=y, exog=x, exog_infl=x_infl, inflation="logit")
    elif distribution == "zinb":
        model = ZeroInflatedNegativeBinomialP(endog=y, exog=x, exog_infl=x_infl, inflation="logit")
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", HessianInversionWarning)
            result = model.fit(method="bfgs", maxiter=maxiter, disp=0)
        params = np.asarray(result.params, dtype=float)
        if not np.all(np.isfinite(params)):
            raise ValueError("non-finite fitted parameters")
        return {
            "distribution": distribution,
            "result": result,
            "feature_cols": list(feature_cols),
            "inflation_feature_cols": list(inflation_feature_cols),
            "status": "ok",
            **design_info,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        warnings.warn(f"{distribution.upper()} fit failed; using fallback. Error: {exc}")
        return {
            "distribution": distribution,
            "result": None,
            "feature_cols": list(feature_cols),
            "inflation_feature_cols": list(inflation_feature_cols),
            "status": f"failed: {exc}",
            **design_info,
        }


def _history_feature_row(current_date: pd.Timestamp, sales_hist: Sequence[float], occ_hist: Sequence[int]) -> Dict[str, float]:
    sales = np.asarray(sales_hist, dtype=float)
    occ = np.asarray(occ_hist, dtype=float)

    def _lag(arr: np.ndarray, k: int) -> float:
        return float(arr[-k]) if arr.size >= k else 0.0

    def _tail_mean(arr: np.ndarray, w: int) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr[-min(w, arr.size) :]))

    def _days_since_last_sale(occ_arr: np.ndarray) -> int:
        if occ_arr.size == 0 or np.all(occ_arr <= 0):
            return 999
        idx = np.where(occ_arr > 0)[0]
        return int((occ_arr.size - 1) - idx[-1])

    dow = int(current_date.dayofweek)
    week = int(current_date.isocalendar().week)
    month = int(current_date.month)
    quarter = int(current_date.quarter)
    days_since = _days_since_last_sale(occ)

    row = {
        "lag_1": _lag(sales, 1),
        "lag_7": _lag(sales, 7),
        "lag_14": _lag(sales, 14),
        "lag_28": _lag(sales, 28),
        "lag_is_sale_1": _lag(occ, 1),
        "lag_is_sale_7": _lag(occ, 7),
        "lag_is_sale_14": _lag(occ, 14),
        "sale_rate_7": _tail_mean(occ, 7),
        "sale_rate_28": _tail_mean(occ, 28),
        "roll_mean_7": _tail_mean(sales, 7),
        "roll_mean_28": _tail_mean(sales, 28),
        "days_since_last_sale": float(days_since),
        "days_since_last_sale_clip56": float(min(days_since, 56)),
        "dow": dow,
        "month": month,
        "weekofyear": week,
        "is_weekend": int(dow in (5, 6)),
        "is_q4": int(quarter == 4),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "week_sin": float(np.sin(2.0 * np.pi * week / 52.0)),
        "week_cos": float(np.cos(2.0 * np.pi * week / 52.0)),
    }
    return row


def _predict_count_model(
    fit_obj: Mapping[str, Any],
    feat_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    if fit_obj.get("result") is None:
        mean_pred = np.zeros(len(feat_df), dtype=float)
        p_nonzero = np.zeros(len(feat_df), dtype=float)
        return mean_pred, p_nonzero

    x, x_infl, _ = _prepare_design_matrices(
        feat_df,
        fit_obj["feature_cols"],
        fit_obj["inflation_feature_cols"],
        feature_center=fit_obj.get("feature_center"),
        feature_scale=fit_obj.get("feature_scale"),
        inflation_center=fit_obj.get("inflation_center"),
        inflation_scale=fit_obj.get("inflation_scale"),
    )
    result = fit_obj["result"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_pred = np.asarray(result.predict(exog=x, exog_infl=x_infl, which="mean"), dtype=float)
        prob_zero = np.asarray(result.predict(exog=x, exog_infl=x_infl, which="prob-zero"), dtype=float)
    mean_pred = np.nan_to_num(mean_pred, nan=0.0, posinf=0.0, neginf=0.0)
    prob_zero = np.nan_to_num(prob_zero, nan=1.0, posinf=1.0, neginf=1.0)
    p_nonzero = 1.0 - prob_zero
    mean_pred = np.clip(mean_pred, 0.0, None)
    p_nonzero = np.clip(p_nonzero, 0.0, 1.0)
    return mean_pred, p_nonzero


def _recursive_count_forecast(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    fit_obj: Mapping[str, Any],
    output_col: str,
    p_col: str,
    nonzero_threshold: float = 0.50,
) -> pd.DataFrame:
    cluster_map = (
        train_panel[["product_family_name", "cluster"]]
        .drop_duplicates()
        .set_index("product_family_name")["cluster"]
        .to_dict()
    )
    sku_list = sorted(cluster_map.keys())
    history_sales: Dict[str, List[float]] = {
        sku: train_panel.loc[train_panel["product_family_name"] == sku, "total_sales"].astype(float).tolist()
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
            row["product_family_name"] = sku
            row["cluster"] = cluster_map[sku]
            feat_rows.append(row)
        feat_df = pd.DataFrame(feat_rows)
        mean_pred, p_nonzero = _predict_count_model(fit_obj, feat_df)
        pred = np.where(p_nonzero >= nonzero_threshold, mean_pred, 0.0)
        pred = np.clip(pred, 0.0, None)

        feat_df[p_col] = p_nonzero
        feat_df[output_col] = pred
        rows.append(feat_df[["date", "product_family_name", "cluster", p_col, output_col]])

        for sku, pred_value, p_value in zip(feat_df["product_family_name"], pred.tolist(), p_nonzero.tolist()):
            history_sales[sku].append(float(pred_value))
            history_occ[sku].append(int(p_value >= nonzero_threshold and pred_value > 0))

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df = pred_df.merge(
        test_panel[["date", "product_family_name", "cluster", "total_sales"]],
        on=["date", "product_family_name", "cluster"],
        how="left",
    )
    pred_df["y"] = pred_df["total_sales"].astype(float)
    pred_df["is_sale"] = (pred_df["y"] > 0).astype(int)
    return pred_df.drop(columns=["total_sales"])


# ----------------------------
# Metrics
# ----------------------------
def _ape_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps) * 100.0


def _wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_pred - y_true)) / max(np.sum(np.abs(y_true)), eps) * 100.0)


def _build_metric_table(pred_df: pd.DataFrame, model_cols: Mapping[str, str], eps: float = 1.0) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = pred_df["y"].to_numpy(dtype=float)
    actual_nonzero_rate = float(np.mean(y > 0))
    for method_name, col in model_cols.items():
        pred = pred_df[col].to_numpy(dtype=float)
        ape = _ape_eps(y, pred, eps=eps)
        signed_pct = (pred - y) / np.maximum(np.abs(y), eps) * 100.0
        rows.append(
            {
                "method": method_name,
                "wmape_pct": _wmape(y, pred, eps=eps),
                "epsilon_mape_pct": float(np.mean(ape)),
                "cap_mape_0_100": float(np.mean(np.clip(ape, 0.0, 100.0))),
                "tail_ape_p90": float(np.quantile(ape, 0.90)),
                "tail_ape_p95": float(np.quantile(ape, 0.95)),
                "pred_nonzero_rate": float(np.mean(pred > 0)),
                "actual_nonzero_rate": actual_nonzero_rate,
                "nonzero_rate_gap": float(np.mean(pred > 0) - actual_nonzero_rate),
                "signed_pct_error_mean": float(np.mean(signed_pct)),
                "signed_abs_bias_ratio": float(np.sum(pred - y) / max(np.sum(np.abs(y)), eps)),
            }
        )
    return pd.DataFrame(rows).sort_values(["wmape_pct", "epsilon_mape_pct"]).reset_index(drop=True)


# ----------------------------
# Main pipeline
# ----------------------------
def run_c1_forecasting_new(
    train_path: str | Path = TRAIN_DEFAULT,
    test_path: str | Path = TEST_DEFAULT,
    cluster_id: int = 1,
    zip_threshold: float = 0.50,
    zinb_threshold: float = 0.50,
    save_predictions: bool = False,
    prediction_output_path: str | Path = PREDICTION_OUTPUT_DEFAULT,
    metrics_output_path: Optional[str | Path] = METRICS_OUTPUT_DEFAULT,
    eps_mape: float = 1.0,
) -> C1NewArtifacts:
    train_daily, test_daily = _load_daily(train_path, test_path)
    train_raw = train_daily[train_daily["cluster"] == cluster_id].copy()
    test_raw = test_daily[test_daily["cluster"] == cluster_id].copy()
    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_features = _build_train_features(train_panel)

    pred_df = _build_univariate_baselines(train_panel, test_panel)

    zip_fit = _fit_zero_inflated_count(train_features, distribution="zip")
    zinb_fit = _fit_zero_inflated_count(train_features, distribution="zinb")

    zip_pred = _recursive_count_forecast(
        train_panel=train_panel,
        test_panel=test_panel,
        fit_obj=zip_fit,
        output_col="pred_zip",
        p_col="p_nonzero_zip",
        nonzero_threshold=zip_threshold,
    )
    zinb_pred = _recursive_count_forecast(
        train_panel=train_panel,
        test_panel=test_panel,
        fit_obj=zinb_fit,
        output_col="pred_zinb",
        p_col="p_nonzero_zinb",
        nonzero_threshold=zinb_threshold,
    )

    pred_df = pred_df.merge(
        zip_pred[["date", "product_family_name", "cluster", "pred_zip", "p_nonzero_zip"]],
        on=["date", "product_family_name", "cluster"],
        how="left",
    )
    pred_df = pred_df.merge(
        zinb_pred[["date", "product_family_name", "cluster", "pred_zinb", "p_nonzero_zinb"]],
        on=["date", "product_family_name", "cluster"],
        how="left",
    )
    pred_df["p_sale"] = pred_df[["p_nonzero_zip", "p_nonzero_zinb"]].max(axis=1).fillna(0.0)

    model_map = {
        "naive7": "pred_naive7",
        "sma28": "pred_sma28",
        "tsb": "pred_tsb",
        "adida": "pred_adida",
        "imapa": "pred_imapa",
        "zip": "pred_zip",
        "zinb": "pred_zinb",
    }
    metrics_table = _build_metric_table(pred_df, model_map, eps=eps_mape)

    prediction_output_path = str(prediction_output_path)
    metrics_output_path = str(metrics_output_path) if metrics_output_path is not None else None
    if save_predictions:
        prediction_output_path = _save_table(pred_df, prediction_output_path)
        if metrics_output_path is not None:
            metrics_output_path = _save_table(metrics_table, metrics_output_path)

    metadata = {
        "cluster_id": cluster_id,
        "zip_status": zip_fit.get("status"),
        "zinb_status": zinb_fit.get("status"),
        "zip_threshold": zip_threshold,
        "zinb_threshold": zinb_threshold,
        "count_feature_cols": COUNT_FEATURE_COLS,
        "inflation_feature_cols": INFLATION_FEATURE_COLS,
    }

    return C1NewArtifacts(
        cluster_id=cluster_id,
        train_raw=train_raw,
        test_raw=test_raw,
        train_panel=train_panel,
        test_panel=test_panel,
        train_features=train_features,
        pred_df=pred_df,
        metrics_table=metrics_table,
        model_cols=MODEL_COLS_ORDER.copy(),
        metadata=metadata,
        prediction_output_path=prediction_output_path if save_predictions else None,
        metrics_output_path=metrics_output_path if save_predictions else None,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leak-safe C1 model comparison pipeline")
    parser.add_argument("--train-path", default=TRAIN_DEFAULT)
    parser.add_argument("--test-path", default=TEST_DEFAULT)
    parser.add_argument("--cluster-id", type=int, default=1)
    parser.add_argument("--zip-threshold", type=float, default=0.50)
    parser.add_argument("--zinb-threshold", type=float, default=0.50)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-output-path", default=PREDICTION_OUTPUT_DEFAULT)
    parser.add_argument("--metrics-output-path", default=METRICS_OUTPUT_DEFAULT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    art = run_c1_forecasting_new(
        train_path=args.train_path,
        test_path=args.test_path,
        cluster_id=args.cluster_id,
        zip_threshold=args.zip_threshold,
        zinb_threshold=args.zinb_threshold,
        save_predictions=args.save_predictions,
        prediction_output_path=args.prediction_output_path,
        metrics_output_path=args.metrics_output_path,
    )
    print(json.dumps({
        "cluster_id": art.cluster_id,
        "models": art.model_cols,
        "metrics_head": art.metrics_table.head(10).to_dict(orient="records"),
        "metadata": art.metadata,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
