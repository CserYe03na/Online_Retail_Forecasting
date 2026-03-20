from __future__ import annotations

"""C1 forecasting pipeline with naive7 baseline vs leak-safe ZINB.

This script follows the reporting structure of ``c1_forecasting.py`` while
keeping only the C1 comparison requested by the project:

- baseline: ``pred_naive7``
- model: ``pred_zinb``

The train/test split is taken strictly from the provided parquet files. Final
reporting still uses only that split. To tune a small number of ZINB
parameters, the script uses a short time-ordered holdout cut from the tail of
the training panel only. ZINB test-period features are built recursively from
historical train data plus prior predictions only, so test targets are never
used during forecasting.
"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.tools.sm_exceptions import HessianInversionWarning

from c1_forecasting import compute_metric_bundle, pointwise_safe_ape
from c1_forecasting_new import (
    COUNT_FEATURE_COLS,
    INFLATION_FEATURE_COLS,
    _build_train_features,
    _history_feature_row,
    _load_daily,
    _predict_count_model,
    _prepare_design_matrices,
    _save_table,
    _build_zero_filled_panel,
)


TRAIN_DEFAULT = "data/forecasting/train_daily.parquet"
TEST_DEFAULT = "data/forecasting/test_daily.parquet"
PREDICTION_OUTPUT_DEFAULT = "forecasting/c1_prediction.parquet"
SEARCH_HOLDOUT_DAYS_DEFAULT = 42
DEFAULT_ZINB_THRESHOLD = 0.50
DEFAULT_ZINB_CLIP_Q = 0.98
DEFAULT_ZINB_CLIP_MAX_CAP = 200
DEFAULT_TUNING_OBJECTIVE = "wmape"


@dataclass
class C1Forecasting2Artifacts:
    cluster_id: int
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    prediction_output_path: Optional[str] = None
    metrics_output_path: Optional[str] = None


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


def _build_naive7_predictions(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for sku, tr_sku in train_panel.groupby("product_family_name"):
        te_sku = test_panel[test_panel["product_family_name"] == sku].sort_values("date")
        horizon = len(te_sku)
        train_series = tr_sku.sort_values("date")["total_sales"].to_numpy(dtype=float)

        part = te_sku[["date", "product_family_name", "cluster", "total_sales"]].copy()
        part["y"] = part["total_sales"].astype(float)
        part["is_sale"] = (part["y"] > 0).astype(int)
        part["pred_naive7"] = _forecast_naive7(train_series, horizon)
        rows.append(part.drop(columns=["total_sales"]))

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )


def _clip_count_target(
    y: np.ndarray,
    q: float = DEFAULT_ZINB_CLIP_Q,
    max_cap: int = DEFAULT_ZINB_CLIP_MAX_CAP,
) -> np.ndarray:
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


def _fit_zinb_with_config(
    train_feat: pd.DataFrame,
    clip_q: float,
    clip_max_cap: int,
    maxiter: int = 200,
    feature_cols: Sequence[str] = COUNT_FEATURE_COLS,
    inflation_feature_cols: Sequence[str] = INFLATION_FEATURE_COLS,
) -> Dict[str, Any]:
    y = _clip_count_target(train_feat["y"].to_numpy(dtype=float), q=clip_q, max_cap=clip_max_cap)
    x, x_infl, design_info = _prepare_design_matrices(train_feat, feature_cols, inflation_feature_cols)
    model = ZeroInflatedNegativeBinomialP(endog=y, exog=x, exog_infl=x_infl, inflation="logit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", HessianInversionWarning)
            result = model.fit(method="bfgs", maxiter=maxiter, disp=0)
        params = np.asarray(result.params, dtype=float)
        if not np.all(np.isfinite(params)):
            raise ValueError("non-finite fitted parameters")
        return {
            "distribution": "zinb",
            "result": result,
            "feature_cols": list(feature_cols),
            "inflation_feature_cols": list(inflation_feature_cols),
            "status": "ok",
            "clip_q": float(clip_q),
            "clip_max_cap": int(clip_max_cap),
            "maxiter": int(maxiter),
            **design_info,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        warnings.warn(f"ZINB fit failed; using fallback. Error: {exc}")
        return {
            "distribution": "zinb",
            "result": None,
            "feature_cols": list(feature_cols),
            "inflation_feature_cols": list(inflation_feature_cols),
            "status": f"failed: {exc}",
            "clip_q": float(clip_q),
            "clip_max_cap": int(clip_max_cap),
            "maxiter": int(maxiter),
            **design_info,
        }


def _recursive_zinb_forecast(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    fit_obj: Mapping[str, Any],
    nonzero_threshold: float,
    output_col: str = "pred_zinb",
    p_col: str = "p_nonzero_zinb",
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


def _split_train_holdout_panel(
    train_panel: pd.DataFrame,
    holdout_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(pd.to_datetime(train_panel["date"].dropna().unique())))
    if unique_dates.size < 56:
        raise ValueError("Not enough training dates for parameter search.")

    holdout_days = int(max(14, holdout_days))
    holdout_days = int(min(holdout_days, unique_dates.size // 3))
    cut_date = pd.Timestamp(unique_dates[-holdout_days])

    core_panel = train_panel[train_panel["date"] < cut_date].copy()
    holdout_panel = train_panel[train_panel["date"] >= cut_date].copy()
    if core_panel.empty or holdout_panel.empty:
        raise ValueError("Invalid train/holdout split generated for parameter search.")
    return core_panel, holdout_panel


def _positive_underprediction_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if int(np.sum(mask)) == 0:
        return 0.0
    under = np.maximum(y_true[mask] - y_pred[mask], 0.0)
    denom = np.maximum(np.abs(y_true[mask]), eps)
    return float(np.mean(under / denom) * 100.0)


def _candidate_caps(train_feat: pd.DataFrame) -> List[int]:
    positive = train_feat.loc[train_feat["y"] > 0, "y"].to_numpy(dtype=float)
    if positive.size == 0:
        return [DEFAULT_ZINB_CLIP_MAX_CAP]
    q995 = int(np.ceil(np.quantile(positive, 0.995)))
    q999 = int(np.ceil(np.quantile(positive, 0.999)))
    cap_base = int(np.clip(q995, DEFAULT_ZINB_CLIP_MAX_CAP, 300))
    cap_loose = int(np.clip(max(q999, cap_base), DEFAULT_ZINB_CLIP_MAX_CAP, 400))
    return sorted(set([DEFAULT_ZINB_CLIP_MAX_CAP, cap_base, cap_loose]))


def _search_zinb_params(
    train_panel: pd.DataFrame,
    eps_mape: float,
    metric_name: str,
    holdout_days: int = SEARCH_HOLDOUT_DAYS_DEFAULT,
    tuning_objective: str = DEFAULT_TUNING_OBJECTIVE,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    core_panel, holdout_panel = _split_train_holdout_panel(train_panel, holdout_days=holdout_days)
    core_feat = _build_train_features(core_panel)
    actual_holdout_days = int(len(sorted(pd.to_datetime(holdout_panel["date"].dropna().unique()))))

    threshold_grid = [0.40, 0.50]
    clip_q_grid = [DEFAULT_ZINB_CLIP_Q, 0.99]
    cap_grid = _candidate_caps(core_feat)

    early_dates = np.array(sorted(pd.to_datetime(holdout_panel["date"].dropna().unique())))
    early_dates = set(pd.to_datetime(early_dates[: min(14, len(early_dates))]))

    trial_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None

    for threshold, clip_q, clip_max_cap in product(threshold_grid, clip_q_grid, cap_grid):
        fit_obj = _fit_zinb_with_config(
            train_feat=core_feat,
            clip_q=clip_q,
            clip_max_cap=clip_max_cap,
            maxiter=200,
        )
        pred_df = _recursive_zinb_forecast(
            train_panel=core_panel,
            test_panel=holdout_panel,
            fit_obj=fit_obj,
            nonzero_threshold=threshold,
        )

        overall_bundle = compute_metric_bundle(
            y_true=pred_df["y"].to_numpy(dtype=float),
            y_pred=pred_df["pred_zinb"].to_numpy(dtype=float),
            y_true_sale=pred_df["is_sale"].to_numpy(dtype=int),
            metric_name=metric_name,
            eps=eps_mape,
        )

        early_mask = pred_df["date"].isin(early_dates)
        early_bundle = compute_metric_bundle(
            y_true=pred_df.loc[early_mask, "y"].to_numpy(dtype=float),
            y_pred=pred_df.loc[early_mask, "pred_zinb"].to_numpy(dtype=float),
            y_true_sale=pred_df.loc[early_mask, "is_sale"].to_numpy(dtype=int),
            metric_name=metric_name,
            eps=eps_mape,
        )
        early_under = _positive_underprediction_pct(
            y_true=pred_df.loc[early_mask, "y"].to_numpy(dtype=float),
            y_pred=pred_df.loc[early_mask, "pred_zinb"].to_numpy(dtype=float),
            eps=eps_mape,
        )

        if tuning_objective == "wmape":
            overall_primary = float(overall_bundle["WMAPE_0_100"])
            early_primary = float(early_bundle["WMAPE_0_100"])
        elif tuning_objective == "mape":
            overall_primary = float(overall_bundle["MAPE_0_100"])
            early_primary = float(early_bundle["MAPE_0_100"])
        else:
            raise ValueError("tuning_objective must be one of: wmape, mape")

        score = 0.60 * overall_primary + 0.25 * early_primary + 0.15 * float(early_under)
        if fit_obj.get("status") != "ok":
            score += 25.0
        row = {
            "score": float(score),
            "tuning_objective": tuning_objective,
            "threshold": float(threshold),
            "clip_q": float(clip_q),
            "clip_max_cap": int(clip_max_cap),
            "fit_status": fit_obj.get("status"),
            "overall_mape_0_100": float(overall_bundle["MAPE_0_100"]),
            "overall_wmape_0_100": float(overall_bundle["WMAPE_0_100"]),
            "early_mape_0_100": float(early_bundle["MAPE_0_100"]),
            "early_wmape_0_100": float(early_bundle["WMAPE_0_100"]),
            "early_underprediction_pct": float(early_under),
            "pred_nonzero_rate": float(np.mean(pred_df["pred_zinb"].to_numpy(dtype=float) > 0)),
            "actual_nonzero_rate": float(np.mean(pred_df["y"].to_numpy(dtype=float) > 0)),
        }
        trial_rows.append(row)
        if best_row is None or (
            row["score"],
            row["early_wmape_0_100"] if tuning_objective == "wmape" else row["early_mape_0_100"],
            row["overall_wmape_0_100"] if tuning_objective == "wmape" else row["overall_mape_0_100"],
        ) < (
            best_row["score"],
            best_row["early_wmape_0_100"] if tuning_objective == "wmape" else best_row["early_mape_0_100"],
            best_row["overall_wmape_0_100"] if tuning_objective == "wmape" else best_row["overall_mape_0_100"],
        ):
            best_row = row

    if best_row is None:
        raise RuntimeError("ZINB parameter search produced no valid candidates.")

    trials_df = pd.DataFrame(trial_rows).sort_values(
        ["score", "early_wmape_0_100", "overall_wmape_0_100"]
        if tuning_objective == "wmape"
        else ["score", "early_mape_0_100", "overall_mape_0_100"]
    ).reset_index(drop=True)
    best_cfg = {
        "zinb_threshold": float(best_row["threshold"]),
        "clip_q": float(best_row["clip_q"]),
        "clip_max_cap": int(best_row["clip_max_cap"]),
        "search_holdout_days": actual_holdout_days,
        "maxiter": 200,
        "score": float(best_row["score"]),
        "fit_status": best_row["fit_status"],
        "tuning_objective": tuning_objective,
    }
    return best_cfg, trials_df


def _build_days_since_features(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(
        [train_panel.assign(split="train"), test_panel.assign(split="test")],
        ignore_index=True,
    ).sort_values(["product_family_name", "date"]).reset_index(drop=True)

    all_df["y"] = all_df["total_sales"].astype(float)
    all_df["is_sale"] = (all_df["y"] > 0).astype(int)

    last_sale_date = all_df["date"].where(all_df["is_sale"] == 1)
    last_sale_date = last_sale_date.groupby(all_df["product_family_name"]).ffill()
    last_sale_date = last_sale_date.groupby(all_df["product_family_name"]).shift(1)
    all_df["days_since_last_sale"] = (
        (all_df["date"] - last_sale_date).dt.days.fillna(999).astype(int)
    )

    keep_cols = [
        "date",
        "product_family_name",
        "cluster",
        "y",
        "is_sale",
        "days_since_last_sale",
    ]
    train_feat = all_df.loc[all_df["split"] == "train", keep_cols].reset_index(drop=True)
    test_feat = all_df.loc[all_df["split"] == "test", keep_cols].reset_index(drop=True)
    return train_feat, test_feat


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
        return pd.DataFrame(
            columns=["method", "period", "count", "q50", "q75", "q90", "q95", "q99", "mean"]
        )

    rows: List[Dict[str, Any]] = []
    for (method, period), grp in ape_box_df.groupby(["method", "period"]):
        arr = grp["APE_0_100"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "period": period,
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


def _build_reporting_tables(
    pred_df: pd.DataFrame,
    eps_mape: float,
    metric_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_map = [("naive7", "pred_naive7"), ("zinb", "pred_zinb")]

    overall_rows: List[Dict[str, Any]] = []
    for method, col in method_map:
        metric_bundle = compute_metric_bundle(
            y_true=pred_df["y"].to_numpy(dtype=float),
            y_pred=pred_df[col].to_numpy(dtype=float),
            y_true_sale=pred_df["is_sale"].to_numpy(dtype=int),
            metric_name=metric_name,
            eps=eps_mape,
        )
        overall_rows.append({"method": method, **metric_bundle})
    metrics_overall = pd.DataFrame(overall_rows).sort_values("MAPE_0_100").reset_index(drop=True)

    period_rows: List[Dict[str, Any]] = []
    ape_rows: List[pd.DataFrame] = []
    for method, col in method_map:
        ape_eps = pointwise_safe_ape(
            pred_df["y"].to_numpy(dtype=float),
            pred_df[col].to_numpy(dtype=float),
            eps=eps_mape,
        )
        ape_cap = np.clip(ape_eps, 0.0, 100.0)
        box_part = pred_df[["date", "period", "product_family_name"]].copy()
        box_part["method"] = method
        box_part["APE_EPS_PCT"] = ape_eps
        box_part["APE_CAP_0_100"] = ape_cap
        box_part["APE_0_100"] = ape_cap
        ape_rows.append(box_part)

        for period, grp in pred_df.groupby("period"):
            metric_bundle = compute_metric_bundle(
                y_true=grp["y"].to_numpy(dtype=float),
                y_pred=grp[col].to_numpy(dtype=float),
                y_true_sale=grp["is_sale"].to_numpy(dtype=int),
                metric_name=metric_name,
                eps=eps_mape,
            )
            period_rows.append({"method": method, "period": period, **metric_bundle})

    ape_box_df = pd.concat(ape_rows, ignore_index=True)
    metrics_by_period = (
        pd.DataFrame(period_rows).sort_values(["method", "period"]).reset_index(drop=True)
    )

    ape_box_df_trimmed = ape_box_df.copy()
    if not ape_box_df_trimmed.empty:
        keep_idx: List[int] = []
        for (_, _), grp in ape_box_df_trimmed.groupby(["method", "period"]):
            cap = float(np.quantile(grp["APE_0_100"].to_numpy(dtype=float), 0.99))
            keep_idx.extend(grp.index[grp["APE_0_100"] <= cap].tolist())
        ape_box_df_trimmed = ape_box_df_trimmed.loc[sorted(set(keep_idx))].copy()

    error_quantiles = _build_error_quantiles(ape_box_df)
    return metrics_overall, metrics_by_period, ape_box_df, ape_box_df_trimmed, error_quantiles


def run_c1_forecasting_2(
    train_path: str | Path = TRAIN_DEFAULT,
    test_path: str | Path = TEST_DEFAULT,
    cluster_id: int = 1,
    n_periods: int = 4,
    eps_mape: float = 1.0,
    metric_name: str = "bounded_mape",
    zinb_threshold: Optional[float] = None,
    tune: bool = False,
    tuning_objective: str = DEFAULT_TUNING_OBJECTIVE,
    search_holdout_days: int = SEARCH_HOLDOUT_DAYS_DEFAULT,
    search_enabled: Optional[bool] = None,
    save_predictions: bool = True,
    prediction_output_path: str | Path = PREDICTION_OUTPUT_DEFAULT,
    metrics_output_path: Optional[str | Path] = None,
) -> C1Forecasting2Artifacts:
    train_daily, test_daily = _load_daily(train_path, test_path)
    train_raw = train_daily[train_daily["cluster"] == cluster_id].copy()
    test_raw = test_daily[test_daily["cluster"] == cluster_id].copy()

    if train_raw.empty or test_raw.empty:
        raise ValueError(f"No rows found for cluster={cluster_id} in train/test.")

    train_panel, test_panel = _build_zero_filled_panel(train_raw, test_raw)
    train_model_feat = _build_train_features(train_panel)
    train_feat, test_feat = _build_days_since_features(train_panel, test_panel)

    pred_df = _build_naive7_predictions(train_panel, test_panel)

    do_tune = bool(tune if search_enabled is None else search_enabled)

    tuning_trials: Optional[pd.DataFrame] = None
    tuning_best_config: Optional[Dict[str, Any]] = None
    if do_tune:
        tuning_best_config, tuning_trials = _search_zinb_params(
            train_panel=train_panel,
            eps_mape=eps_mape,
            metric_name=metric_name,
            holdout_days=search_holdout_days,
            tuning_objective=tuning_objective,
        )
    else:
        tuning_best_config = {
            "zinb_threshold": DEFAULT_ZINB_THRESHOLD,
            "clip_q": DEFAULT_ZINB_CLIP_Q,
            "clip_max_cap": DEFAULT_ZINB_CLIP_MAX_CAP,
            "search_holdout_days": 0,
            "maxiter": 200,
            "score": float("nan"),
            "fit_status": "fixed_optimal_params_tuning_disabled",
            "tuning_objective": tuning_objective,
        }

    chosen_threshold = float(
        tuning_best_config["zinb_threshold"] if zinb_threshold is None else zinb_threshold
    )
    zinb_fit = _fit_zinb_with_config(
        train_feat=train_model_feat,
        clip_q=float(tuning_best_config["clip_q"]),
        clip_max_cap=int(tuning_best_config["clip_max_cap"]),
        maxiter=int(tuning_best_config["maxiter"]),
    )
    zinb_pred = _recursive_zinb_forecast(
        train_panel=train_panel,
        test_panel=test_panel,
        fit_obj=zinb_fit,
        nonzero_threshold=chosen_threshold,
    )

    pred_df = pred_df.merge(
        zinb_pred[["date", "product_family_name", "cluster", "pred_zinb", "p_nonzero_zinb"]],
        on=["date", "product_family_name", "cluster"],
        how="left",
    )
    pred_df["pred_zinb"] = pred_df["pred_zinb"].fillna(0.0)
    pred_df["p_nonzero_zinb"] = pred_df["p_nonzero_zinb"].fillna(0.0)
    pred_df["p_sale"] = pred_df["p_nonzero_zinb"]
    pred_df = _periodize_test(pred_df, n_periods=n_periods)

    (
        metrics_overall,
        metrics_by_period,
        ape_box_df,
        ape_box_df_trimmed,
        error_quantiles,
    ) = _build_reporting_tables(pred_df=pred_df, eps_mape=eps_mape, metric_name=metric_name)

    prediction_output_path_str: Optional[str] = None
    if save_predictions:
        prediction_output_path_str = _save_table(pred_df, prediction_output_path)

    metadata = {
        "cluster_id": cluster_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int(len(train_raw)),
        "test_rows": int(len(test_raw)),
        "n_periods": int(n_periods),
        "metric_name": metric_name,
        "eps_mape": float(eps_mape),
        "zinb_threshold": float(chosen_threshold),
        "zinb_status": zinb_fit.get("status"),
        "tune": bool(do_tune),
        "tuning_objective": tuning_objective,
        "search_holdout_days": int(tuning_best_config["search_holdout_days"]),
        "clip_q": float(tuning_best_config["clip_q"]),
        "clip_max_cap": int(tuning_best_config["clip_max_cap"]),
        "split_policy": (
            "strict parquet train/test for final reporting; parameter search uses train-tail holdout only"
            if do_tune
            else "strict parquet train/test only; tune disabled and default ZINB parameters used"
        ),
        "baseline_col": "pred_naive7",
        "model_col": "pred_zinb",
    }

    return C1Forecasting2Artifacts(
        cluster_id=cluster_id,
        train_raw=train_raw,
        test_raw=test_raw,
        train_panel=train_panel,
        test_panel=test_panel,
        train_feat=train_feat,
        test_feat=test_feat,
        pred_df=pred_df,
        metrics_overall=metrics_overall,
        metrics_by_period=metrics_by_period,
        ape_box_df=ape_box_df,
        ape_box_df_positive=None,
        ape_box_df_trimmed=ape_box_df_trimmed,
        error_quantiles=error_quantiles,
        tuning_trials=tuning_trials,
        tuning_best_config=tuning_best_config,
        metadata=metadata,
        prediction_output_path=prediction_output_path_str,
        metrics_output_path=None,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C1 forecasting: naive7 vs leak-safe ZINB")
    parser.add_argument("--train-path", default=TRAIN_DEFAULT)
    parser.add_argument("--test-path", default=TEST_DEFAULT)
    parser.add_argument("--cluster-id", type=int, default=1)
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--eps-mape", type=float, default=1.0)
    parser.add_argument("--metric-name", default="bounded_mape", choices=["bounded_mape", "safe_mape"])
    parser.add_argument("--zinb-threshold", type=float, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tuning-objective", default=DEFAULT_TUNING_OBJECTIVE, choices=["wmape", "mape"])
    parser.add_argument("--search-holdout-days", type=int, default=SEARCH_HOLDOUT_DAYS_DEFAULT)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-output-path", default=PREDICTION_OUTPUT_DEFAULT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    art = run_c1_forecasting_2(
        train_path=args.train_path,
        test_path=args.test_path,
        cluster_id=args.cluster_id,
        n_periods=args.n_periods,
        eps_mape=args.eps_mape,
        metric_name=args.metric_name,
        zinb_threshold=args.zinb_threshold,
        tune=args.tune,
        tuning_objective=args.tuning_objective,
        search_holdout_days=args.search_holdout_days,
        save_predictions=args.save_predictions,
        prediction_output_path=args.prediction_output_path,
    )
    print(
        json.dumps(
            {
                "cluster_id": art.cluster_id,
                "metrics_overall_head": art.metrics_overall.head(10).to_dict(orient="records"),
                "metrics_by_period_head": art.metrics_by_period.head(10).to_dict(orient="records"),
                "tuning_best_config": art.tuning_best_config,
                "tuning_trials_head": None if art.tuning_trials is None else art.tuning_trials.head(10).to_dict(orient="records"),
                "metadata": art.metadata,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
