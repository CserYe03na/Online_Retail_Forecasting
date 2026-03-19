from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return eps


def _prepare_metric_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)

    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    valid = np.isfinite(yt) & np.isfinite(yp)
    return yt, yp, valid


def pointwise_safe_ape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0
) -> np.ndarray:
    eps = _validate_eps(eps)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), eps)
    return np.abs(yt - yp) / denom * 100.0


def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0
) -> float:
    eps = _validate_eps(eps)
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if valid.sum() == 0:
        return float("nan")
    ape = pointwise_safe_ape(yt[valid], yp[valid], eps=eps)
    return float(np.mean(ape))


def bounded_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0,
    ape_cap: float = 100.0
) -> float:
    eps = _validate_eps(eps)
    ape_cap = float(ape_cap)
    if ape_cap <= 0:
        raise ValueError("ape_cap must be > 0")

    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if valid.sum() == 0:
        return float("nan")

    ape = pointwise_safe_ape(yt[valid], yp[valid], eps=eps)
    return float(np.mean(np.clip(ape, 0.0, ape_cap)))


def wmape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0
) -> float:
    eps = _validate_eps(eps)
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if valid.sum() == 0:
        return float("nan")

    yt = yt[valid]
    yp = yp[valid]

    numerator = np.sum(np.abs(yt - yp))
    denominator = max(float(np.sum(np.abs(yt))), eps)
    return float(numerator / denominator * 100.0)


def positive_only_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12
) -> float:
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    yt = yt[valid]
    yp = yp[valid]

    mask = yt > 0
    if mask.sum() == 0:
        return float("nan")

    denom = np.maximum(yt[mask], eps)
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom) * 100.0)


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[valid] - yp[valid])))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt[valid] - yp[valid]) ** 2)))


def occurrence_f1(
    y_true_sale: np.ndarray,
    y_pred_sale: np.ndarray
) -> float:
    y_true_sale = np.asarray(y_true_sale).astype(int).reshape(-1)
    y_pred_sale = np.asarray(y_pred_sale).astype(int).reshape(-1)

    tp = int(np.sum((y_true_sale == 1) & (y_pred_sale == 1)))
    fp = int(np.sum((y_true_sale == 0) & (y_pred_sale == 1)))
    fn = int(np.sum((y_true_sale == 1) & (y_pred_sale == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def zero_day_fpr(
    y_true_sale: np.ndarray,
    y_pred_sale: np.ndarray
) -> float:
    y_true_sale = np.asarray(y_true_sale).astype(int).reshape(-1)
    y_pred_sale = np.asarray(y_pred_sale).astype(int).reshape(-1)

    zero_mask = y_true_sale == 0
    if zero_mask.sum() == 0:
        return float("nan")

    fp = np.sum((y_pred_sale == 1) & zero_mask)
    return float(fp / np.sum(zero_mask))


# --------------------------------------------------
# 1) 整体 pointwise 评估：把所有 sku-day 拉平后一起算
# --------------------------------------------------
def compute_overall_metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1.0,
    metric_name: str = "bounded_mape"
) -> Dict[str, float]:
    eps = _validate_eps(eps)

    yt, yp, valid = _prepare_metric_arrays(y_true, y_pred)
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

    y_true_sale = (yt > 0).astype(int)
    y_pred_sale = (yp > 0).astype(int)

    metric_bundle = {
        # c1/c3-aligned keys
        "MAPE_0_100": float(main_mape),
        "EPSILON_MAPE_PCT": float(epsilon_mape),
        "CAP_MAPE_0_100": float(cap_mape),
        "POSITIVE_ONLY_MAPE_PCT": float(positive_only_mape(yt, yp)),
        "WMAPE_0_100": float(wmape(yt, yp, eps=eps)),
        "OCCURRENCE_F1": float(occurrence_f1(y_true_sale, y_pred_sale)),
        "ZERO_DAY_FPR": float(zero_day_fpr(y_true_sale, y_pred_sale)),
        # keep existing names for backward compatibility
        "MAIN_MAPE_PCT": float(main_mape),
        "SAFE_MAPE_PCT": float(epsilon_mape),
        "BOUNDED_MAPE_PCT": float(cap_mape),
        "WMAPE_PCT": float(wmape(yt, yp, eps=eps)),
        "MAE": float(mae(yt, yp)),
        "RMSE": float(rmse(yt, yp)),
    }
    return metric_bundle


# --------------------------------------------------
# 2) 单个 sku 的 metric
# --------------------------------------------------
def compute_sku_metric_bundle(
    sku_actual: np.ndarray,
    sku_pred: np.ndarray,
    eps: float = 1.0,
    metric_name: str = "bounded_mape"
) -> Dict[str, float]:
    return compute_overall_metric_bundle(
        y_true=sku_actual,
        y_pred=sku_pred,
        eps=eps,
        metric_name=metric_name
    )


# --------------------------------------------------
# 3) 宽表（date x sku）的 sku-level 平均指标
#    每个 sku 先算一遍，再 across sku 取平均
# --------------------------------------------------
def compute_sku_level_metrics(
    sku_actual_df: pd.DataFrame,
    sku_pred_df: pd.DataFrame,
    eps: float = 1.0,
    metric_name: str = "bounded_mape"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    eps = _validate_eps(eps)

    common_cols = sorted(set(sku_actual_df.columns).intersection(set(sku_pred_df.columns)))
    if len(common_cols) == 0:
        raise ValueError("No common SKU columns found between sku_actual_df and sku_pred_df")

    rows = []

    for sku in common_cols:
        y_true = sku_actual_df[sku].to_numpy(dtype=float)
        y_pred = sku_pred_df[sku].to_numpy(dtype=float)

        metrics = compute_sku_metric_bundle(
            sku_actual=y_true,
            sku_pred=y_pred,
            eps=eps,
            metric_name=metric_name
        )
        metrics["sku"] = sku
        metrics["n_days"] = int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred)))
        metrics["nonzero_days"] = int(np.sum((np.isfinite(y_true)) & (y_true > 0)))
        metrics["actual_sum"] = float(np.nansum(y_true))
        metrics["pred_sum"] = float(np.nansum(y_pred))
        rows.append(metrics)

    sku_metrics_df = pd.DataFrame(rows)

    avg_metrics = {
        "SKU_MAIN_MAPE_MEAN_PCT": float(sku_metrics_df["MAIN_MAPE_PCT"].mean()),
        "SKU_SAFE_MAPE_MEAN_PCT": float(sku_metrics_df["SAFE_MAPE_PCT"].mean()),
        "SKU_BOUNDED_MAPE_MEAN_PCT": float(sku_metrics_df["BOUNDED_MAPE_PCT"].mean()),
        "SKU_POSITIVE_ONLY_MAPE_MEAN_PCT": float(sku_metrics_df["POSITIVE_ONLY_MAPE_PCT"].mean()),
        "SKU_WMAPE_MEAN_PCT": float(sku_metrics_df["WMAPE_PCT"].mean()),
        "SKU_MAE_MEAN": float(sku_metrics_df["MAE"].mean()),
        "SKU_RMSE_MEAN": float(sku_metrics_df["RMSE"].mean()),
        "SKU_OCCURRENCE_F1_MEAN": float(sku_metrics_df["OCCURRENCE_F1"].mean()),
        "SKU_ZERO_DAY_FPR_MEAN": float(sku_metrics_df["ZERO_DAY_FPR"].mean()),
    }

    return sku_metrics_df, avg_metrics


# --------------------------------------------------
# 4) 宽表（date x sku）的整体 metrics
#    所有 sku-day 一起 flatten 后算
# --------------------------------------------------
def compute_panel_overall_metrics(
    sku_actual_df: pd.DataFrame,
    sku_pred_df: pd.DataFrame,
    eps: float = 1.0,
    metric_name: str = "bounded_mape"
) -> Dict[str, float]:
    eps = _validate_eps(eps)

    common_cols = sorted(set(sku_actual_df.columns).intersection(set(sku_pred_df.columns)))
    if len(common_cols) == 0:
        raise ValueError("No common SKU columns found between sku_actual_df and sku_pred_df")

    y_true = sku_actual_df[common_cols].to_numpy(dtype=float).reshape(-1)
    y_pred = sku_pred_df[common_cols].to_numpy(dtype=float).reshape(-1)

    return compute_overall_metric_bundle(
        y_true=y_true,
        y_pred=y_pred,
        eps=eps,
        metric_name=metric_name
    )


def build_prediction_frame(
    sku_actual_df: pd.DataFrame,
    sku_pred_df: pd.DataFrame,
    cluster_id: int = 0,
    pred_col: str = "pred_baseline"
) -> pd.DataFrame:
    actual_long = (
        sku_actual_df.reset_index(names="date")
        .melt(id_vars="date", var_name="product_family_name", value_name="y")
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    pred_long = (
        sku_pred_df.reset_index(names="date")
        .melt(id_vars="date", var_name="product_family_name", value_name=pred_col)
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    pred_df = actual_long.merge(pred_long, on=["date", "product_family_name"], how="left")
    pred_df["cluster"] = int(cluster_id)
    pred_df["is_sale"] = (pred_df["y"] > 0).astype(int)
    pred_df[pred_col] = pred_df[pred_col].fillna(0.0).astype(float)
    pred_df["y"] = pred_df["y"].astype(float)
    return pred_df


def build_single_prediction_frame(
    sku_actual_df: pd.DataFrame,
    sku_pred_df: pd.DataFrame,
    pred_col: str,
    cluster_id: Optional[int] = None,
) -> pd.DataFrame:
    pred_df = build_prediction_frame(
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        cluster_id=0 if cluster_id is None else int(cluster_id),
        pred_col=pred_col,
    )
    if cluster_id is None and "cluster" in pred_df.columns:
        pred_df = pred_df.drop(columns=["cluster"])
    return pred_df


def _periodize_test(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    out = df.copy()
    unique_dates = np.array(sorted(out["date"].dropna().unique()))
    chunks = np.array_split(unique_dates, n_periods)
    mapper = {}
    for i, chunk in enumerate(chunks, start=1):
        for d in chunk:
            mapper[pd.Timestamp(d)] = f"P{i}"
    out["period"] = out["date"].map(mapper)
    return out


def _build_error_quantiles(ape_box_df: pd.DataFrame) -> pd.DataFrame:
    if ape_box_df.empty:
        return pd.DataFrame(columns=["method", "period", "count", "q50", "q75", "q90", "q95", "q99", "mean"])

    rows = []
    for (method, period), grp in ape_box_df.groupby(["method", "period"]):
        arr = grp["APE_0_100"].values.astype(float)
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


def evaluate_prediction_frame(
    pred_df: pd.DataFrame,
    eps: float = 1.0,
    metric_name: str = "bounded_mape",
    n_periods: int = 4,
    method_name: str = "baseline_ets_fallback",
    pred_col: str = "pred_baseline"
) -> Dict[str, object]:
    out = pred_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0.0)
    if "is_sale" not in out.columns:
        out["is_sale"] = (out["y"] > 0).astype(int)
    else:
        out["is_sale"] = pd.to_numeric(out["is_sale"], errors="coerce").fillna(0).astype(int)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce").fillna(0.0)

    overall_bundle = compute_overall_metric_bundle(
        y_true=out["y"].values,
        y_pred=out[pred_col].values,
        eps=eps,
        metric_name=metric_name,
    )
    metrics_overall = pd.DataFrame(
        [{"method": method_name, "prediction_col": pred_col, **overall_bundle}]
    ).sort_values("MAPE_0_100").reset_index(drop=True)

    out = _periodize_test(out, n_periods=n_periods)
    period_rows = []
    ape_eps = pointwise_safe_ape(out["y"].values, out[pred_col].values, eps=eps)
    ape_cap = np.clip(ape_eps, 0.0, 100.0)

    ape_box_df = out[["date", "period", "product_family_name"]].copy()
    ape_box_df["method"] = method_name
    ape_box_df["prediction_col"] = pred_col
    ape_box_df["APE_EPS_PCT"] = ape_eps
    ape_box_df["APE_CAP_0_100"] = ape_cap
    ape_box_df["APE_0_100"] = ape_cap

    for period, grp in out.groupby("period"):
        metric_bundle = compute_overall_metric_bundle(
            y_true=grp["y"].values,
            y_pred=grp[pred_col].values,
            eps=eps,
            metric_name=metric_name,
        )
        period_rows.append({"method": method_name, "prediction_col": pred_col, "period": period, **metric_bundle})

    metrics_by_period = (
        pd.DataFrame(period_rows)
        .sort_values(["method", "period"])
        .reset_index(drop=True)
    )

    pos = out[out["y"] > 0][["date", "period", "product_family_name", "y"]].copy()
    if pos.empty:
        ape_box_df_positive = pd.DataFrame(
            columns=[
                "date", "period", "product_family_name", "method", "prediction_col",
                "APE_EPS_PCT", "APE_CAP_0_100", "APE_0_100"
            ]
        )
    else:
        ape_pos = pointwise_safe_ape(pos["y"].values, out.loc[pos.index, pred_col].values, eps=1e-12)
        pos["method"] = method_name
        pos["prediction_col"] = pred_col
        pos["APE_EPS_PCT"] = ape_pos
        pos["APE_CAP_0_100"] = np.clip(ape_pos, 0.0, 100.0)
        pos["APE_0_100"] = pos["APE_CAP_0_100"]
        ape_box_df_positive = pos[
            ["date", "period", "product_family_name", "method", "prediction_col", "APE_EPS_PCT", "APE_CAP_0_100", "APE_0_100"]
        ].copy()

    ape_box_df_trimmed = ape_box_df.copy()
    if not ape_box_df_trimmed.empty:
        keep_idx = []
        for (method, period), grp in ape_box_df_trimmed.groupby(["method", "period"]):
            cap = np.quantile(grp["APE_0_100"].values, 0.99)
            keep_idx.extend(grp.index[grp["APE_0_100"] <= cap].tolist())
        ape_box_df_trimmed = ape_box_df_trimmed.loc[keep_idx].copy()

    error_quantiles = _build_error_quantiles(ape_box_df)

    return {
        "pred_df": out,
        "metrics_overall": metrics_overall,
        "metrics_by_period": metrics_by_period,
        "ape_box_df": ape_box_df,
        "ape_box_df_positive": ape_box_df_positive,
        "ape_box_df_trimmed": ape_box_df_trimmed,
        "error_quantiles": error_quantiles,
    }


# --------------------------------------------------
# 5) 一次性输出你最常用的结果
# --------------------------------------------------
def evaluate_forecasts(
    sku_actual_df: pd.DataFrame,
    sku_pred_df: pd.DataFrame,
    eps: float = 1.0,
    metric_name: str = "bounded_mape",
    cluster_id: int = 0,
    n_periods: int = 4,
    method_name: str = "baseline_ets_fallback",
    pred_col: str = "pred_baseline",
) -> Dict[str, object]:
    sku_metrics_df, sku_avg_metrics = compute_sku_level_metrics(
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        eps=eps,
        metric_name=metric_name
    )

    overall_metrics = compute_panel_overall_metrics(
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        eps=eps,
        metric_name=metric_name
    )

    pred_df = build_prediction_frame(
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        cluster_id=cluster_id,
        pred_col=pred_col,
    )
    analysis_outputs = evaluate_prediction_frame(
        pred_df=pred_df,
        eps=eps,
        metric_name=metric_name,
        n_periods=n_periods,
        method_name=method_name,
        pred_col=pred_col,
    )

    return {
        "sku_metrics_df": sku_metrics_df.sort_values("MAIN_MAPE_PCT"),
        "sku_avg_metrics": sku_avg_metrics,
        "overall_metrics": overall_metrics,
        **analysis_outputs,
    }


def build_analysis_artifact(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Any:
    return SimpleNamespace(
        train_feat=train_feat.copy(),
        test_feat=test_feat.copy(),
        pred_df=pred_df.copy(),
    )


def build_pair_prediction_frame(
    sku_actual_df: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
    model_pred_df: pd.DataFrame,
    baseline_col: str,
    model_col: str,
    cluster_id: Optional[int] = None,
) -> pd.DataFrame:
    actual_long = (
        sku_actual_df.reset_index(names="date")
        .melt(id_vars="date", var_name="product_family_name", value_name="y")
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    baseline_long = (
        baseline_pred_df.reset_index(names="date")
        .melt(id_vars="date", var_name="product_family_name", value_name=baseline_col)
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    model_long = (
        model_pred_df.reset_index(names="date")
        .melt(id_vars="date", var_name="product_family_name", value_name=model_col)
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )

    pred_df = (
        actual_long
        .merge(baseline_long, on=["date", "product_family_name"], how="left")
        .merge(model_long, on=["date", "product_family_name"], how="left")
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    pred_df["y"] = pd.to_numeric(pred_df["y"], errors="coerce").fillna(0.0)
    pred_df[baseline_col] = pd.to_numeric(pred_df[baseline_col], errors="coerce").fillna(0.0)
    pred_df[model_col] = pd.to_numeric(pred_df[model_col], errors="coerce").fillna(0.0)
    pred_df["is_sale"] = (pred_df["y"] > 0).astype(int)
    if cluster_id is not None:
        pred_df["cluster"] = int(cluster_id)
    return pred_df
