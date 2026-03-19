import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from c0c2_analysis import evaluate_forecasts

date_col = "date"
target_col = "total_sales"
sku_col = "product_family_name"
cluster_col = "cluster"
cluster_id = 0
PRED_COL = "pred_tsb"
METHOD_NAME = "c0_m1_tsb_baseline"
DISPLAY_NAME = "TSB"


def load_data(
    train_path: str = "data/forecasting/train_daily.parquet",
    test_path: str = "data/forecasting/test_daily.parquet",
    cluster_id: int = cluster_id,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train = train_df[train_df[cluster_col] == cluster_id].copy()
    test = test_df[test_df[cluster_col] == cluster_id].copy()

    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col])
    return train, test


def build_daily_series(
    df: pd.DataFrame,
    sku: str,
    full_index: pd.DatetimeIndex,
    date_col: str = "date",
    target_col: str = "total_sales",
    sku_col: str = "product_family_name",
) -> pd.Series:
    tmp = (
        df[df[sku_col] == sku]
        .groupby(date_col, as_index=False)[target_col]
        .sum()
        .sort_values(date_col)
        .copy()
    )

    if len(tmp) == 0:
        return pd.Series(0.0, index=full_index, name=sku)

    tmp[date_col] = pd.to_datetime(tmp[date_col])
    s = tmp.set_index(date_col)[target_col].reindex(full_index, fill_value=0.0).astype(float)
    s.name = sku
    return s


def compute_history_cap(
    train_values,
    cap_quantile: float,
    cap_multiplier: float,
    cap_lookback: int,
) -> float:
    arr = np.asarray(train_values, dtype=float)
    if len(arr) == 0:
        return 0.0

    if len(arr) > int(cap_lookback):
        arr = arr[-int(cap_lookback):]

    nz = arr[arr > 0]
    if len(nz) == 0:
        return 0.0

    q = float(np.percentile(nz, float(cap_quantile)))
    mean_level = float(np.mean(nz)) * float(cap_multiplier)
    return max(0.0, min(max(q, mean_level), float(np.max(nz) * 1.5)))


def apply_postprocess(pred, sale_threshold: float, cap_value: Optional[float] = None) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, 0.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    if cap_value is not None and np.isfinite(cap_value):
        pred = np.clip(pred, 0.0, float(cap_value))
    pred[pred < float(sale_threshold)] = 0.0
    return pred


def zero_day_false_positive_rate(y_true, y_pred, pred_positive_threshold: float) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    zero_mask = yt <= 0
    if zero_mask.sum() == 0:
        return 0.0
    fp = np.sum(yp[zero_mask] > float(pred_positive_threshold))
    return float(fp / zero_mask.sum())


def zero_day_overshoot_pct(
    y_true,
    y_pred,
    eps: float,
    pred_positive_threshold: float,
    zero_scale_quantile: float,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    zero_mask = yt <= 0
    if zero_mask.sum() == 0:
        return 0.0

    overshoot = np.maximum(yp[zero_mask], 0.0)
    overshoot = overshoot[overshoot > float(pred_positive_threshold)]
    if len(overshoot) == 0:
        return 0.0

    pos = yt[yt > 0]
    scale = max(float(np.percentile(pos, float(zero_scale_quantile))) if len(pos) > 0 else 0.0, float(eps))
    return float(np.mean(overshoot) / scale * 100.0)


def default_params(eps: float = 1.0) -> Dict[str, object]:
    return {
        "min_train_len": 28,
        "min_nonzero_count": 6,
        "min_nonzero_ratio": 0.05,
        "wmape_eps": float(eps),
        "sale_threshold": 1.0,
        "pred_positive_threshold": 0.5,
        "lambda_zero_fp": 50.0,
        "lambda_zero_overshoot": 0.20,
        "zero_scale_quantile": 50.0,
        "cap_quantile": 95.0,
        "cap_multiplier": 1.5,
        "cap_lookback": 84,
        "fallback_lookback_days": 56,
        "fallback_global_window": 28,
        "fallback_shrink": 0.7,
        "fallback_cap_quantile": 90.0,
        "fallback_cap_multiplier": 1.5,
        "fallback_cap_lookback": 56,
        "alpha_d": 0.15,
        "alpha_p": 0.10,
    }


def resolve_params(params: Optional[Dict[str, object]] = None, eps: float = 1.0) -> Dict[str, object]:
    base = default_params(eps=eps)
    if params is None:
        return base
    merged = dict(base)
    merged.update(params)
    return merged


def fallback_forecast_weekday(train_series: pd.Series, horizon: int, params: Dict[str, object]) -> np.ndarray:
    y = train_series.values.astype(float)
    idx = pd.DatetimeIndex(train_series.index)

    if len(y) == 0:
        return np.zeros(horizon, dtype=float)

    future_index = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    lookback_days = int(params["fallback_lookback_days"])
    global_window = int(params["fallback_global_window"])
    shrink = float(params["fallback_shrink"])

    hist = train_series.iloc[-lookback_days:].copy() if len(train_series) >= lookback_days else train_series.copy()
    hist_idx = pd.DatetimeIndex(hist.index)

    recent_vals = hist.iloc[-global_window:].values.astype(float) if len(hist) >= global_window else hist.values.astype(float)
    global_mean = float(np.mean(recent_vals)) if len(recent_vals) > 0 else 0.0
    global_mean = global_mean if np.isfinite(global_mean) else 0.0

    weekday_means = {}
    for wd in range(7):
        vals = hist[hist_idx.weekday == wd].values.astype(float)
        if len(vals) > 0:
            weekday_means[wd] = shrink * float(np.mean(vals)) + (1.0 - shrink) * global_mean
        else:
            weekday_means[wd] = global_mean

    cap_value = compute_history_cap(
        train_values=hist.values,
        cap_quantile=float(params["fallback_cap_quantile"]),
        cap_multiplier=float(params["fallback_cap_multiplier"]),
        cap_lookback=int(params["fallback_cap_lookback"]),
    )
    preds = np.array([weekday_means[dt.weekday()] for dt in future_index], dtype=float)
    return apply_postprocess(preds, sale_threshold=float(params["sale_threshold"]), cap_value=cap_value)


def fit_tsb(train_values, horizon: int, params: Dict[str, object]) -> np.ndarray:
    y = np.asarray(train_values, dtype=float)
    if len(y) == 0:
        return np.zeros(horizon, dtype=float)

    z = (y > 0).astype(float)
    pos = y[y > 0]
    if len(pos) == 0:
        return np.zeros(horizon, dtype=float)

    alpha_d = float(params["alpha_d"])
    alpha_p = float(params["alpha_p"])

    p_t = float(z[0])
    d_t = float(pos[0])

    for val, occ in zip(y, z):
        p_t = alpha_p * occ + (1.0 - alpha_p) * p_t
        if occ > 0:
            d_t = alpha_d * float(val) + (1.0 - alpha_d) * d_t

    raw_pred = np.full(horizon, p_t * d_t, dtype=float)
    cap_value = compute_history_cap(
        train_values=y,
        cap_quantile=float(params["cap_quantile"]),
        cap_multiplier=float(params["cap_multiplier"]),
        cap_lookback=int(params["cap_lookback"]),
    )
    return apply_postprocess(raw_pred, sale_threshold=float(params["sale_threshold"]), cap_value=cap_value)


def fit_tsb_or_fallback(
    train_series: pd.Series,
    horizon: int,
    params: Optional[Dict[str, object]] = None,
    eps: float = 1.0,
):
    params = resolve_params(params=params, eps=eps)
    y = train_series.values.astype(float)

    if len(y) == 0:
        return np.zeros(horizon, dtype=float), {"model_type": "all_zero"}

    nonzero_count = int(np.sum(y > 0))
    nonzero_ratio = float(np.mean(y > 0))
    if len(y) < int(params["min_train_len"]):
        pred = fallback_forecast_weekday(train_series, horizon, params=params)
        return pred, {"model_type": "fallback_short"}
    if nonzero_count < int(params["min_nonzero_count"]) or nonzero_ratio < float(params["min_nonzero_ratio"]):
        pred = fallback_forecast_weekday(train_series, horizon, params=params)
        return pred, {"model_type": "fallback_sparse"}

    pred = fit_tsb(y, horizon=horizon, params=params)
    return pred, {
        "model_type": "tsb",
        "nonzero_count": nonzero_count,
        "nonzero_ratio": nonzero_ratio,
    }


def build_panel_series_cache(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    train_dates = pd.date_range(train_df[date_col].min(), train_df[date_col].max(), freq="D")
    test_dates = pd.date_range(test_df[date_col].min(), test_df[date_col].max(), freq="D")
    all_skus = sorted(set(train_df[sku_col].unique()).union(set(test_df[sku_col].unique())))

    train_series_map = {
        sku: build_daily_series(train_df, sku, full_index=train_dates, date_col=date_col, target_col=target_col, sku_col=sku_col)
        for sku in all_skus
    }
    test_series_map = {
        sku: build_daily_series(test_df, sku, full_index=test_dates, date_col=date_col, target_col=target_col, sku_col=sku_col)
        for sku in all_skus
    }
    return {
        "train_dates": train_dates,
        "test_dates": test_dates,
        "all_skus": all_skus,
        "train_series_map": train_series_map,
        "test_series_map": test_series_map,
    }


def forecast_panel(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict[str, object],
    series_cache: Optional[Dict[str, object]] = None,
):
    params = resolve_params(params=params, eps=float(params.get("wmape_eps", 1.0)))
    if series_cache is None:
        series_cache = build_panel_series_cache(train_df, test_df)

    test_dates = series_cache["test_dates"]
    all_skus = series_cache["all_skus"]
    train_series_map = series_cache["train_series_map"]
    test_series_map = series_cache["test_series_map"]

    sku_preds = {}
    sku_actuals = {}
    sku_model_info = []

    for sku in all_skus:
        train_s = train_series_map[sku]
        test_s = test_series_map[sku]
        pred, info = fit_tsb_or_fallback(
            train_s,
            horizon=len(test_dates),
            params=params,
            eps=float(params["wmape_eps"]),
        )
        sku_preds[sku] = pd.Series(pred, index=test_dates, name=sku)
        sku_actuals[sku] = pd.Series(test_s.values, index=test_dates, name=sku)
        sku_model_info.append(
            {
                "sku": sku,
                "model_type": info.get("model_type"),
                "train_len": int(len(train_s)),
                "nonzero_count": int(np.sum(train_s.values > 0)),
                "nonzero_ratio": float(np.mean(train_s.values > 0)) if len(train_s) > 0 else np.nan,
            }
        )

    sku_pred_df = pd.DataFrame(sku_preds, index=test_dates).sort_index(axis=1)
    sku_actual_df = pd.DataFrame(sku_actuals, index=test_dates).sort_index(axis=1)
    sku_model_info_df = pd.DataFrame(sku_model_info).sort_values(["model_type", "nonzero_ratio"], na_position="last")
    return sku_actual_df, sku_pred_df, sku_model_info_df


def make_train_val_split(train_df: pd.DataFrame, val_days: int):
    all_dates = pd.Series(pd.to_datetime(train_df[date_col]).sort_values().unique())
    if len(all_dates) <= val_days:
        raise ValueError("Not enough train dates to create validation split.")

    cutoff = all_dates.iloc[-val_days]
    train_part = train_df[train_df[date_col] < cutoff].copy()
    val_part = train_df[train_df[date_col] >= cutoff].copy()
    if len(train_part) == 0 or len(val_part) == 0:
        raise ValueError("Train/validation split is empty.")
    return train_part, val_part


def default_param_space(eps: float = 1.0, mode: str = "fast") -> Dict[str, List[object]]:
    return {
        "alpha_d": [0.10, 0.15, 0.20, 0.30],
        "alpha_p": [0.05, 0.10, 0.15, 0.20],
        "sale_threshold": [0.5, 1.0, 2.0],
        "cap_quantile": [90.0, 95.0],
        "cap_multiplier": [1.25, 1.5],
    }


def sample_random_param_sets(
    param_space: Dict[str, List[object]],
    n_iter: int,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    rng = random.Random(random_state)
    sampled = []
    seen = set()
    max_attempts = max(100, int(n_iter) * 20)
    attempts = 0

    while len(sampled) < int(n_iter) and attempts < max_attempts:
        attempts += 1
        params = {k: rng.choice(v) for k, v in param_space.items()}
        key = tuple((k, repr(params[k])) for k in sorted(params))
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)
    return sampled


def tune_hyperparams_random_search(
    train_df: pd.DataFrame,
    param_space: Dict[str, List[object]],
    val_days: int = 28,
    n_iter: int = 12,
    random_state: int = 42,
    verbose: bool = True,
):
    train_part, val_part = make_train_val_split(train_df, val_days=val_days)
    series_cache = build_panel_series_cache(train_part, val_part)
    trial_params = sample_random_param_sets(param_space, n_iter=n_iter, random_state=random_state)

    tuning_rows = []
    best_params = None
    best_score = np.inf

    for i, params in enumerate(trial_params, start=1):
        try:
            resolved_params = resolve_params(params=params)
            val_actual_df, val_pred_df, _ = forecast_panel(
                train_df=train_part,
                test_df=val_part,
                params=resolved_params,
                series_cache=series_cache,
            )
            eval_out = evaluate_forecasts(
                sku_actual_df=val_actual_df,
                sku_pred_df=val_pred_df,
                eps=float(resolved_params["wmape_eps"]),
                metric_name="bounded_mape",
                cluster_id=cluster_id,
                n_periods=4,
                method_name="tsb_baseline",
                pred_col="pred_tsb",
            )
            metrics = eval_out["metrics_overall"].iloc[0].to_dict()
            pred_long = eval_out["pred_df"]
            zero_overshoot = zero_day_overshoot_pct(
                pred_long["y"].to_numpy(),
                pred_long["pred_tsb"].to_numpy(),
                eps=float(resolved_params["wmape_eps"]),
                pred_positive_threshold=float(resolved_params["pred_positive_threshold"]),
                zero_scale_quantile=float(resolved_params["zero_scale_quantile"]),
            )
            score = (
                float(metrics["CAP_MAPE_0_100"])
                + float(resolved_params["lambda_zero_fp"]) * float(metrics["ZERO_DAY_FPR"])
                + float(resolved_params["lambda_zero_overshoot"]) * zero_overshoot
            )
            row = {
                **resolved_params,
                "search_score": float(score),
                "metric_mape": float(metrics["MAPE_0_100"]),
                "metric_epsilon_mape": float(metrics["EPSILON_MAPE_PCT"]),
                "metric_cap_mape": float(metrics["CAP_MAPE_0_100"]),
                "metric_wmape": float(metrics["WMAPE_0_100"]),
                "metric_zero_day_fpr": float(metrics["ZERO_DAY_FPR"]),
                "zero_day_overshoot_pct": float(zero_overshoot),
            }
            tuning_rows.append(row)
            if verbose:
                print(
                    f"[{i}/{len(trial_params)}] "
                    f"score={score:.4f} "
                    f"cap_mape={metrics['CAP_MAPE_0_100']:.4f} "
                    f"zero_fpr={metrics['ZERO_DAY_FPR']:.4f} "
                    f"overshoot={zero_overshoot:.4f}"
                )
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_params = dict(resolved_params)
        except Exception as e:
            tuning_rows.append({**params, "search_score": np.nan, "error": str(e)})
            if verbose:
                print(f"[{i}/{len(trial_params)}] FAILED | error={e}")

    tuning_df = pd.DataFrame(tuning_rows).sort_values("search_score", na_position="last").reset_index(drop=True)
    return best_params, tuning_df


def main():
    train, test = load_data()
    param_space = default_param_space(eps=1.0, mode="fast")
    best_params, tuning_df = tune_hyperparams_random_search(
        train_df=train,
        param_space=param_space,
        val_days=28,
        n_iter=12,
        random_state=42,
        verbose=True,
    )

    if best_params is None:
        raise RuntimeError("Random search did not produce a valid parameter set.")

    sku_actual_df, sku_pred_df, sku_model_info_df = forecast_panel(train, test, params=best_params)
    results = evaluate_forecasts(
        sku_actual_df=sku_actual_df,
        sku_pred_df=sku_pred_df,
        eps=float(best_params["wmape_eps"]),
        metric_name="bounded_mape",
        cluster_id=cluster_id,
        n_periods=4,
        method_name="tsb_baseline",
        pred_col="pred_tsb",
    )

    print("=== Best Params ===")
    print(best_params)

    print("\n=== Top Random Search Results ===")
    cols = [
        "search_score",
        "metric_cap_mape",
        "metric_epsilon_mape",
        "metric_wmape",
        "metric_zero_day_fpr",
        "zero_day_overshoot_pct",
        "alpha_d",
        "alpha_p",
        "sale_threshold",
    ]
    print(tuning_df[cols].head(10))

    print("\n=== Overall panel metrics ===")
    for k, v in results["overall_metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n=== Model type counts ===")
    print(sku_model_info_df["model_type"].value_counts(dropna=False))

    return {
        "best_params": best_params,
        "tuning_df": tuning_df,
        "sku_actual_df": sku_actual_df,
        "sku_pred_df": sku_pred_df,
        "sku_model_info_df": sku_model_info_df,
        "results": results,
    }


if __name__ == "__main__":
    main()
