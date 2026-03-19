from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
except ImportError:  # pragma: no cover
    ExponentialSmoothing = None
    Holt = None
    SimpleExpSmoothing = None

from c0c2_analysis import evaluate_forecasts

date_col = "date"
target_col = "total_sales"
sku_col = "product_family_name"
cluster_col = "cluster"
cluster_id = 2
PRED_COL = "pred_holt_ets"
METHOD_NAME = "c2_m1_holt_ets_baseline"
DISPLAY_NAME = "Holt-Winters / ETS"


def _require_statsmodels() -> None:
    if ExponentialSmoothing is None or Holt is None or SimpleExpSmoothing is None:
        raise ImportError(
            "statsmodels is not installed in the current environment. "
            "Please install `statsmodels` before running c2_m1.py."
        )


def load_data(
    train_path: str = "data/forecasting/train_daily.parquet",
    test_path: str = "data/forecasting/test_daily.parquet",
    cluster_id: int = cluster_id,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train = train_df[train_df[cluster_col] == cluster_id].copy()
    test = test_df[test_df[cluster_col] == cluster_id].copy()

    train[date_col] = pd.to_datetime(train[date_col], errors="coerce")
    test[date_col] = pd.to_datetime(test[date_col], errors="coerce")
    train[target_col] = pd.to_numeric(train[target_col], errors="coerce").fillna(0.0)
    test[target_col] = pd.to_numeric(test[target_col], errors="coerce").fillna(0.0)
    train[sku_col] = train[sku_col].astype("string").str.strip()
    test[sku_col] = test[sku_col].astype("string").str.strip()
    return train, test


def _build_sku_daily_matrix(df: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    grouped = (
        df.groupby([date_col, sku_col], as_index=False)[target_col]
        .sum()
        .pivot(index=date_col, columns=sku_col, values=target_col)
        .reindex(full_index)
        .fillna(0.0)
        .sort_index(axis=1)
    )
    return grouped.astype(float)


def build_daily_series(
    df: pd.DataFrame,
    sku: str,
    full_index: pd.DatetimeIndex,
) -> pd.Series:
    series = (
        df[df[sku_col] == sku]
        .groupby(date_col)[target_col]
        .sum()
        .reindex(full_index)
        .fillna(0.0)
        .astype(float)
    )
    series.name = sku
    return series


def default_params(eps: float = 1.0, val_size: int = 14) -> Dict[str, object]:
    return {
        "seasonal_period": 7,
        "min_train_len": 28,
        "wmape_eps": float(eps),
        "val_size": int(val_size),
        "selection_metric": "0.7*wmape + 0.3*zero_day_fpr",
        "candidate_configs": [
            {"model_type": "ets", "trend": None, "seasonal": "add", "seasonal_periods": 7, "transform": "log1p"},
            {"model_type": "ets", "trend": None, "seasonal": "mul", "seasonal_periods": 7, "transform": "log1p"},
            {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
            {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
            {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
            {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
        ],
    }


def resolve_params(
    params: Optional[Dict[str, object]] = None,
    eps: float = 1.0,
    val_size: int = 14,
) -> Dict[str, object]:
    base = default_params(eps=eps, val_size=val_size)
    if params is None:
        return base
    merged = dict(base)
    merged.update(params)
    return merged


def _wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    numerator = float(np.sum(np.abs(y_true - y_pred)))
    denominator = max(float(np.sum(np.abs(y_true))), float(eps))
    return numerator / denominator * 100.0


def _zero_day_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    zero_mask = y_true <= 0
    if int(np.sum(zero_mask)) == 0:
        return 0.0
    false_positive = np.sum((y_pred > 0) & zero_mask)
    return float(false_positive / np.sum(zero_mask))


def _candidate_to_name(cfg: Dict[str, object]) -> str:
    model_type = str(cfg.get("model_type", "unknown"))
    if model_type == "ses":
        return "ses"
    if model_type == "holt":
        trend = cfg.get("trend", "add")
        damped = bool(cfg.get("damped_trend", False))
        return f"holt_{trend}_{'damped' if damped else 'plain'}"
    if model_type == "ets":
        trend = cfg.get("trend", "none")
        seasonal = cfg.get("seasonal", "none")
        damped = bool(cfg.get("damped_trend", False))
        sp = int(cfg.get("seasonal_periods", 0) or 0)
        return f"ets_t{trend}_s{seasonal}_p{sp}_{'damped' if damped else 'plain'}"
    return model_type


def _fit_statsmodels_candidate(
    train_values: np.ndarray,
    horizon: int,
    candidate_cfg: Dict[str, object],
) -> np.ndarray:
    _require_statsmodels()
    train_values = np.asarray(train_values, dtype=float)
    horizon = int(horizon)
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if len(train_values) == 0:
        return np.zeros(horizon, dtype=float)

    model_type = str(candidate_cfg.get("model_type", "ses"))
    transform = str(candidate_cfg.get("transform", "none")).lower()
    positive_offset = float(candidate_cfg.get("positive_offset", 1e-3))

    transformed_train = train_values.copy()
    inverse_transform = lambda arr: np.asarray(arr, dtype=float)

    if transform == "log1p":
        transformed_train = np.log1p(np.clip(train_values, 0.0, None))

        if candidate_cfg.get("trend") == "mul" or candidate_cfg.get("seasonal") == "mul":
            transformed_train = transformed_train + positive_offset

            def inverse_transform(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=float) - positive_offset
                arr = np.clip(arr, 0.0, None)
                return np.expm1(arr)
        else:
            def inverse_transform(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=float)
                return np.expm1(arr)

    if model_type == "ses":
        fitted = SimpleExpSmoothing(transformed_train, initialization_method="estimated").fit(optimized=True)
        pred = fitted.forecast(horizon)
        return np.clip(inverse_transform(pred), 0.0, None)

    if model_type == "holt":
        fitted = Holt(
            transformed_train,
            exponential=bool(candidate_cfg.get("exponential", False)),
            damped_trend=bool(candidate_cfg.get("damped_trend", False)),
            initialization_method="estimated",
        ).fit(optimized=True)
        pred = fitted.forecast(horizon)
        return np.clip(inverse_transform(pred), 0.0, None)

    if model_type == "ets":
        seasonal_periods = int(candidate_cfg.get("seasonal_periods", 0) or 0)
        if seasonal_periods <= 1 or len(transformed_train) < 2 * seasonal_periods:
            raise ValueError("Not enough history for ETS seasonal candidate")
        fitted = ExponentialSmoothing(
            transformed_train,
            trend=candidate_cfg.get("trend"),
            damped_trend=bool(candidate_cfg.get("damped_trend", False)),
            seasonal=candidate_cfg.get("seasonal"),
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True, remove_bias=False)
        pred = fitted.forecast(horizon)
        return np.clip(inverse_transform(pred), 0.0, None)

    raise ValueError(f"Unsupported candidate model_type: {model_type}")


def fit_baseline_forecast(
    train_series: pd.Series,
    horizon: int,
    params: Dict[str, object],
) -> tuple[np.ndarray, Dict[str, object]]:
    values = np.asarray(train_series, dtype=float)
    val_size = int(params.get("val_size", 14))
    eps = float(params.get("wmape_eps", 1.0))
    candidate_configs = list(params.get("candidate_configs", []))
    if not candidate_configs:
        raise ValueError("candidate_configs must contain at least one candidate model config")
    if len(values) <= val_size:
        raise ValueError("Not enough training history to run per-SKU model selection")

    fit_values = values[:-val_size]
    val_values = values[-val_size:]
    best_score = np.inf
    best_candidate: Optional[Dict[str, object]] = None
    best_val_pred: Optional[np.ndarray] = None
    trial_rows: List[Dict[str, object]] = []

    for candidate_cfg in candidate_configs:
        candidate_name = _candidate_to_name(candidate_cfg)
        try:
            val_pred = _fit_statsmodels_candidate(
                train_values=fit_values,
                horizon=val_size,
                candidate_cfg=candidate_cfg,
            )
            val_wmape = _wmape(val_values, val_pred, eps=eps)
            val_zero_day_fpr = _zero_day_fpr(val_values, val_pred)
            score = 0.7 * val_wmape + 0.3 * val_zero_day_fpr
            trial_rows.append(
                {
                    "candidate_name": candidate_name,
                    "val_wmape": float(val_wmape),
                    "val_zero_day_fpr": float(val_zero_day_fpr),
                    "selection_score": float(score),
                }
            )
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_candidate = dict(candidate_cfg)
                best_val_pred = np.asarray(val_pred, dtype=float)
        except Exception as exc:
            trial_rows.append(
                {
                    "candidate_name": candidate_name,
                    "val_wmape": np.nan,
                    "val_zero_day_fpr": np.nan,
                    "selection_score": np.nan,
                    "error": str(exc),
                }
            )

    if best_candidate is None:
        raise RuntimeError(f"No valid Holt/ETS candidate found for sku={train_series.name}")

    pred = _fit_statsmodels_candidate(
        train_values=values,
        horizon=horizon,
        candidate_cfg=best_candidate,
    )
    return pred, {
        "model_type": str(best_candidate.get("model_type")),
        "candidate_name": _candidate_to_name(best_candidate),
        "trend": best_candidate.get("trend"),
        "seasonal": best_candidate.get("seasonal"),
        "seasonal_periods": best_candidate.get("seasonal_periods"),
        "damped_trend": best_candidate.get("damped_trend"),
        "transform": best_candidate.get("transform"),
        "selection_score": float(best_score),
        "selection_trials": trial_rows,
        "val_pred_sum": float(np.sum(best_val_pred)) if best_val_pred is not None else np.nan,
    }


def forecast_panel(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict[str, object],
    series_cache: Optional[Dict[str, object]] = None,
):
    del series_cache
    resolved_params = resolve_params(params=params)

    train_dates = pd.date_range(train_df[date_col].min(), train_df[date_col].max(), freq="D")
    test_dates = pd.date_range(test_df[date_col].min(), test_df[date_col].max(), freq="D")
    all_skus = sorted(set(train_df[sku_col].unique()).union(set(test_df[sku_col].unique())))

    sku_actual_map = {}
    sku_pred_map = {}
    info_rows = []

    for sku in all_skus:
        train_series = build_daily_series(train_df, sku=sku, full_index=train_dates)
        test_series = build_daily_series(test_df, sku=sku, full_index=test_dates)
        if len(train_series) < int(resolved_params["min_train_len"]):
            raise ValueError(f"Not enough training history for sku={sku}")

        pred, info = fit_baseline_forecast(
            train_series=train_series,
            horizon=len(test_series),
            params=resolved_params,
        )
        sku_actual_map[sku] = test_series
        sku_pred_map[sku] = pd.Series(pred, index=test_dates, name=sku)
        info_rows.append(
            {
                "sku": sku,
                "model_type": info["model_type"],
                "candidate_name": info.get("candidate_name"),
                "trend": info.get("trend"),
                "seasonal": info.get("seasonal"),
                "seasonal_periods": info.get("seasonal_periods"),
                "damped_trend": info.get("damped_trend"),
                "transform": info.get("transform"),
                "selection_score": info.get("selection_score"),
                "train_sum": float(train_series.sum()),
                "test_sum": float(test_series.sum()),
                "pred_sum": float(np.sum(pred)),
            }
        )

    sku_actual_df = pd.DataFrame(sku_actual_map, index=test_dates).sort_index(axis=1)
    sku_pred_df = pd.DataFrame(sku_pred_map, index=test_dates).sort_index(axis=1)
    sku_model_info_df = pd.DataFrame(info_rows).sort_values("sku").reset_index(drop=True)
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
    del eps
    if mode == "fast":
        return {
            "min_train_len": [28],
            "val_size": [14],
            "candidate_configs": [[
                {"model_type": "ets", "trend": None, "seasonal": "add", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": None, "seasonal": "mul", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
            ]],
        }
    return {
        "min_train_len": [21, 28, 56],
        "val_size": [7, 14, 21],
        "candidate_configs": [
            [
                {"model_type": "ets", "trend": None, "seasonal": "add", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": None, "seasonal": "mul", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
            ],
            [
                {"model_type": "ets", "trend": None, "seasonal": "add", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": None, "seasonal": "mul", "seasonal_periods": 7, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": False, "transform": "log1p"},
            ],
            [
                {"model_type": "ets", "trend": "add", "seasonal": "add", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
                {"model_type": "ets", "trend": "add", "seasonal": "mul", "seasonal_periods": 7, "damped_trend": True, "transform": "log1p"},
            ],
        ],
    }


def sample_random_param_sets(
    param_space: Dict[str, List[object]],
    n_iter: int,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(random_state)
    sampled: List[Dict[str, object]] = []
    seen = set()
    max_attempts = max(100, int(n_iter) * 20)
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


def tune_hyperparams_random_search(
    train_df: pd.DataFrame,
    param_space: Dict[str, List[object]],
    val_days: int = 28,
    n_iter: int = 8,
    random_state: int = 42,
    verbose: bool = True,
):
    train_part, val_part = make_train_val_split(train_df, val_days=val_days)
    trial_params = sample_random_param_sets(param_space, n_iter=n_iter, random_state=random_state)

    tuning_rows = []
    best_params = None
    best_score = np.inf

    for i, params in enumerate(trial_params, start=1):
        try:
            resolved_params = resolve_params(params=params)
            val_actual_df, val_pred_df, _ = forecast_panel(train_df=train_part, test_df=val_part, params=resolved_params)
            eval_out = evaluate_forecasts(
                sku_actual_df=val_actual_df,
                sku_pred_df=val_pred_df,
                eps=float(resolved_params["wmape_eps"]),
                metric_name="bounded_mape",
                cluster_id=cluster_id,
                n_periods=4,
                method_name="holt_ets_baseline",
                pred_col="pred_holt_ets",
            )
            metrics = eval_out["metrics_overall"].iloc[0].to_dict()
            score = float(metrics["WMAPE_0_100"])
            row = {
                **resolved_params,
                "search_score": score,
                "metric_mape": float(metrics["MAPE_0_100"]),
                "metric_epsilon_mape": float(metrics["EPSILON_MAPE_PCT"]),
                "metric_cap_mape": float(metrics["CAP_MAPE_0_100"]),
                "metric_wmape": float(metrics["WMAPE_0_100"]),
                "metric_zero_day_fpr": float(metrics["ZERO_DAY_FPR"]),
            }
            tuning_rows.append(row)
            if verbose:
                print(
                    f"[{i}/{len(trial_params)}] "
                    f"score={score:.4f} "
                    f"cap_mape={metrics['CAP_MAPE_0_100']:.4f} "
                    f"wmape={metrics['WMAPE_0_100']:.4f}"
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
    train, test = load_data(cluster_id=cluster_id)
    param_space = default_param_space(eps=1.0, mode="fast")
    best_params, tuning_df = tune_hyperparams_random_search(
        train_df=train,
        param_space=param_space,
        val_days=28,
        n_iter=8,
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
        method_name="holt_ets_baseline",
        pred_col="pred_holt_ets",
    )

    print("=== Best Params ===")
    print(best_params)

    print("\n=== Top Random Search Results ===")
    cols = ["search_score", "metric_cap_mape", "metric_wmape", "val_size", "min_train_len"]
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
