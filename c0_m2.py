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
    "sku_code",
    "dow",
    "dow_sin",
    "dow_cos",
    "dom",
    "weekofyear",
    "weekofyear_sin",
    "weekofyear_cos",
    "month",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "sku_train_mean",
    "sku_train_std",
    "sku_train_nonzero_rate",
    "sku_train_p_zero",
    "sku_train_positive_mean",
    "sku_train_positive_median",
    "sku_train_adi",
    "sku_train_cv2",
    "sku_train_mean_interarrival",
    "sku_last_nonzero_sales",
    "days_since_last_sale",
    "lag_1_proxy",
    "lag_7_proxy",
    "roll_mean_7_proxy",
    "roll_mean_28_proxy",
    "ewm_mean_7",
    "ewm_mean_28",
    "ewm_sale_prob_14",
    "ewm_pos_mean_14",
]
DEFAULT_OCCURRENCE_FEATURE_COLS = [
    "sku_code",
    "dow",
    "dow_sin",
    "dow_cos",
    "weekofyear",
    "weekofyear_sin",
    "weekofyear_cos",
    "month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "sku_train_nonzero_rate",
    "sku_train_p_zero",
    "sku_train_adi",
    "sku_train_mean_interarrival",
    "days_since_last_sale",
    "ewm_sale_prob_14",
    "sale_occur_lag_1",
    "sale_occur_lag_7",
    "sale_occur_roll_7",
    "sale_occur_roll_14",
    "sale_occur_roll_28",
    "sale_occur_ewm_7",
    "sale_occur_ewm_28",
]
DEFAULT_REGRESSION_FEATURE_COLS = [
    "sku_code",
    "dow",
    "dow_sin",
    "dow_cos",
    "dom",
    "weekofyear",
    "weekofyear_sin",
    "weekofyear_cos",
    "month",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "sku_train_mean",
    "sku_train_std",
    "sku_train_positive_mean",
    "sku_train_positive_median",
    "sku_train_cv2",
    "sku_last_nonzero_sales",
    "days_since_last_sale",
    "lag_1_proxy",
    "lag_7_proxy",
    "roll_mean_7_proxy",
    "roll_mean_28_proxy",
    "ewm_mean_7",
    "ewm_mean_28",
    "ewm_pos_mean_14",
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


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["dom"] = out["date"].dt.day
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    out["is_quarter_start"] = out["date"].dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = out["date"].dt.is_quarter_end.astype(int)

    out["dow_sin"] = np.sin(2.0 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dow"] / 7.0)
    out["weekofyear_sin"] = np.sin(2.0 * np.pi * out["weekofyear"] / 52.0)
    out["weekofyear_cos"] = np.cos(2.0 * np.pi * out["weekofyear"] / 52.0)
    return out


def _compute_sku_profile_features(train_panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sku_idx, (sku, grp) in enumerate(train_panel.groupby("product_family_name"), start=1):
        grp = grp.sort_values("date")
        y = grp["total_sales"].astype(float).values
        y_pos = y[y > 0]
        nonzero_count = int(np.sum(y > 0))
        total_count = int(len(y))
        p_zero = float(np.mean(y <= 0))
        nonzero_rate = float(1.0 - p_zero)
        pos_idx = np.where(y > 0)[0]
        if len(pos_idx) >= 2:
            gaps = np.diff(pos_idx).astype(float)
            mean_interarrival = float(np.mean(gaps))
            p90_interarrival = float(np.quantile(gaps, 0.90))
        else:
            mean_interarrival = float(total_count if total_count > 0 else 999.0)
            p90_interarrival = float(mean_interarrival)

        adi = float(total_count / max(nonzero_count, 1))
        if len(y_pos) > 1 and float(np.mean(y_pos)) > 0:
            pos_mean = float(np.mean(y_pos))
            pos_std = float(np.std(y_pos, ddof=0))
            cv2 = float((pos_std / max(pos_mean, 1e-8)) ** 2)
        else:
            cv2 = 0.0

        last_nonzero = float(y_pos[-1]) if len(y_pos) > 0 else 0.0
        last_sale_date = grp.loc[grp["total_sales"] > 0, "date"].max() if len(y_pos) > 0 else pd.NaT
        last_lag1 = float(y[-1]) if total_count >= 1 else 0.0
        last_lag7 = float(y[-7]) if total_count >= 7 else float(np.mean(y[-min(total_count, 7) :])) if total_count > 0 else 0.0
        last_roll7 = float(np.mean(y[-min(total_count, 7) :])) if total_count > 0 else 0.0
        last_roll28 = float(np.mean(y[-min(total_count, 28) :])) if total_count > 0 else 0.0

        rows.append(
            {
                "product_family_name": sku,
                "sku_code": int(sku_idx),
                "sku_train_mean": float(np.mean(y)) if total_count > 0 else 0.0,
                "sku_train_std": float(np.std(y, ddof=0)) if total_count > 0 else 0.0,
                "sku_train_nonzero_rate": nonzero_rate,
                "sku_train_p_zero": p_zero,
                "sku_train_positive_mean": float(np.mean(y_pos)) if len(y_pos) > 0 else 0.0,
                "sku_train_positive_median": float(np.median(y_pos)) if len(y_pos) > 0 else 0.0,
                "sku_train_adi": adi,
                "sku_train_cv2": cv2,
                "sku_train_mean_interarrival": mean_interarrival,
                "sku_last_nonzero_sales": last_nonzero,
                "sku_last_sale_date": last_sale_date,
                "sku_last_lag1": last_lag1,
                "sku_last_lag7": last_lag7,
                "sku_last_roll7": last_roll7,
                "sku_last_roll28": last_roll28,
            }
        )

    return pd.DataFrame(rows)


def _add_ewm_history_features(train_feat: pd.DataFrame) -> pd.DataFrame:
    out = train_feat.copy().sort_values(["product_family_name", "date"])
    grp = out.groupby("product_family_name", group_keys=False)

    y_prev = grp["y"].shift(1)
    sale_prev = grp["is_sale"].shift(1)
    pos_prev = y_prev.where(y_prev > 0)

    out["ewm_mean_7"] = y_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=7, adjust=False, min_periods=1).mean()
    )
    out["ewm_mean_28"] = y_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=28, adjust=False, min_periods=1).mean()
    )
    out["ewm_sale_prob_14"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=14, adjust=False, min_periods=1).mean()
    )
    out["ewm_pos_mean_14"] = pos_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=14, adjust=False, min_periods=1).mean()
    )
    return out


def _add_ewm_test_features(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> pd.DataFrame:
    out = test_feat.copy().sort_values(["product_family_name", "date"])
    spans = {
        "ewm_mean_7": 7,
        "ewm_mean_28": 28,
        "ewm_sale_prob_14": 14,
        "ewm_pos_mean_14": 14,
    }

    last_state = (
        train_feat.sort_values(["product_family_name", "date"])
        .groupby("product_family_name", as_index=False)
        .tail(1)[["product_family_name"] + list(spans.keys())]
        .set_index("product_family_name")
    )

    out["_h"] = out.groupby("product_family_name").cumcount() + 1
    out = out.merge(last_state, on="product_family_name", how="left")
    prior_map = {
        "ewm_mean_7": "sku_train_mean",
        "ewm_mean_28": "sku_train_mean",
        "ewm_sale_prob_14": "sku_train_nonzero_rate",
        "ewm_pos_mean_14": "sku_train_positive_mean",
    }
    for feat, span in spans.items():
        alpha = 2.0 / (float(span) + 1.0)
        beta = 1.0 - alpha
        prior = out[prior_map[feat]].astype(float).fillna(0.0)
        decay = np.power(beta, out["_h"].astype(float))
        out[feat] = out[feat].fillna(0.0) * decay + prior * (1.0 - decay)
    out = out.drop(columns=["_h"])
    return out


def _add_occurrence_history_features(train_feat: pd.DataFrame) -> pd.DataFrame:
    out = train_feat.copy().sort_values(["product_family_name", "date"])
    grp = out.groupby("product_family_name", group_keys=False)
    sale_prev = grp["is_sale"].shift(1)
    out["sale_occur_lag_1"] = sale_prev
    out["sale_occur_lag_7"] = grp["is_sale"].shift(7)
    out["sale_occur_roll_7"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["sale_occur_roll_14"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.rolling(14, min_periods=1).mean()
    )
    out["sale_occur_roll_28"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    out["sale_occur_ewm_7"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=7, adjust=False, min_periods=1).mean()
    )
    out["sale_occur_ewm_28"] = sale_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.ewm(span=28, adjust=False, min_periods=1).mean()
    )
    return out


def _add_occurrence_test_features(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> pd.DataFrame:
    out = test_feat.copy().sort_values(["product_family_name", "date"])
    occ_cols = [
        "sale_occur_lag_1",
        "sale_occur_lag_7",
        "sale_occur_roll_7",
        "sale_occur_roll_14",
        "sale_occur_roll_28",
        "sale_occur_ewm_7",
        "sale_occur_ewm_28",
    ]
    last_occ_state = (
        train_feat.sort_values(["product_family_name", "date"])
        .groupby("product_family_name", as_index=False)
        .tail(1)[["product_family_name"] + occ_cols]
    )
    out = out.merge(last_occ_state, on="product_family_name", how="left")
    out["_h"] = out.groupby("product_family_name").cumcount() + 1

    def _decay_prob(col: str, beta: float) -> pd.Series:
        last = out[col].astype(float).fillna(0.0)
        prior = out["sku_train_nonzero_rate"].astype(float).fillna(0.0)
        decay = np.power(beta, out["_h"].astype(float))
        return last * decay + prior * (1.0 - decay)

    out["sale_occur_lag_1"] = _decay_prob("sale_occur_lag_1", beta=0.65)
    out["sale_occur_lag_7"] = _decay_prob("sale_occur_lag_7", beta=0.85)
    out["sale_occur_roll_7"] = _decay_prob("sale_occur_roll_7", beta=0.85)
    out["sale_occur_roll_14"] = _decay_prob("sale_occur_roll_14", beta=0.90)
    out["sale_occur_roll_28"] = _decay_prob("sale_occur_roll_28", beta=0.95)
    out["sale_occur_ewm_7"] = _decay_prob("sale_occur_ewm_7", beta=0.85)
    out["sale_occur_ewm_28"] = _decay_prob("sale_occur_ewm_28", beta=0.95)

    out = out.drop(columns=["_h"])
    return out


def _add_train_dynamic_proxy_features(train_feat: pd.DataFrame) -> pd.DataFrame:
    out = train_feat.copy().sort_values(["product_family_name", "date"])
    grp = out.groupby("product_family_name", group_keys=False)
    out["lag_1_proxy"] = grp["y"].shift(1)
    out["lag_7_proxy"] = grp["y"].shift(7)
    y_prev = grp["y"].shift(1)
    out["roll_mean_7_proxy"] = y_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out["roll_mean_28_proxy"] = y_prev.groupby(out["product_family_name"]).transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    return out


def _add_test_dynamic_proxy_features(test_feat: pd.DataFrame) -> pd.DataFrame:
    out = test_feat.copy().sort_values(["product_family_name", "date"])
    out["_h"] = out.groupby("product_family_name").cumcount() + 1

    def _decay_to_prior(last_col: str, prior_col: str, beta: float) -> pd.Series:
        decay = np.power(beta, out["_h"].astype(float))
        last = out[last_col].astype(float).fillna(0.0)
        prior = out[prior_col].astype(float).fillna(0.0)
        return last * decay + prior * (1.0 - decay)

    out["lag_1_proxy"] = _decay_to_prior("sku_last_lag1", "sku_train_mean", beta=0.75)
    out["lag_7_proxy"] = _decay_to_prior("sku_last_lag7", "sku_train_mean", beta=0.88)
    out["roll_mean_7_proxy"] = _decay_to_prior("sku_last_roll7", "sku_train_mean", beta=0.90)
    out["roll_mean_28_proxy"] = _decay_to_prior("sku_last_roll28", "sku_train_mean", beta=0.96)
    out = out.drop(columns=["_h"])
    return out


def _prepare_model_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    category_levels: Optional[Dict[str, List[Any]]] = None,
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    category_levels = {} if category_levels is None else category_levels
    for col in feature_cols:
        if col in category_levels:
            X[col] = pd.Categorical(X[col], categories=category_levels[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
    return X


def _build_features(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_feat = train_panel.copy().sort_values(["product_family_name", "date"])
    test_feat = test_panel.copy().sort_values(["product_family_name", "date"])

    train_feat["y"] = train_feat["total_sales"].astype(float)
    test_feat["y"] = test_feat["total_sales"].astype(float)
    train_feat["is_sale"] = (train_feat["y"] > 0).astype(int)
    test_feat["is_sale"] = (test_feat["y"] > 0).astype(int)

    train_feat = _add_calendar_features(train_feat)
    test_feat = _add_calendar_features(test_feat)

    sku_profile = _compute_sku_profile_features(train_panel)
    train_feat = train_feat.merge(sku_profile, on="product_family_name", how="left")
    test_feat = test_feat.merge(sku_profile, on="product_family_name", how="left")

    # Train side uses historical last-sale date shifted by one day to preserve causality.
    train_last_sale = train_feat["date"].where(train_feat["y"] > 0)
    train_feat["hist_last_sale_date"] = (
        train_last_sale.groupby(train_feat["product_family_name"]).ffill().groupby(train_feat["product_family_name"]).shift(1)
    )
    train_feat["days_since_last_sale"] = (train_feat["date"] - train_feat["hist_last_sale_date"]).dt.days
    train_feat["days_since_last_sale"] = train_feat["days_since_last_sale"].fillna(999).astype(float)

    # Test side uses only last observed sale date from train period.
    test_feat["days_since_last_sale"] = (test_feat["date"] - test_feat["sku_last_sale_date"]).dt.days
    test_feat["days_since_last_sale"] = test_feat["days_since_last_sale"].fillna(999).astype(float)

    train_feat = train_feat.drop(columns=["hist_last_sale_date", "sku_last_sale_date"])
    test_feat = test_feat.drop(columns=["sku_last_sale_date"])

    train_feat = _add_train_dynamic_proxy_features(train_feat)
    test_feat = _add_test_dynamic_proxy_features(test_feat)
    train_feat = _add_ewm_history_features(train_feat)
    test_feat = _add_ewm_test_features(train_feat=train_feat, test_feat=test_feat)
    train_feat = _add_occurrence_history_features(train_feat)
    test_feat = _add_occurrence_test_features(train_feat=train_feat, test_feat=test_feat)

    train_feat["sku_code"] = train_feat["sku_code"].astype("Int64").astype(str)
    test_feat["sku_code"] = test_feat["sku_code"].astype("Int64").astype(str)

    numeric_fill_zero_cols = [
        "sku_train_mean",
        "sku_train_std",
        "sku_train_nonzero_rate",
        "sku_train_p_zero",
        "sku_train_positive_mean",
        "sku_train_positive_median",
        "sku_train_adi",
        "sku_train_cv2",
        "sku_train_mean_interarrival",
        "sku_last_nonzero_sales",
        "lag_1_proxy",
        "lag_7_proxy",
        "roll_mean_7_proxy",
        "roll_mean_28_proxy",
        "days_since_last_sale",
        "ewm_mean_7",
        "ewm_mean_28",
        "ewm_sale_prob_14",
        "ewm_pos_mean_14",
        "sale_occur_lag_1",
        "sale_occur_lag_7",
        "sale_occur_roll_7",
        "sale_occur_roll_14",
        "sale_occur_roll_28",
        "sale_occur_ewm_7",
        "sale_occur_ewm_28",
    ]
    train_feat[numeric_fill_zero_cols] = train_feat[numeric_fill_zero_cols].fillna(0.0)
    test_feat[numeric_fill_zero_cols] = test_feat[numeric_fill_zero_cols].fillna(0.0)

    drop_internal_cols = ["sku_last_lag1", "sku_last_lag7", "sku_last_roll7", "sku_last_roll28"]
    train_feat = train_feat.drop(columns=[c for c in drop_internal_cols if c in train_feat.columns])
    test_feat = test_feat.drop(columns=[c for c in drop_internal_cols if c in test_feat.columns])

    return train_feat, test_feat


def _split_train_val_by_time(train_feat: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train_feat["date"].dropna().unique()))
    cut_idx = max(1, int(len(unique_dates) * train_ratio))
    cut_idx = min(cut_idx, len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx - 1]
    tr = train_feat[train_feat["date"] <= cut_date].copy()
    va = train_feat[train_feat["date"] > cut_date].copy()
    return tr, va


def _write_prediction_output(pred_df: pd.DataFrame, prediction_output_path: str | Path) -> None:
    key_cols = ["date", "product_family_name", "cluster"]
    prediction_output_path = str(prediction_output_path)
    output_path = Path(prediction_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pred_df.copy()
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        if set(key_cols).issubset(existing_df.columns):
            overlap_cols = [c for c in out_df.columns if c not in key_cols]
            keep_cols = [c for c in existing_df.columns if c not in overlap_cols]
            out_df = existing_df[keep_cols].merge(out_df, on=key_cols, how="outer")

    out_df = out_df.sort_values(key_cols).reset_index(drop=True)
    out_df.to_parquet(output_path, index=False)


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


def _build_positive_day_sample_weight(y_pos: np.ndarray) -> np.ndarray:
    y_pos = np.asarray(y_pos, dtype=float).reshape(-1)
    if len(y_pos) == 0:
        return np.zeros(0, dtype=float)
    median_pos = max(float(np.median(y_pos[y_pos > 0])) if np.any(y_pos > 0) else 0.0, 1.0)
    base = np.sqrt(np.clip(y_pos, 0.0, None) / median_pos + 1.0)
    q90 = float(np.quantile(y_pos, 0.90))
    q98 = float(np.quantile(y_pos, 0.98))
    tail_boost = np.ones_like(base)
    tail_boost = np.where(y_pos >= q90, tail_boost * 1.5, tail_boost)
    tail_boost = np.where(y_pos >= q98, tail_boost * 2.0, tail_boost)
    return np.clip(base * tail_boost, 1.0, 8.0)


def _occurrence_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    pos_mask = yt > 0
    pos_n = int(np.sum(pos_mask))
    if pos_n == 0:
        return 0.0
    tp = int(np.sum((yp > 0) & pos_mask))
    return float(tp / pos_n)


def _fit_two_stage_lgbm(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    occurrence_feature_cols: List[str],
    regression_feature_cols: List[str],
    cls_params: Optional[Dict[str, Any]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
    tau: float = 0.5,
    alpha: float = 1.0,
    cap_value: float = np.inf,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_lightgbm()

    if cls_params is None:
        cls_params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            #"class_weight": "balanced",
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

    p_sale = _fit_occurrence_model(
        train_feat=train_feat,
        test_feat=test_feat,
        occurrence_feature_cols=occurrence_feature_cols,
        cls_params=cls_params,
        random_state=random_state,
    )
    pred_pos = _fit_positive_regression_model(
        train_feat=train_feat,
        test_feat=test_feat,
        regression_feature_cols=regression_feature_cols,
        reg_params=reg_params,
        random_state=random_state,
    )
    pred_final = _apply_two_stage_postprocess(
        p_sale=p_sale,
        pred_pos=pred_pos,
        tau=tau,
        alpha=alpha,
        cap_value=cap_value,
    )
    return p_sale, pred_pos, pred_final


def _fit_occurrence_model(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    occurrence_feature_cols: List[str],
    cls_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> np.ndarray:
    categorical_occ_cols = [c for c in ["sku_code"] if c in occurrence_feature_cols]
    occ_category_levels = {
        col: train_feat[col].astype("category").cat.categories.tolist()
        for col in categorical_occ_cols
    }
    X_train = _prepare_model_matrix(train_feat, occurrence_feature_cols, category_levels=occ_category_levels)
    y_cls = train_feat["is_sale"].astype(int).values
    X_test = _prepare_model_matrix(test_feat, occurrence_feature_cols, category_levels=occ_category_levels)

    if cls_params is None:
        cls_params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    clf = lgb.LGBMClassifier(**cls_params)
    clf.fit(X_train, y_cls, categorical_feature=categorical_occ_cols)
    return clf.predict_proba(X_test)[:, 1]


def _fit_positive_regression_model(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    regression_feature_cols: List[str],
    reg_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> np.ndarray:
    categorical_reg_cols = [c for c in ["sku_code"] if c in regression_feature_cols]
    reg_category_levels = {
        col: train_feat[col].astype("category").cat.categories.tolist()
        for col in categorical_reg_cols
    }
    reg_train = train_feat[train_feat["is_sale"] == 1].copy()
    X_reg = _prepare_model_matrix(reg_train, regression_feature_cols, category_levels=reg_category_levels)
    y_reg_raw = reg_train["y"].astype(float).values
    y_reg = np.log1p(np.clip(y_reg_raw, 0.0, None))
    X_reg_test = _prepare_model_matrix(test_feat, regression_feature_cols, category_levels=reg_category_levels)

    if reg_params is None:
        reg_params = {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    reg = lgb.LGBMRegressor(**reg_params)
    sample_weight = _build_positive_day_sample_weight(y_reg_raw)
    reg.fit(
        X_reg,
        y_reg,
        sample_weight=sample_weight,
        categorical_feature=categorical_reg_cols,
    )
    pred_log = reg.predict(X_reg_test)
    return np.expm1(pred_log).clip(min=0.0)


def _search_two_stage_params(
    train_feat: pd.DataFrame,
    occurrence_feature_cols: List[str],
    regression_feature_cols: List[str],
    eps_mape: float = 1.0,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    _require_lightgbm()
    tr, va = _split_train_val_by_time(train_feat, train_ratio=0.8)
    y_va = va["y"].astype(float).values

    cls_grid = [
        {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": random_state},
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": random_state},
        {"n_estimators": 350, "learning_rate": 0.02, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": random_state},
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
            "learning_rate": 0.08,
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
    tau_grid = [0.30, 0.40, 0.45, 0.55]
    alpha_grid = [0.95, 1.0, 1.1, 1.2]
    cap_q_grid = [0.95, 0.98, 0.995]
    lambda_zero_fp_grid = [1.0, 5.0, 10.0]
    lambda_zero_overshoot_grid = [0.01, 0.05, 0.10]
    lambda_recall_grid = [5.0, 10.0]

    cap_values = {cap_q: _compute_cap_value(tr, cap_q=cap_q) for cap_q in cap_q_grid}
    trial_rows = []

    # Stage 1: tune occurrence only (classifier + tau).
    best_occ_score = np.inf
    best_occ_cfg: Optional[Dict[str, Any]] = None
    best_occ_pred: Optional[np.ndarray] = None

    for cls_params in cls_grid:
        p_sale = _fit_occurrence_model(
            train_feat=tr,
            test_feat=va,
            occurrence_feature_cols=occurrence_feature_cols,
            cls_params=cls_params,
            random_state=random_state,
        )
        for tau in tau_grid:
            pred_occ = (p_sale >= tau).astype(float)
            recall = _occurrence_recall(y_va, pred_occ)
            zero_day_fpr = float(np.mean((pred_occ > 0) & (y_va <= 0))) / max(float(np.mean(y_va <= 0)), 1e-8)
            pred_nonzero_rate = float(np.mean(pred_occ > 0))
            actual_nonzero_rate = float(np.mean(y_va > 0))
            nonzero_rate_gap = float(abs(pred_nonzero_rate - actual_nonzero_rate))
            occ_score = (
                1.0 * nonzero_rate_gap
                + 0.8 * zero_day_fpr
                + 0.4 * (1.0 - recall)
            )
            trial_rows.append(
                {
                    "stage": "occurrence",
                    "score": float(occ_score),
                    "metric_occurrence_recall": recall,
                    "metric_zero_day_fpr": zero_day_fpr,
                    "metric_pred_nonzero_rate": pred_nonzero_rate,
                    "metric_actual_nonzero_rate": actual_nonzero_rate,
                    "metric_nonzero_rate_gap_abs": nonzero_rate_gap,
                    "cls_params": cls_params,
                    "tau": tau,
                }
            )
            if occ_score < best_occ_score:
                best_occ_score = occ_score
                best_occ_cfg = {"cls_params": cls_params, "tau": tau}
                best_occ_pred = p_sale.copy()

    if best_occ_cfg is None or best_occ_pred is None:
        raise RuntimeError("Failed to find valid occurrence config during stage-1 tuning.")

    # Stage 2: fix occurrence config, tune regression + post-process.
    best_score = np.inf
    best_cfg: Optional[Dict[str, Any]] = None

    for reg_params in reg_grid:
        pred_pos = _fit_positive_regression_model(
            train_feat=tr,
            test_feat=va,
            regression_feature_cols=regression_feature_cols,
            reg_params=reg_params,
            random_state=random_state,
        )
        for alpha, cap_q in product(alpha_grid, cap_q_grid):
            cap_value = cap_values[cap_q]
            pred = _apply_two_stage_postprocess(
                p_sale=best_occ_pred,
                pred_pos=pred_pos,
                tau=float(best_occ_cfg["tau"]),
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
            positive_only_mape = float(metric_bundle["POSITIVE_ONLY_MAPE_PCT"])
            zero_day_fpr = float(metric_bundle["ZERO_DAY_FPR"])
            wmape = float(metric_bundle["WMAPE_0_100"])
            recall = _occurrence_recall(y_va, pred)
            pred_nonzero_rate = float(np.mean(pred > 0))
            actual_nonzero_rate = float(np.mean(y_va > 0))
            nonzero_rate_gap = float(abs(pred_nonzero_rate - actual_nonzero_rate))

            for lambda_zero_fp, lambda_zero_overshoot, lambda_recall in product(
                lambda_zero_fp_grid,
                lambda_zero_overshoot_grid,
                lambda_recall_grid,
            ):
                score = (
                    0.6 * wmape
                    + 0.4 * positive_only_mape
                    + lambda_zero_fp * zero_day_fpr
                    + lambda_zero_overshoot * 0.1 * zero_overshoot
                    + lambda_recall * (1.0 - recall) * 100.0
                    + 30.0 * nonzero_rate_gap * 100.0
                )
                row = {
                    "stage": "regression",
                    "score": float(score),
                    "metric_mape": float(metric_bundle["MAPE_0_100"]),
                    "metric_epsilon_mape": float(metric_bundle["EPSILON_MAPE_PCT"]),
                    "metric_wmape": float(metric_bundle["WMAPE_0_100"]),
                    "metric_positive_only_mape": positive_only_mape,
                    "metric_zero_day_fpr": zero_day_fpr,
                    "metric_occurrence_recall": recall,
                    "metric_pred_nonzero_rate": pred_nonzero_rate,
                    "metric_actual_nonzero_rate": actual_nonzero_rate,
                    "metric_nonzero_rate_gap_abs": nonzero_rate_gap,
                    "zero_day_overshoot_pct": float(zero_overshoot),
                    "cls_params": best_occ_cfg["cls_params"],
                    "reg_params": reg_params,
                    "tau": float(best_occ_cfg["tau"]),
                    "alpha": alpha,
                    "cap_q": cap_q,
                    "cap_value": cap_value,
                    "lambda_zero_fp": lambda_zero_fp,
                    "lambda_zero_overshoot": lambda_zero_overshoot,
                    "lambda_recall": lambda_recall,
                }
                trial_rows.append(row)
                if score < best_score:
                    best_score = score
                    best_cfg = {
                        "cls_params": best_occ_cfg["cls_params"],
                        "reg_params": reg_params,
                        "tau": float(best_occ_cfg["tau"]),
                        "alpha": alpha,
                        "cap_q": cap_q,
                        "cap_value": cap_value,
                        "lambda_zero_fp": lambda_zero_fp,
                        "lambda_zero_overshoot": lambda_zero_overshoot,
                        "lambda_recall": lambda_recall,
                        "stage1_occ_score": float(best_occ_score),
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
            "lambda_zero_fp": 10.0,
            "lambda_zero_overshoot": 0.10,
            "lambda_recall": 10.0,
        }
    return best_cfg, trials_df


def run_c0_m2_pipeline(
    train_path: str | Path = "data/forecasting/train_daily.parquet",
    test_path: str | Path = "data/forecasting/test_daily.parquet",
    cluster_id: int = 0,
    feature_cols: Optional[List[str]] = None,
    occurrence_feature_cols: Optional[List[str]] = None,
    regression_feature_cols: Optional[List[str]] = None,
    tune: bool = True,
    eps_mape: float = 1.0,
    random_state: int = 42,
    prediction_output_path: str | Path = "data/forecasting/c0_prediction.parquet",
) -> C0M2Artifacts:
    _require_lightgbm()
    if feature_cols is not None:
        if occurrence_feature_cols is None:
            occurrence_feature_cols = list(feature_cols)
        if regression_feature_cols is None:
            regression_feature_cols = list(feature_cols)
    else:
        if occurrence_feature_cols is None:
            occurrence_feature_cols = list(DEFAULT_OCCURRENCE_FEATURE_COLS)
        if regression_feature_cols is None:
            regression_feature_cols = list(DEFAULT_REGRESSION_FEATURE_COLS)

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
            occurrence_feature_cols=occurrence_feature_cols,
            regression_feature_cols=regression_feature_cols,
            eps_mape=eps_mape,
            random_state=random_state,
        )
        p_sale, pred_pos, pred_final = _fit_two_stage_lgbm(
            train_feat=train_feat,
            test_feat=test_feat,
            occurrence_feature_cols=occurrence_feature_cols,
            regression_feature_cols=regression_feature_cols,
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
            occurrence_feature_cols=occurrence_feature_cols,
            regression_feature_cols=regression_feature_cols,
            random_state=random_state,
        )

    pred_df = test_feat[["date", "product_family_name", "cluster", "y", "is_sale"]].copy()
    pred_df["p_sale"] = p_sale
    pred_df["pred_pos_if_sale"] = pred_pos
    pred_df["pred_two_stage_lgbm"] = pred_final

    sku_actual_df = pred_df.pivot(index="date", columns="product_family_name", values="y").sort_index(axis=1)
    sku_pred_df = pred_df.pivot(index="date", columns="product_family_name", values="pred_two_stage_lgbm").sort_index(axis=1)

    _write_prediction_output(pred_df=pred_df, prediction_output_path=prediction_output_path)
    prediction_output_path = str(prediction_output_path)

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
