from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"


def _ensure_period_column(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    out = df.copy()
    if "period" in out.columns:
        return out
    dates = np.array(sorted(out["date"].dropna().unique()))
    chunks = np.array_split(dates, n_periods)
    mapper: Dict[pd.Timestamp, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        for d in chunk:
            mapper[pd.Timestamp(d)] = f"P{i}"
    out["period"] = out["date"].map(mapper)
    return out


def _period_sort_key(period_label: str) -> tuple[int, str]:
    if isinstance(period_label, str) and period_label.startswith("P"):
        suffix = period_label[1:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, str(period_label))


def _infer_method_cols(
    pred_df: pd.DataFrame,
    baseline_col: Optional[str] = None,
    model_col: str = "pred_two_stage",
) -> Dict[str, str]:
    if baseline_col is None:
        for c in ["pred_snaive7", "pred_adida", "pred_sba", "pred_tsb"]:
            if c in pred_df.columns and c != model_col:
                baseline_col = c
                break
    if baseline_col is None:
        raise ValueError("Cannot infer baseline column. Please pass baseline_col explicitly.")
    if model_col not in pred_df.columns:
        raise ValueError(f"Model column not found: {model_col}")
    return {"baseline": baseline_col, "model": model_col}


def _method_display_name(role: str, col_name: str) -> str:
    if role == "baseline":
        if col_name == "pred_snaive7":
            return "Baseline Forecast (Seasonal Naive 7)"
        if col_name == "pred_adida":
            return "Baseline Forecast (ADIDA)"
        if col_name == "pred_sba":
            return "Baseline Forecast (SBA)"
        if col_name == "pred_tsb":
            return "Baseline Forecast (TSB)"
        return "Baseline Forecast"
    if role == "model":
        if col_name == "pred_two_stage":
            return "Model Forecast (Two-stage HGB)"
        return "Model Forecast"
    return col_name


def _agg_daily(pred_df: pd.DataFrame, method_cols: Dict[str, str]) -> pd.DataFrame:
    cols = ["y"] + list(method_cols.values())
    agg = pred_df.groupby("date", as_index=False)[cols].sum().sort_values("date")
    return agg


def _ape_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> np.ndarray:
    denom = np.maximum(np.abs(y_true), eps)
    return np.abs(y_true - y_pred) / denom * 100.0


def _wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    num = np.sum(np.abs(y_true - y_pred))
    den = max(float(np.sum(np.abs(y_true))), eps)
    return float(100.0 * num / den)


def _save_fig(fig: plt.Figure, file_stem: str, save_svg: bool, out_dir: Path) -> None:
    if not save_svg:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.svg", format="svg", bbox_inches="tight")


def plot_cluster_aggregate_lines(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> None:
    _set_style()
    agg = _agg_daily(pred_df, method_cols)
    baseline_col = method_cols["baseline"]
    model_col = method_cols["model"]

    # Residual-based uncertainty band:
    # sigma_t uses expanding std of past residuals (shifted by 1 day) to avoid look-ahead.
    # Fallback to full residual std for early points.
    z95 = 1.96
    baseline_resid = agg["y"] - agg[baseline_col]
    model_resid = agg["y"] - agg[model_col]

    baseline_sigma = baseline_resid.expanding(min_periods=7).std().shift(1)
    model_sigma = model_resid.expanding(min_periods=7).std().shift(1)
    baseline_sigma = baseline_sigma.fillna(float(baseline_resid.std(ddof=0)))
    model_sigma = model_sigma.fillna(float(model_resid.std(ddof=0)))

    baseline_low = np.clip(agg[baseline_col] - z95 * baseline_sigma, 0.0, None)
    baseline_up = agg[baseline_col] + z95 * baseline_sigma
    model_low = np.clip(agg[model_col] - z95 * model_sigma, 0.0, None)
    model_up = agg[model_col] + z95 * model_sigma

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg["date"], agg["y"], label="Actual", linewidth=2.2, color="#1f77b4")
    ax.plot(
        agg["date"],
        agg[baseline_col],
        label=_method_display_name("baseline", baseline_col),
        linewidth=1.8,
        color="#ff7f0e",
    )
    ax.fill_between(
        agg["date"],
        baseline_low.values,
        baseline_up.values,
        color="#ff7f0e",
        alpha=0.16,
        label="Baseline 95% interval",
    )
    ax.plot(
        agg["date"],
        agg[model_col],
        label=_method_display_name("model", model_col),
        linewidth=1.8,
        color="#2ca02c",
    )
    ax.fill_between(
        agg["date"],
        model_low.values,
        model_up.values,
        color="#2ca02c",
        alpha=0.16,
        label="Model 95% interval",
    )
    ax.set_title(f"{cluster_label} Test Daily Aggregate: Actual vs Prediction with 95% Intervals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cluster daily total_sales")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "01_cluster_aggregate_lines", save_svg=save_svg, out_dir=out_dir)
    plt.show()


def plot_residual_time_series(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    rolling_window: int = 7,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> pd.DataFrame:
    _set_style()
    agg = _agg_daily(pred_df, method_cols)
    baseline_col = method_cols["baseline"]
    model_col = method_cols["model"]

    out = agg[["date"]].copy()
    out["baseline_residual"] = agg["y"] - agg[baseline_col]
    out["model_residual"] = agg["y"] - agg[model_col]
    out["baseline_resid_roll_mean"] = out["baseline_residual"].rolling(rolling_window, min_periods=rolling_window).mean()
    out["model_resid_roll_mean"] = out["model_residual"].rolling(rolling_window, min_periods=rolling_window).mean()

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.plot(
        out["date"],
        out["baseline_residual"],
        color="#ff7f0e",
        alpha=0.35,
        linewidth=1.2,
        label=f"{_method_display_name('baseline', baseline_col)} residual",
    )
    ax.plot(
        out["date"],
        out["model_residual"],
        color="#2ca02c",
        alpha=0.35,
        linewidth=1.2,
        label=f"{_method_display_name('model', model_col)} residual",
    )
    ax.plot(
        out["date"],
        out["baseline_resid_roll_mean"],
        color="#ff7f0e",
        linewidth=2.0,
        label=f"Baseline residual {rolling_window}d mean",
    )
    ax.plot(
        out["date"],
        out["model_resid_roll_mean"],
        color="#2ca02c",
        linewidth=2.0,
        label=f"Model residual {rolling_window}d mean",
    )
    ax.set_title(f"{cluster_label} Residual Time Series (Actual - Prediction)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "06_residual_time_series", save_svg=save_svg, out_dir=out_dir)
    plt.show()
    return out


def plot_rolling_error_trend(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    window: int = 7,
    rolling_metric: str = "wmape",
    eps: float = 1.0,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> None:
    _set_style()
    agg = _agg_daily(pred_df, method_cols)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    for name, col in method_cols.items():
        if rolling_metric.lower() == "wmape":
            num = np.abs(agg["y"] - agg[col]).rolling(window, min_periods=window).sum()
            den = agg["y"].abs().rolling(window, min_periods=window).sum().clip(lower=eps)
            series = 100.0 * num / den
            ylabel = f"Rolling {window}-day WMAPE (%)"
        else:
            ape = _ape_eps(agg["y"].values, agg[col].values, eps=eps)
            series = pd.Series(ape, index=agg.index).rolling(window, min_periods=window).mean()
            ylabel = f"Rolling {window}-day Epsilon-MAPE (%)"
        ax.plot(agg["date"], series, label=_method_display_name(name, col), linewidth=2)

    ax.set_title(f"{cluster_label} Temporal Stability ({rolling_metric.upper()})")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(title="Method")
    fig.tight_layout()
    _save_fig(fig, "02_rolling_error_trend", save_svg=save_svg, out_dir=out_dir)
    plt.show()


def plot_occurrence_confusion_by_period(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> pd.DataFrame:
    _set_style()
    df = _ensure_period_column(pred_df)
    y_true_sale = (df["y"].values > 0).astype(int)
    periods = sorted(df["period"].dropna().unique(), key=_period_sort_key)

    rows = []
    for method_name, col in method_cols.items():
        y_pred_sale = (df[col].values > 0).astype(int)
        for p in periods:
            m = df["period"] == p
            yt = y_true_sale[m.values]
            yp = y_pred_sale[m.values]
            rows.append(
                {
                    "method": method_name,
                    "period": p,
                    "TP": int(np.sum((yt == 1) & (yp == 1))),
                    "FP": int(np.sum((yt == 0) & (yp == 1))),
                    "FN": int(np.sum((yt == 1) & (yp == 0))),
                    "TN": int(np.sum((yt == 0) & (yp == 0))),
                }
            )
    conf_df = pd.DataFrame(rows)
    total = conf_df[["TP", "FP", "FN", "TN"]].sum(axis=1).replace(0, np.nan)
    conf_df["TP_pct"] = (conf_df["TP"] / total * 100.0).fillna(0.0)
    conf_df["FP_pct"] = (conf_df["FP"] / total * 100.0).fillna(0.0)
    conf_df["FN_pct"] = (conf_df["FN"] / total * 100.0).fillna(0.0)
    conf_df["TN_pct"] = (conf_df["TN"] / total * 100.0).fillna(0.0)

    fig, axes = plt.subplots(1, len(method_cols), figsize=(4 * len(method_cols), 4.4), sharey=True)
    if len(method_cols) == 1:
        axes = [axes]

    colors = {"TP": "#2ca02c", "FP": "#d62728", "FN": "#ff7f0e", "TN": "#1f77b4"}
    for ax, (method_name, _) in zip(axes, method_cols.items()):
        sub = conf_df[conf_df["method"] == method_name].set_index("period").loc[periods]
        sub_pct = sub[["TP_pct", "FP_pct", "FN_pct", "TN_pct"]].copy()
        sub_pct.columns = ["TP", "FP", "FN", "TN"]
        bottom = np.zeros(len(sub_pct))
        for k in ["TP", "FP", "FN", "TN"]:
            vals = sub_pct[k].values
            bars = ax.bar(sub_pct.index, vals, bottom=bottom, label=k, color=colors[k], alpha=0.9)
            for bar, v, b in zip(bars, vals, bottom):
                if v >= 4.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        b + v / 2.0,
                        f"{v:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
            bottom = bottom + vals
        ax.set_title(f"{cluster_label} {_method_display_name(method_name, method_cols[method_name])}")
        ax.set_xlabel("Test period")
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "03_occurrence_confusion_by_period", save_svg=save_svg, out_dir=out_dir)
    plt.show()
    return conf_df


def plot_interval_width_over_time(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    rolling_window: int = 7,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> pd.DataFrame:
    _set_style()
    agg = _agg_daily(pred_df, method_cols)
    baseline_col = method_cols["baseline"]
    model_col = method_cols["model"]
    z95 = 1.96

    baseline_resid = agg["y"] - agg[baseline_col]
    model_resid = agg["y"] - agg[model_col]

    baseline_sigma = baseline_resid.expanding(min_periods=7).std().shift(1)
    model_sigma = model_resid.expanding(min_periods=7).std().shift(1)
    baseline_sigma = baseline_sigma.fillna(float(baseline_resid.std(ddof=0)))
    model_sigma = model_sigma.fillna(float(model_resid.std(ddof=0)))

    # Width of 95% interval = upper - lower = 2 * 1.96 * sigma
    out = agg[["date"]].copy()
    out["baseline_ci95_width"] = 2.0 * z95 * baseline_sigma
    out["model_ci95_width"] = 2.0 * z95 * model_sigma
    out["baseline_ci95_width_roll"] = out["baseline_ci95_width"].rolling(
        rolling_window, min_periods=rolling_window
    ).mean()
    out["model_ci95_width_roll"] = out["model_ci95_width"].rolling(
        rolling_window, min_periods=rolling_window
    ).mean()

    fig, ax = plt.subplots(figsize=(14, 4.0))
    ax.plot(
        out["date"],
        out["baseline_ci95_width"],
        color="#ff7f0e",
        alpha=0.35,
        linewidth=1.2,
        label="Baseline 95% CI width",
    )
    ax.plot(
        out["date"],
        out["model_ci95_width"],
        color="#2ca02c",
        alpha=0.35,
        linewidth=1.2,
        label="Model 95% CI width",
    )
    ax.plot(
        out["date"],
        out["baseline_ci95_width_roll"],
        color="#ff7f0e",
        linewidth=2.0,
        label=f"Baseline CI width {rolling_window}d mean",
    )
    ax.plot(
        out["date"],
        out["model_ci95_width_roll"],
        color="#2ca02c",
        linewidth=2.0,
        label=f"Model CI width {rolling_window}d mean",
    )
    ax.set_title(f"{cluster_label} 95% Prediction Interval Width Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Interval width")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "06b_interval_width_over_time", save_svg=save_svg, out_dir=out_dir)
    plt.show()
    return out


def plot_positive_day_scatter(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> None:
    _set_style()
    agg = _agg_daily(pred_df, method_cols)
    agg = agg[agg["y"] > 0].copy()
    if agg.empty:
        return

    fig, axes = plt.subplots(1, len(method_cols), figsize=(7 * len(method_cols), 5), sharex=False, sharey=False)
    if len(method_cols) == 1:
        axes = [axes]

    for ax, (method_name, col) in zip(axes, method_cols.items()):
        x = agg["y"].values
        y = agg[col].values
        lim = float(max(np.max(x), np.max(y)) * 1.05)
        ax.scatter(x, y, alpha=0.5, s=20, color="#1f77b4")
        ax.plot([0, lim], [0, lim], linestyle="--", color="#d62728", linewidth=1.8)
        ax.set_title(f"{cluster_label} Positive-demand scatter: {method_name}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    fig.tight_layout()
    _save_fig(fig, "04_positive_day_scatter", save_svg=save_svg, out_dir=out_dir)
    plt.show()


def plot_days_since_last_sale_shift(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    cluster_label: str,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> Dict[str, float]:
    _set_style()
    tr = train_feat["days_since_last_sale"].astype(float).dropna().values
    te = test_feat["days_since_last_sale"].astype(float).dropna().values
    if len(tr) == 0 or len(te) == 0:
        return {}

    cap = float(np.quantile(np.concatenate([tr, te]), 0.99))
    tr_plot = np.clip(tr, 0, cap)
    te_plot = np.clip(te, 0, cap)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.kdeplot(
        x=tr_plot,
        bw_adjust=1.1,
        cut=0,
        fill=True,
        alpha=0.30,
        linewidth=2.0,
        color="#1f77b4",
        label="Train",
        ax=ax,
    )
    sns.kdeplot(
        x=te_plot,
        bw_adjust=1.1,
        cut=0,
        fill=True,
        alpha=0.30,
        linewidth=2.0,
        color="#ff7f0e",
        label="Test",
        ax=ax,
    )
    ax.set_title(f"{cluster_label} How Time Since Last Sale Changed (Train vs Test, clipped at P99)")
    ax.set_xlabel("days_since_last_sale")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "05_days_since_last_sale_shift", save_svg=save_svg, out_dir=out_dir)
    plt.show()

    stats = {
        "train_mean": float(np.mean(tr)),
        "test_mean": float(np.mean(te)),
        "train_p90": float(np.quantile(tr, 0.90)),
        "test_p90": float(np.quantile(te, 0.90)),
        "train_999_share": float(np.mean(tr >= 999)),
        "test_999_share": float(np.mean(te >= 999)),
    }
    return stats


def plot_occurrence_calibration(
    pred_df: pd.DataFrame,
    cluster_label: str,
    n_bins: int = 10,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> Optional[pd.DataFrame]:
    _set_style()
    if "p_sale" not in pred_df.columns:
        return None

    df = pred_df.copy()
    df = df[(df["p_sale"] >= 0) & (df["p_sale"] <= 1)].copy()
    if df.empty:
        return None
    df["actual_sale"] = (df["y"] > 0).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["p_bin"] = pd.cut(df["p_sale"], bins=bins, include_lowest=True, duplicates="drop")
    cal = (
        df.groupby("p_bin", observed=False)
        .agg(
            p_mean=("p_sale", "mean"),
            sale_rate=("actual_sale", "mean"),
            count=("actual_sale", "size"),
        )
        .dropna()
        .reset_index()
    )
    if cal.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Ideal")
    axes[0].plot(cal["p_mean"], cal["sale_rate"], marker="o", linewidth=2, color="#1f77b4", label="Observed")
    axes[0].set_title(f"{cluster_label} Occurrence Calibration")
    axes[0].set_xlabel("Predicted probability (bin mean)")
    axes[0].set_ylabel("Empirical sale rate")
    axes[0].legend()

    axes[1].bar(range(len(cal)), cal["count"], color="#ff7f0e", alpha=0.8)
    axes[1].set_title(f"{cluster_label} Calibration Bin Counts")
    axes[1].set_xlabel("Probability bin index")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(range(len(cal)))
    axes[1].set_xticklabels([str(i + 1) for i in range(len(cal))])

    fig.tight_layout()
    _save_fig(fig, "06_occurrence_calibration", save_svg=save_svg, out_dir=out_dir)
    plt.show()
    return cal


def plot_error_decomposition_by_state(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    cluster_label: str,
    eps: float = 1.0,
    save_svg: bool = False,
    out_dir: Path = Path("images/c1_results"),
) -> pd.DataFrame:
    _set_style()
    rows = []
    y = pred_df["y"].values.astype(float)

    for method_name, col in method_cols.items():
        pred = pred_df[col].values.astype(float)
        for state_name, mask in [("y=0", y == 0), ("y>0", y > 0)]:
            if int(np.sum(mask)) == 0:
                continue
            y_s = y[mask]
            p_s = pred[mask]
            ape_eps = _ape_eps(y_s, p_s, eps=eps)
            rows.append(
                {
                    "method": method_name,
                    "state": state_name,
                    "count": int(np.sum(mask)),
                    "epsilon_mape_pct": float(np.mean(ape_eps)),
                    "wmape_pct": _wmape(y_s, p_s, eps=eps),
                    "mae": float(np.mean(np.abs(y_s - p_s))),
                }
            )
    out = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    sns.barplot(data=out, x="state", y="epsilon_mape_pct", hue="method", ax=axes[0])
    axes[0].set_title(f"{cluster_label} Error by demand state (Epsilon-MAPE)")
    axes[0].set_xlabel("Demand state")
    axes[0].set_ylabel("Epsilon-MAPE (%)")

    sns.barplot(data=out, x="state", y="mae", hue="method", ax=axes[1])
    axes[1].set_title(f"{cluster_label} Error by demand state (MAE)")
    axes[1].set_xlabel("Demand state")
    axes[1].set_ylabel("MAE")

    if axes[0].legend_ is not None:
        axes[0].legend_.set_title("Method")
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    fig.tight_layout()
    _save_fig(fig, "07_error_decomposition_by_state", save_svg=save_svg, out_dir=out_dir)
    plt.show()
    return out


def build_cluster_metric_tables(
    pred_df: pd.DataFrame,
    method_cols: Dict[str, str],
    eps: float = 1.0,
) -> pd.DataFrame:
    rows = []
    y = pred_df["y"].values.astype(float)
    actual_nonzero_rate = float(np.mean(y > 0))

    for method_name, col in method_cols.items():
        pred = pred_df[col].values.astype(float)
        ape_eps = _ape_eps(y, pred, eps=eps)
        signed_pct = (pred - y) / np.maximum(np.abs(y), eps) * 100.0
        rows.append(
            {
                "method": method_name,
                "wmape_pct": _wmape(y, pred, eps=eps),
                "epsilon_mape_pct": float(np.mean(ape_eps)),
                "cap_mape_0_100": float(np.mean(np.clip(ape_eps, 0.0, 100.0))),
                "tail_ape_p90": float(np.quantile(ape_eps, 0.90)),
                "tail_ape_p95": float(np.quantile(ape_eps, 0.95)),
                "pred_nonzero_rate": float(np.mean(pred > 0)),
                "actual_nonzero_rate": actual_nonzero_rate,
                "nonzero_rate_gap": float(np.mean(pred > 0) - actual_nonzero_rate),
                "signed_pct_error_mean": float(np.mean(signed_pct)),
                "signed_abs_bias_ratio": float(np.sum(pred - y) / max(np.sum(np.abs(y)), eps)),
            }
        )
    return pd.DataFrame(rows).sort_values("wmape_pct").reset_index(drop=True)


def run_cluster_analysis(
    art: Any,
    cluster_tag: str = "c1",
    baseline_col: Optional[str] = None,
    model_col: str = "pred_two_stage",
    rolling_metric: str = "wmape",
    rolling_window: int = 7,
    eps: float = 1.0,
    n_periods: int = 4,
    save_svg: bool = False,
    output_root: str = "images",
) -> Dict[str, pd.DataFrame]:
    """
    Run a full post-forecast analysis suite for C1/C3 artifacts.

    Parameters
    ----------
    art : Any
        Output artifact from run_c1_pipeline or run_c3_pipeline.
    cluster_tag : str
        Name used in chart titles and output directory, e.g. "c1" or "c3".
    baseline_col : str | None
        Baseline prediction column. If None, infer from pred_df.
    model_col : str
        Main model prediction column.
    rolling_metric : str
        "wmape" or "mape" for rolling error plot.
    rolling_window : int
        Rolling window size in days.
    eps : float
        Epsilon for safe percentage calculations.
    n_periods : int
        Number of periods if pred_df has no "period" column.
    save_svg : bool
        If True, save figures as svg files.
    output_root : str
        Root output folder. Files are saved in images/{cluster_tag}_results.
    """
    pred_df = art.pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    pred_df = _ensure_period_column(pred_df, n_periods=n_periods)

    method_cols = _infer_method_cols(pred_df, baseline_col=baseline_col, model_col=model_col)
    out_dir = Path(output_root) / f"{cluster_tag}_results"
    cluster_label = cluster_tag.upper()

    plot_cluster_aggregate_lines(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    plot_rolling_error_trend(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        window=rolling_window,
        rolling_metric=rolling_metric,
        eps=eps,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    confusion_by_period = plot_occurrence_confusion_by_period(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    plot_positive_day_scatter(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    shift_stats = plot_days_since_last_sale_shift(
        train_feat=art.train_feat,
        test_feat=art.test_feat,
        cluster_label=cluster_label,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    residual_ts = plot_residual_time_series(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        rolling_window=rolling_window,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    interval_width_ts = plot_interval_width_over_time(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        rolling_window=rolling_window,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    error_decomposition = plot_error_decomposition_by_state(
        pred_df=pred_df,
        method_cols=method_cols,
        cluster_label=cluster_label,
        eps=eps,
        save_svg=save_svg,
        out_dir=out_dir,
    )
    metric_table = build_cluster_metric_tables(
        pred_df=pred_df,
        method_cols=method_cols,
        eps=eps,
    )

    return {
        "metric_table": metric_table,
        "confusion_by_period": confusion_by_period,
        "error_decomposition": error_decomposition,
        "residual_time_series": residual_ts,
        "interval_width_time_series": interval_width_ts,
        "distribution_shift_stats": pd.DataFrame([shift_stats]) if shift_stats else pd.DataFrame(),
        "method_cols": pd.DataFrame([method_cols]),
    }
