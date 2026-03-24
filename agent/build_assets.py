from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from agent import PROJECT_ROOT
from agent.config import (
    CLUSTER_MODEL_CONFIG,
    FORECAST_SCHEMA,
    FUTURE_FORECAST_HORIZON_DAYS,
    MANIFEST_PATH,
    REGISTRY_PATH,
    ensure_agent_dirs,
)
from agent.registry import (
    build_product_metadata_frame,
    build_registry_frame,
    product_key_for,
    serialize_registry_frame,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ClusterBuildResult:
    cluster_id: int
    artifact_path: str
    rows: int
    products: int
    generated_at: str
    model_id: str
    model_name: str


def _relative_project_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


C0_FIXED_BEST_CONFIG: Dict[str, Any] = {
    "cls_params": {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "reg_params": {
        "n_estimators": 450,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    "tau": 0.25,
    "alpha": 0.9,
    "cap_q": 0.98,
}

C2_FIXED_BEST_CONFIG: Dict[str, Any] = {
    "reg_params": {
        "objective": "poisson",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 80,
        "subsample": 1.0,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    },
    "cap_q": 0.99,
    "threshold_mode": "sku",
    "tau_quantile": 0.05,
    "recent_weight_max": 2.0,
}


def validate_canonical_forecast_frame(df: pd.DataFrame) -> None:
    missing = [col for col in FORECAST_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(f"Canonical forecast frame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Canonical forecast frame is empty.")
    if df["forecast_value"].isna().all():
        raise ValueError("Canonical forecast frame contains only missing forecast_value rows.")


def normalize_prediction_frame(
    pred_df: pd.DataFrame,
    cluster_id: int,
    generated_at: Optional[str] = None,
    product_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    config = CLUSTER_MODEL_CONFIG[cluster_id]
    if pred_df.empty:
        raise ValueError(f"Prediction frame for cluster {cluster_id} is empty.")

    frame = pred_df.copy()
    if "sku_id" in frame.columns and "product_family_name" not in frame.columns:
        frame = frame.rename(columns={"sku_id": "product_family_name"})
    if "date" not in frame.columns:
        raise ValueError("Prediction frame must include a `date` column.")
    if "product_family_name" not in frame.columns:
        raise ValueError("Prediction frame must include `product_family_name` or `sku_id`.")
    if config.prediction_column not in frame.columns:
        raise ValueError(
            f"Prediction frame for cluster {cluster_id} is missing required model column "
            f"`{config.prediction_column}`."
        )

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    out = pd.DataFrame(
        {
            "product_family_name": frame["product_family_name"].astype("string").fillna("").str.strip(),
            "cluster": int(cluster_id),
            "model_id": config.model_id,
            "model_name": config.model_name,
            "forecast_date": pd.to_datetime(frame["date"], errors="coerce"),
            "forecast_value": pd.to_numeric(frame[config.prediction_column], errors="coerce"),
            "generated_at": generated_at,
        }
    )

    if config.probability_column and config.probability_column in frame.columns:
        out["p_sale"] = pd.to_numeric(frame[config.probability_column], errors="coerce")
    else:
        out["p_sale"] = pd.NA

    out["product_key"] = out["product_family_name"].map(lambda value: product_key_for(cluster_id, str(value)))
    out["product_id"] = pd.NA
    if "y" in frame.columns:
        out["actual_value"] = pd.to_numeric(frame["y"], errors="coerce")

    if product_metadata is not None and not product_metadata.empty:
        id_map = (
            product_metadata[["product_family_name", "product_id"]]
            .drop_duplicates(subset=["product_family_name"])
            .set_index("product_family_name")["product_id"]
        )
        out["product_id"] = out["product_family_name"].map(id_map).fillna(pd.NA)

    extra_cols = [col for col in out.columns if col not in FORECAST_SCHEMA]
    ordered_cols = FORECAST_SCHEMA + extra_cols
    out = out[ordered_cols].sort_values(["product_family_name", "forecast_date"]).reset_index(drop=True)
    validate_canonical_forecast_frame(out)
    return out


def build_cluster_asset(
    cluster_id: int,
    product_metadata: Optional[pd.DataFrame] = None,
    pred_df: Optional[pd.DataFrame] = None,
    mode: str = "evaluation",
) -> ClusterBuildResult:
    ensure_agent_dirs()
    config = CLUSTER_MODEL_CONFIG[cluster_id]
    if mode == "evaluation" and not config.enabled:
        raise ValueError(f"Cluster {cluster_id} is not enabled in the current agent build.")
    generated_at = datetime.now(timezone.utc).isoformat()

    if pred_df is None:
        if mode == "evaluation":
            pred_df = _load_cluster_prediction_source(cluster_id=cluster_id)
        elif mode == "future":
            pred_df = _build_future_prediction_frame(cluster_id=cluster_id, horizon_days=FUTURE_FORECAST_HORIZON_DAYS)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    canonical = normalize_prediction_frame(
        pred_df=pred_df,
        cluster_id=cluster_id,
        generated_at=generated_at,
        product_metadata=product_metadata,
    )
    artifact_path = config.artifact_path if mode == "evaluation" else config.future_artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(artifact_path, index=False)

    reloaded = pd.read_csv(artifact_path)
    validate_canonical_forecast_frame(reloaded)

    return ClusterBuildResult(
        cluster_id=cluster_id,
        artifact_path=_relative_project_path(artifact_path),
        rows=int(len(canonical)),
        products=int(canonical["product_key"].nunique()),
        generated_at=generated_at,
        model_id=config.model_id,
        model_name=config.model_name,
    )


def build_all_assets() -> Dict[str, object]:
    return build_assets(include_future=False)


def build_assets(
    include_future: bool = False,
    cluster_ids: Optional[Iterable[int]] = None,
    evaluation_only: bool = False,
) -> Dict[str, object]:
    ensure_agent_dirs()
    product_metadata = build_product_metadata_frame()
    selected_clusters = sorted(
        int(cluster_id) for cluster_id in (cluster_ids if cluster_ids is not None else CLUSTER_MODEL_CONFIG.keys())
    )
    selected_cluster_set = set(selected_clusters)

    results: List[ClusterBuildResult] = []
    future_results: List[ClusterBuildResult] = []
    registry_frames: List[pd.DataFrame] = []
    errors: List[str] = []
    skipped_clusters: List[Dict[str, object]] = []
    available_artifact_paths: Dict[int, Path] = {}

    for cluster_id in sorted(CLUSTER_MODEL_CONFIG):
        config = CLUSTER_MODEL_CONFIG[cluster_id]
        if cluster_id not in selected_cluster_set:
            if config.artifact_path.exists():
                registry_frames.append(pd.read_csv(config.artifact_path))
                available_artifact_paths.setdefault(cluster_id, config.artifact_path)
            if include_future and config.future_artifact_path.exists():
                registry_frames.append(pd.read_csv(config.future_artifact_path))
                available_artifact_paths.setdefault(cluster_id, config.future_artifact_path)
            continue
        evaluation_built = False
        if not config.enabled or config.source_prediction_path is None:
            skipped_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "reason": "Evaluation artifact is unavailable because no readable prediction parquet is configured.",
                }
            )
        else:
            try:
                result = build_cluster_asset(cluster_id=cluster_id, product_metadata=product_metadata, mode="evaluation")
                results.append(result)
                registry_frames.append(pd.read_csv(config.artifact_path))
                available_artifact_paths[cluster_id] = config.artifact_path
                evaluation_built = True
            except Exception as exc:
                errors.append(f"cluster {cluster_id} evaluation: {exc}")

        if include_future and not evaluation_only:
            try:
                future_result = build_cluster_asset(cluster_id=cluster_id, product_metadata=product_metadata, mode="future")
                future_results.append(future_result)
                registry_frames.append(pd.read_csv(config.future_artifact_path))
                available_artifact_paths.setdefault(cluster_id, config.future_artifact_path)
            except Exception as exc:
                errors.append(f"cluster {cluster_id} future: {exc}")
        elif evaluation_built:
            available_artifact_paths.setdefault(cluster_id, config.artifact_path)

    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"Asset build failed:\n{joined}")
    if not registry_frames:
        raise RuntimeError("Asset build failed: no enabled cluster artifacts were produced.")

    combined = pd.concat(registry_frames, ignore_index=True)
    registry = build_registry_frame(
        canonical_forecasts=combined,
        product_metadata=product_metadata,
        artifact_paths=available_artifact_paths,
    )
    serialize_registry_frame(registry).to_csv(REGISTRY_PATH, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": _relative_project_path(REGISTRY_PATH),
        "clusters": [asdict(result) for result in results],
        "future_clusters": [asdict(result) for result in future_results],
        "enabled_clusters": sorted(available_artifact_paths),
        "requested_clusters": selected_clusters,
        "skipped_clusters": skipped_clusters,
        "products": int(registry["product_key"].nunique()),
        "rows": int(combined.shape[0]),
        "future_horizon_days": FUTURE_FORECAST_HORIZON_DAYS if include_future else None,
        "future_assets_built": bool(include_future),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _load_cluster_prediction_source(cluster_id: int) -> pd.DataFrame:
    config = CLUSTER_MODEL_CONFIG[cluster_id]
    if not config.enabled or config.source_prediction_path is None:
        raise ValueError(f"Cluster {cluster_id} does not have an enabled prediction source.")
    return _read_table(config.source_prediction_path)


def _build_future_prediction_frame(cluster_id: int, horizon_days: int) -> pd.DataFrame:
    _configure_future_runtime()
    builder_map = {
        0: _build_c0_future_prediction_frame,
        1: _build_c1_future_prediction_frame,
        2: _build_c2_future_prediction_frame,
        3: _build_c3_future_prediction_frame,
    }
    if cluster_id not in builder_map:
        raise ValueError(f"Future mode is not implemented for cluster {cluster_id}.")
    try:
        return builder_map[cluster_id](horizon_days=horizon_days)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"Future build for cluster {cluster_id} requires dependencies from `environment.yml` "
            "in the active Python environment."
        ) from exc


def _load_train_test_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _read_table(PROJECT_ROOT / "data" / "forecasting" / "train_daily.parquet")
    test = _read_table(PROJECT_ROOT / "data" / "forecasting" / "test_daily.parquet")
    for df in (train, test):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["product_family_name"] = df["product_family_name"].astype("string").str.strip()
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
        df["total_sales"] = pd.to_numeric(df["total_sales"], errors="coerce").fillna(0.0)
    return train, test


def _future_grid(
    history_df: pd.DataFrame,
    product_col: str,
    cluster_col: str,
    value_col: str,
    horizon_days: int,
) -> pd.DataFrame:
    sku_map = history_df[[product_col, cluster_col]].drop_duplicates().reset_index(drop=True)
    last_date = pd.to_datetime(history_df["date"]).max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")
    future_grid = sku_map.assign(_tmp=1).merge(
        pd.DataFrame({"date": future_dates, "_tmp": 1}),
        on="_tmp",
        how="inner",
    )
    future_grid = future_grid.drop(columns="_tmp")
    future_grid[value_col] = 0.0
    return future_grid[[product_col, "date", cluster_col, value_col]]


def _build_c0_future_prediction_frame(horizon_days: int) -> pd.DataFrame:
    from c0_m2 import (
        DEFAULT_OCCURRENCE_FEATURE_COLS,
        DEFAULT_REGRESSION_FEATURE_COLS,
        _build_features,
        _build_zero_filled_panel,
        _compute_cap_value,
        _fit_two_stage_lgbm,
    )

    train_daily, test_daily = _load_train_test_daily()
    combined_raw = pd.concat([train_daily, test_daily], ignore_index=True)
    combined_raw = combined_raw[combined_raw["cluster"] == 0].copy()
    if combined_raw.empty:
        raise ValueError("No rows found for cluster 0 in combined train/test history.")

    placeholder = combined_raw.head(1).copy()
    train_panel, _ = _build_zero_filled_panel(combined_raw, placeholder)
    future_panel = _future_grid(
        history_df=train_panel,
        product_col="product_family_name",
        cluster_col="cluster",
        value_col="total_sales",
        horizon_days=horizon_days,
    )
    future_panel = future_panel.sort_values(["product_family_name", "date"]).reset_index(drop=True)
    train_feat, future_feat = _build_features(train_panel, future_panel)
    cap_value = _compute_cap_value(train_feat, cap_q=float(C0_FIXED_BEST_CONFIG["cap_q"]))
    p_sale, pred_pos, pred_final = _fit_two_stage_lgbm(
        train_feat=train_feat,
        test_feat=future_feat,
        occurrence_feature_cols=list(DEFAULT_OCCURRENCE_FEATURE_COLS),
        regression_feature_cols=list(DEFAULT_REGRESSION_FEATURE_COLS),
        cls_params=dict(C0_FIXED_BEST_CONFIG["cls_params"]),
        reg_params=dict(C0_FIXED_BEST_CONFIG["reg_params"]),
        tau=float(C0_FIXED_BEST_CONFIG["tau"]),
        alpha=float(C0_FIXED_BEST_CONFIG["alpha"]),
        cap_value=float(cap_value),
        random_state=42,
    )

    pred_df = future_feat[["date", "product_family_name", "cluster"]].copy()
    pred_df["y"] = pd.NA
    pred_df["is_sale"] = pd.NA
    pred_df["p_sale"] = np.asarray(p_sale, dtype=float)
    pred_df["pred_pos_if_sale"] = np.asarray(pred_pos, dtype=float)
    pred_df["pred_two_stage_lgbm"] = np.asarray(pred_final, dtype=float)
    return pred_df.sort_values(["product_family_name", "date"]).reset_index(drop=True)


def _build_c1_future_prediction_frame(horizon_days: int) -> pd.DataFrame:
    from c1_forecasting import (
        DEFAULT_ZINB_CLIP_MAX_CAP,
        DEFAULT_ZINB_CLIP_Q,
        DEFAULT_ZINB_THRESHOLD,
        _fit_zinb_with_config,
        _recursive_zinb_forecast,
    )
    from c1_forecasting_new import _build_train_features, _build_zero_filled_panel

    train_daily, test_daily = _load_train_test_daily()
    combined_raw = pd.concat([train_daily, test_daily], ignore_index=True)
    combined_raw = combined_raw[combined_raw["cluster"] == 1].copy()
    if combined_raw.empty:
        raise ValueError("No rows found for cluster 1 in combined train/test history.")

    placeholder = combined_raw.head(1).copy()
    train_panel, _ = _build_zero_filled_panel(combined_raw, placeholder)
    future_panel = _future_grid(
        history_df=train_panel,
        product_col="product_family_name",
        cluster_col="cluster",
        value_col="total_sales",
        horizon_days=horizon_days,
    )
    train_features = _build_train_features(train_panel)
    fit_obj = _fit_zinb_with_config(
        train_feat=train_features,
        clip_q=DEFAULT_ZINB_CLIP_Q,
        clip_max_cap=DEFAULT_ZINB_CLIP_MAX_CAP,
        maxiter=200,
    )
    pred_df = _recursive_zinb_forecast(
        train_panel=train_panel,
        test_panel=future_panel,
        fit_obj=fit_obj,
        nonzero_threshold=DEFAULT_ZINB_THRESHOLD,
    )
    if "p_nonzero_zinb" in pred_df.columns and "p_sale" not in pred_df.columns:
        pred_df["p_sale"] = pred_df["p_nonzero_zinb"]
    pred_df["y"] = pd.NA
    pred_df["is_sale"] = pd.NA
    return pred_df


def _build_c2_future_prediction_frame(horizon_days: int) -> pd.DataFrame:
    from c2_m2 import (
        DEFAULT_CATEGORICAL_FEATURE_COLS,
        DEFAULT_FEATURE_COLS,
        _build_zero_filled_panel,
        _build_train_features,
        _recursive_forecast,
        _train_global_lgbm_regressor,
    )

    train_daily, test_daily = _load_train_test_daily()
    combined_raw = pd.concat([train_daily, test_daily], ignore_index=True)
    combined_raw = combined_raw[combined_raw["cluster"] == 2].copy()
    if combined_raw.empty:
        raise ValueError("No rows found for cluster 2 in combined train/test history.")

    placeholder = combined_raw.head(1).copy()
    train_panel, _ = _build_zero_filled_panel(combined_raw, placeholder)
    future_panel = _future_grid(
        history_df=train_panel,
        product_col="product_family_name",
        cluster_col="cluster",
        value_col="total_sales",
        horizon_days=horizon_days,
    )
    future_panel = future_panel.sort_values(["date", "product_family_name"]).reset_index(drop=True)
    train_feat = _build_train_features(train_panel)
    feature_cols = list(DEFAULT_FEATURE_COLS)
    categorical_feature_cols = [c for c in DEFAULT_CATEGORICAL_FEATURE_COLS if c in feature_cols]
    model_artifacts = _train_global_lgbm_regressor(
        train_feat=train_feat,
        feature_cols=feature_cols,
        categorical_feature_cols=categorical_feature_cols,
        reg_params=dict(C2_FIXED_BEST_CONFIG["reg_params"]),
        cap_q=float(C2_FIXED_BEST_CONFIG["cap_q"]),
        recent_weight_max=float(C2_FIXED_BEST_CONFIG["recent_weight_max"]),
        random_state=42,
    )
    raw_pred, pred_final, _ = _recursive_forecast(
        history_panel=train_panel,
        forecast_panel=future_panel,
        model_artifacts=model_artifacts,
        threshold_mode=str(C2_FIXED_BEST_CONFIG["threshold_mode"]),
        tau_quantile=float(C2_FIXED_BEST_CONFIG["tau_quantile"]),
    )

    pred_df = future_panel[["date", "product_family_name", "cluster"]].copy()
    pred_df["y"] = pd.NA
    pred_df["pred_global_lgbm_raw"] = np.asarray(raw_pred, dtype=float)
    pred_df["pred_global_lgbm"] = np.asarray(pred_final, dtype=float)
    return pred_df.sort_values(["product_family_name", "date"]).reset_index(drop=True)


def _build_c3_future_prediction_frame(horizon_days: int) -> pd.DataFrame:
    from c3_forecasting import (
        _build_feature_panel,
        _build_zero_filled_panel,
        _config_to_sklearn_params,
        _default_config,
        _fit_model,
        _recursive_forecast,
    )

    train_daily, test_daily = _load_train_test_daily()
    combined_raw = pd.concat([train_daily, test_daily], ignore_index=True)
    combined_raw = combined_raw[combined_raw["cluster"] == 3].copy()
    if combined_raw.empty:
        raise ValueError("No rows found for cluster 3 in combined train/test history.")

    combined_raw = combined_raw.rename(columns={"product_family_name": "sku_id", "total_sales": "y"})
    placeholder = combined_raw.head(1).copy()
    train_panel, _ = _build_zero_filled_panel(combined_raw, placeholder)
    future_panel = _future_grid(
        history_df=train_panel,
        product_col="sku_id",
        cluster_col="cluster",
        value_col="y",
        horizon_days=horizon_days,
    )
    train_feat = _build_feature_panel(train_panel)
    cfg = _default_config()
    clf_params, reg_params, use_lnc = _config_to_sklearn_params(cfg)
    fit_obj = _fit_model(train_feat, clf_params, reg_params, lognormal_correction=bool(use_lnc))
    pred_df = _recursive_forecast(
        train_panel=train_panel,
        test_panel=future_panel,
        fit_obj=fit_obj,
        occurrence_threshold=float(cfg["occurrence_threshold"]),
        prediction_floor=float(cfg["prediction_floor"]),
        seasonal_anchor_weight=float(cfg["seasonal_anchor_weight"]),
    )
    pred_df["y"] = pd.NA
    return pred_df


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required prediction source not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(source)
        except Exception as exc:
            csv_path = source.with_suffix(".csv")
            pkl_path = source.with_suffix(".pkl")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            if pkl_path.exists():
                return pd.read_pickle(pkl_path)
            raise RuntimeError(
                f"Prediction source {source} is not readable in the active environment and no CSV/PKL fallback was found."
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    raise ValueError(f"Unsupported prediction source type: {source}")


def _configure_future_runtime() -> None:
    runtime_env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "LOKY_MAX_CPU_COUNT": "1",
    }
    for key, value in runtime_env.items():
        os.environ.setdefault(key, value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build validated forecast artifacts for the product forecast agent.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and registry sources without running model builds.")
    parser.add_argument("--include-future", action="store_true", help="Also build future-mode artifacts from full train+test history.")
    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        help="Optional subset of cluster ids to build, for example `--clusters 0 2`.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_only:
        ensure_agent_dirs()
        build_product_metadata_frame()
        cluster_ids = sorted(int(cluster_id) for cluster_id in (args.clusters or CLUSTER_MODEL_CONFIG.keys()))
        for cluster_id in cluster_ids:
            config = CLUSTER_MODEL_CONFIG[cluster_id]
            if not config.enabled:
                continue
            _read_table(config.source_prediction_path)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "enabled_clusters": [cluster_id for cluster_id in cluster_ids if CLUSTER_MODEL_CONFIG[cluster_id].enabled],
                    "message": "Enabled prediction sources and metadata inputs are readable.",
                },
                indent=2,
            )
        )
        return

    manifest = build_assets(include_future=bool(args.include_future), cluster_ids=args.clusters)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
