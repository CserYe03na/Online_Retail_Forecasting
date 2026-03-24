from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agent import PROJECT_ROOT
from agent.config import CLUSTER_MODEL_CONFIG, DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, MANIFEST_PATH, REGISTRY_PATH
from agent.registry import (
    deserialize_registry_frame,
    nearest_registry_matches,
    normalize_text,
    rank_registry_matches,
)

FUZZY_MATCH_THRESHOLD = 0.55


@lru_cache(maxsize=4)
def load_registry(registry_path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Product registry not found at {path}. Run `python -m agent.build_assets` first."
        )
    df = pd.read_csv(path)
    return deserialize_registry_frame(df)


@lru_cache(maxsize=8)
def load_forecast_artifact(artifact_path: str | Path) -> pd.DataFrame:
    path = _resolve_project_path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact not found: {path}")
    df = pd.read_csv(path)
    if "forecast_date" in df.columns:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    return df


def clear_caches() -> None:
    load_registry.cache_clear()
    load_forecast_artifact.cache_clear()
    load_observed_history.cache_clear()


def resolve_product(query: str, registry_path: str | Path = REGISTRY_PATH) -> Dict[str, Any]:
    registry_df = load_registry(registry_path)
    query_norm = normalize_text(query)

    if not query_norm:
        return {"status": "not_found", "query": query, "matches": [], "nearest_matches": []}

    id_matches = registry_df[registry_df["product_id_norm"] == query_norm].copy()
    if len(id_matches) == 1:
        return {"status": "resolved", "query": query, "match": _row_to_payload(id_matches.iloc[0])}
    if len(id_matches) > 1:
        return {"status": "ambiguous", "query": query, "matches": _rows_to_payload(id_matches)}

    name_matches = registry_df[registry_df["normalized_product_name"] == query_norm].copy()
    if len(name_matches) == 1:
        return {"status": "resolved", "query": query, "match": _row_to_payload(name_matches.iloc[0])}
    if len(name_matches) > 1:
        return {"status": "ambiguous", "query": query, "matches": _rows_to_payload(name_matches)}

    alias_exact = registry_df[
        registry_df["search_aliases_norm"].apply(lambda aliases: query_norm in (aliases or []))
    ].copy()
    if len(alias_exact) == 1:
        return {"status": "resolved", "query": query, "match": _row_to_payload(alias_exact.iloc[0])}
    if len(alias_exact) > 1:
        return {"status": "ambiguous", "query": query, "matches": _rows_to_payload(alias_exact)}

    ranked = rank_registry_matches(query=query, registry_df=registry_df, limit=5)
    if ranked.empty:
        return {
            "status": "not_found",
            "query": query,
            "matches": [],
            "nearest_matches": nearest_registry_matches(query, registry_df, limit=5),
        }
    if float(ranked.iloc[0]["score"]) < FUZZY_MATCH_THRESHOLD:
        return {
            "status": "not_found",
            "query": query,
            "matches": [],
            "nearest_matches": nearest_registry_matches(query, registry_df, limit=5),
        }

    if len(ranked) == 1 or float(ranked.iloc[0]["score"] - ranked.iloc[min(1, len(ranked) - 1)]["score"]) >= 0.12:
        return {"status": "resolved", "query": query, "match": _row_to_payload(ranked.iloc[0])}

    return {
        "status": "ambiguous",
        "query": query,
        "matches": _rows_to_payload(ranked),
    }


def get_product_metadata(product_key: str, registry_path: str | Path = REGISTRY_PATH) -> Dict[str, Any]:
    registry_df = load_registry(registry_path)
    match = registry_df[registry_df["product_key"] == product_key]
    if match.empty:
        raise KeyError(f"Unknown product_key: {product_key}")
    return _row_to_payload(match.iloc[0])


def get_product_forecast(
    product_key: str,
    horizon_days: int = 14,
    mode: str = "evaluation",
    registry_path: str | Path = REGISTRY_PATH,
) -> Dict[str, Any]:
    metadata = get_product_metadata(product_key=product_key, registry_path=registry_path)
    cluster_id = int(metadata["cluster"])
    config = CLUSTER_MODEL_CONFIG[cluster_id]
    if mode == "evaluation":
        registry_override = Path(registry_path) != Path(REGISTRY_PATH)
        if registry_override and metadata.get("artifact_path"):
            artifact_path = str(metadata["artifact_path"])
        else:
            if not config.enabled or config.source_prediction_path is None:
                raise FileNotFoundError(
                    f"Evaluation artifacts are not available for cluster {cluster_id}. "
                    f"Use Future mode for this cluster."
                )
            artifact_path = str(config.artifact_path)
    elif mode == "future":
        artifact_path = str(config.future_artifact_path)
    else:
        raise ValueError(f"Unsupported forecast mode: {mode}")
    artifact_df = load_forecast_artifact(artifact_path)
    product_df = artifact_df[artifact_df["product_key"] == product_key].copy()
    if product_df.empty:
        raise KeyError(f"No {mode} forecast rows found for product_key={product_key}")

    product_df = product_df.sort_values("forecast_date").head(int(horizon_days)).reset_index(drop=True)
    return {
        "metadata": {**metadata, "mode": mode, "artifact_path": artifact_path},
        "forecast": product_df,
    }


@lru_cache(maxsize=2)
def load_observed_history(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
) -> pd.DataFrame:
    from agent.build_assets import _read_table

    train_df = _read_table(train_path).copy()
    test_df = _read_table(test_path).copy()
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["product_family_name"] = combined["product_family_name"].astype("string").str.strip()
    combined["cluster"] = pd.to_numeric(combined["cluster"], errors="coerce").astype("Int64")
    combined["total_sales"] = pd.to_numeric(combined["total_sales"], errors="coerce").fillna(0.0)
    combined = (
        combined.groupby(["date", "product_family_name", "cluster"], as_index=False)["total_sales"]
        .sum()
        .sort_values(["product_family_name", "date"])
        .reset_index(drop=True)
    )
    return combined


def get_product_history(product_key: str, registry_path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    metadata = get_product_metadata(product_key=product_key, registry_path=registry_path)
    try:
        history_df = load_observed_history()
    except Exception:
        return pd.DataFrame(columns=["date", "actual_value"])
    match = history_df[
        (history_df["product_family_name"] == metadata["product_family_name"])
        & (history_df["cluster"] == int(metadata["cluster"]))
    ].copy()
    if match.empty:
        return pd.DataFrame(columns=["date", "actual_value"])
    match = match.rename(columns={"total_sales": "actual_value"})
    return match[["date", "actual_value"]].sort_values("date").reset_index(drop=True)


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_to_payload(row: pd.Series) -> Dict[str, Any]:
    payload = row.to_dict()
    for key in ["search_aliases", "search_aliases_norm", "description_examples"]:
        payload[key] = list(payload.get(key) or [])
    return payload


def _rows_to_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_row_to_payload(row) for _, row in df.iterrows()]


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
