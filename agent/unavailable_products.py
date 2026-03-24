from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agent import AGENT_ROOT
from agent.registry import normalize_text


REFERENCE_DIR = AGENT_ROOT / "reference"
DROPPED_PRODUCTS_PATH = REFERENCE_DIR / "dropped_before_clustering.csv"
CLUSTER4_PRODUCTS_PATH = REFERENCE_DIR / "cluster4_products.csv"


@lru_cache(maxsize=1)
def load_unavailable_products() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in [DROPPED_PRODUCTS_PATH, CLUSTER4_PRODUCTS_PATH]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "product_family_name",
                "product_id",
                "reason_type",
                "reason_message",
                "normalized_product_name",
                "product_id_norm",
                "search_aliases_norm",
            ]
        )

    unavailable = pd.concat(frames, ignore_index=True)
    unavailable["product_family_name"] = unavailable["product_family_name"].astype("string").fillna("").str.strip()
    unavailable["product_id"] = unavailable["product_id"].astype("string").fillna("").str.strip()
    unavailable["normalized_product_name"] = unavailable["product_family_name"].map(normalize_text)
    unavailable["product_id_norm"] = unavailable["product_id"].map(normalize_text)
    unavailable["search_aliases_norm"] = unavailable.apply(
        lambda row: sorted(
            {
                alias
                for alias in [
                    normalize_text(row.get("product_family_name")),
                    normalize_text(row.get("product_id")),
                ]
                if alias
            }
        ),
        axis=1,
    )
    unavailable["description_examples"] = [[] for _ in range(len(unavailable))]
    return unavailable


def resolve_unavailable_product(query: str) -> Dict[str, Any] | None:
    unavailable_df = load_unavailable_products()
    if unavailable_df.empty:
        return None

    query_norm = normalize_text(query)
    if not query_norm:
        return None

    id_matches = unavailable_df[unavailable_df["product_id_norm"] == query_norm].copy()
    if len(id_matches) == 1:
        return {"status": "unavailable", "match": _row_to_payload(id_matches.iloc[0])}

    name_matches = unavailable_df[unavailable_df["normalized_product_name"] == query_norm].copy()
    if len(name_matches) == 1:
        return {"status": "unavailable", "match": _row_to_payload(name_matches.iloc[0])}

    alias_matches = unavailable_df[
        unavailable_df["search_aliases_norm"].apply(lambda aliases: query_norm in (aliases or []))
    ].copy()
    if len(alias_matches) == 1:
        return {"status": "unavailable", "match": _row_to_payload(alias_matches.iloc[0])}

    substring_matches = unavailable_df[
        unavailable_df.apply(lambda row: _query_contains_unavailable_alias(query_norm, row), axis=1)
    ].copy()
    if len(substring_matches) == 1:
        return {"status": "unavailable", "match": _row_to_payload(substring_matches.iloc[0])}

    return None


def _query_contains_unavailable_alias(query_norm: str, row: pd.Series) -> bool:
    if not query_norm:
        return False

    padded_query = f" {query_norm} "
    candidates = [
        row.get("normalized_product_name"),
        row.get("product_id_norm"),
        *(row.get("search_aliases_norm") or []),
    ]
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue
        if len(candidate_norm) < 4:
            continue
        if f" {candidate_norm} " in padded_query:
            return True
    return False


def _row_to_payload(row: pd.Series) -> Dict[str, Any]:
    payload = row.to_dict()
    payload["search_aliases_norm"] = list(payload.get("search_aliases_norm") or [])
    return payload
