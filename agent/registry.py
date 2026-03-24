from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from agent import PROJECT_ROOT
from agent.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_CLEANED_XLSX_PATH


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(value: object, max_len: int = 48) -> str:
    text = normalize_text(value).lower().replace(" ", "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "product"
    return text[:max_len].rstrip("-")


def product_key_for(cluster_id: int, product_family_name: str) -> str:
    import hashlib

    normalized = normalize_text(product_family_name)
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
    return f"c{cluster_id}-{slugify(product_family_name)}-{digest}"


def _normalize_desc(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_family_no_space(value: object) -> str:
    return normalize_text(value).replace(" ", "")


def load_desc_family_map(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH) -> Dict[str, str]:
    path = Path(checkpoint_path)
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            desc = _normalize_desc(obj.get("desc"))
            family = obj.get("result", {}).get("family_name")
            if desc:
                mapping[desc] = family or desc
    return mapping


def build_product_metadata_frame(
    cleaned_xlsx_path: str | Path = DEFAULT_CLEANED_XLSX_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> pd.DataFrame:
    xlsx_path = Path(cleaned_xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Cleaned retail workbook not found: {xlsx_path}")

    desc_map = load_desc_family_map(checkpoint_path=checkpoint_path)
    workbook = pd.ExcelFile(xlsx_path, engine="openpyxl")
    parts: List[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
        if "Description" not in df.columns:
            continue
        df = df.copy()
        df["Description"] = df["Description"].map(_normalize_desc)
        df["StockCode"] = df.get("StockCode", pd.Series(dtype="string")).astype("string").str.strip()
        df = df[df["Description"] != ""].copy()
        df["family_name_raw"] = df["Description"].map(lambda x: desc_map.get(x, x))
        df["family_name_norm"] = df["family_name_raw"].map(_normalize_family_no_space)
        parts.append(df[["StockCode", "Description", "family_name_norm"]])

    if not parts:
        return pd.DataFrame(
            columns=[
                "product_family_name",
                "normalized_product_name",
                "product_id",
                "product_id_norm",
                "search_aliases",
                "search_aliases_norm",
                "description_examples",
            ]
        )

    combined = pd.concat(parts, ignore_index=True)
    family_stock_n = combined.groupby("family_name_norm")["StockCode"].nunique(dropna=True)
    keep_families = set(family_stock_n[family_stock_n >= 2].index.tolist())

    combined["product_family_name"] = np.where(
        combined["family_name_norm"].isin(keep_families),
        combined["family_name_norm"],
        combined["Description"],
    )

    rows: List[Dict[str, object]] = []
    for product_family_name, grp in combined.groupby("product_family_name", dropna=False):
        stock_counter = Counter(
            stock for stock in grp["StockCode"].dropna().astype(str).tolist() if stock.strip()
        )
        desc_counter = Counter(
            desc for desc in grp["Description"].dropna().astype(str).tolist() if desc.strip()
        )
        stock_codes = [code for code, _ in stock_counter.most_common()]
        description_examples = [desc for desc, _ in desc_counter.most_common(5)]

        aliases: List[str] = [str(product_family_name)]
        aliases.extend(stock_codes[:5])
        aliases.extend(description_examples[:5])
        aliases = [alias for alias in aliases if alias]

        product_id = stock_codes[0] if stock_codes else pd.NA
        rows.append(
            {
                "product_family_name": str(product_family_name),
                "normalized_product_name": normalize_text(product_family_name),
                "product_id": product_id,
                "product_id_norm": normalize_text(product_id),
                "search_aliases": aliases,
                "search_aliases_norm": sorted({normalize_text(alias) for alias in aliases if normalize_text(alias)}),
                "description_examples": description_examples,
            }
        )

    return pd.DataFrame(rows).sort_values(["normalized_product_name", "product_id"], na_position="last").reset_index(drop=True)


def build_registry_frame(
    canonical_forecasts: pd.DataFrame,
    product_metadata: pd.DataFrame,
    artifact_paths: Mapping[int, Path],
) -> pd.DataFrame:
    if canonical_forecasts.empty:
        return pd.DataFrame(
            columns=[
                "product_key",
                "product_family_name",
                "normalized_product_name",
                "product_id",
                "product_id_norm",
                "cluster",
                "model_id",
                "model_name",
                "artifact_path",
                "generated_at",
                "search_aliases",
                "search_aliases_norm",
                "description_examples",
            ]
        )

    latest_per_product = (
        canonical_forecasts.sort_values(["product_key", "forecast_date"])
        .groupby("product_key", as_index=False)
        .agg(
            product_family_name=("product_family_name", "first"),
            product_id=("product_id", "first"),
            cluster=("cluster", "first"),
            model_id=("model_id", "first"),
            model_name=("model_name", "first"),
            generated_at=("generated_at", "max"),
        )
    )

    registry = latest_per_product.merge(
        product_metadata,
        on=["product_family_name"],
        how="left",
        suffixes=("", "_meta"),
    )

    registry["normalized_product_name"] = registry["normalized_product_name"].fillna(
        registry["product_family_name"].map(normalize_text)
    )
    registry["product_id"] = registry["product_id"].fillna(registry["product_id_meta"])
    registry["product_id_norm"] = registry["product_id_norm"].fillna(registry["product_id"].map(normalize_text))
    registry["search_aliases"] = registry["search_aliases"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    registry["search_aliases_norm"] = registry["search_aliases_norm"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    registry["description_examples"] = registry["description_examples"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    registry["artifact_path"] = registry["cluster"].map(
        lambda value: _relative_project_path(artifact_paths[int(value)])
    )

    return registry[
        [
            "product_key",
            "product_family_name",
            "normalized_product_name",
            "product_id",
            "product_id_norm",
            "cluster",
            "model_id",
            "model_name",
            "artifact_path",
            "generated_at",
            "search_aliases",
            "search_aliases_norm",
            "description_examples",
        ]
    ].sort_values(["cluster", "normalized_product_name"]).reset_index(drop=True)


def serialize_registry_frame(registry_df: pd.DataFrame) -> pd.DataFrame:
    out = registry_df.copy()
    for col in ["search_aliases", "search_aliases_norm", "description_examples"]:
        out[col] = out[col].apply(lambda value: json.dumps(value or [], ensure_ascii=False))
    return out


def deserialize_registry_frame(registry_df: pd.DataFrame) -> pd.DataFrame:
    out = registry_df.copy()
    for col in ["search_aliases", "search_aliases_norm", "description_examples"]:
        if col in out.columns:
            out[col] = out[col].apply(_json_to_list)
    return out


def _json_to_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    except json.JSONDecodeError:
        pass
    return [text]


def _relative_project_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def rank_registry_matches(query: str, registry_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if registry_df.empty:
        return registry_df.assign(score=pd.Series(dtype=float))

    query_norm = normalize_text(query)
    if not query_norm:
        return registry_df.head(0).assign(score=pd.Series(dtype=float))

    scored = registry_df.copy()
    scored["score"] = scored.apply(lambda row: _score_registry_row(query_norm, row), axis=1)
    scored = scored[scored["score"] > 0].sort_values(["score", "normalized_product_name"], ascending=[False, True])
    return scored.head(limit).reset_index(drop=True)


def _score_registry_row(query_norm: str, row: pd.Series) -> float:
    candidates: List[str] = []
    for col in ["normalized_product_name", "product_id_norm"]:
        value = row.get(col)
        if isinstance(value, str) and value:
            candidates.append(value)
    candidates.extend(row.get("search_aliases_norm", []) or [])

    query_tokens = set(query_norm.split())
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == query_norm:
            return 1.0
        if candidate in query_norm or query_norm in candidate:
            best = max(best, 0.88 if len(candidate) >= 4 else 0.74)
        candidate_tokens = set(candidate.split())
        if candidate_tokens and query_tokens:
            jaccard = len(candidate_tokens & query_tokens) / len(candidate_tokens | query_tokens)
            if jaccard > 0:
                best = max(best, 0.40 + 0.45 * jaccard)
        best = max(best, SequenceMatcher(None, query_norm, candidate).ratio() * 0.65)
    return round(best, 6)


def nearest_registry_matches(query: str, registry_df: pd.DataFrame, limit: int = 5) -> List[str]:
    names = registry_df["normalized_product_name"].dropna().astype(str).tolist()
    query_norm = normalize_text(query)
    if not query_norm:
        return []
    matches = get_close_matches(query_norm, names, n=limit, cutoff=0.45)
    return matches
