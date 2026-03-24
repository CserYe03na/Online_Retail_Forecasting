from __future__ import annotations

import json
import re
from typing import Any, Dict

from agent.config import ENABLE_OPENAI_QUERY_PARSER, OPENAI_API_KEY, OPENAI_QUERY_PARSER_MODEL


DEFAULT_HORIZON_DAYS = 14
MAX_HORIZON_DAYS = 90

LEADING_QUERY_PATTERNS = [
    r"^\s*in\s+(?:future|evaluation)\s+mode[:,]?\s*",
    r"^\s*(?:please\s+)?(?:give me|show me|tell me|what is|what's|can you show me|i want)\s+",
    r"^\s*(?:a\s+|the\s+)?(?:weekly|daily)\s+forecast\s+(?:for\s+)?",
    r"^\s*forecast\s+(?:for\s+)?",
]

TRAILING_TIME_PATTERNS = [
    r"\bfor\s+the\s+next\s+\d+\s*(?:days?|weeks?|months?|d|w|天|周)\b",
    r"\bfor\s+\d+\s*(?:days?|weeks?|months?|d|w|天|周)\b",
    r"\bnext\s+\d+\s*(?:days?|weeks?|months?|d|w|天|周)\b",
    r"\bfuture\s+\d+\s*(?:days?|weeks?|months?|d|w|天|周)\b",
]

NOISE_TOKENS = [
    "stock code",
    "product id",
    "product",
    "forecast",
    "weekly",
    "daily",
    "per week",
    "per day",
    "next",
    "future",
    "latest",
]


def parse_request_options(query: str) -> Dict[str, object]:
    text = (query or "").strip()
    text_lower = text.lower()
    horizon_days = DEFAULT_HORIZON_DAYS
    granularity = "daily"

    day_match = re.search(r"(\d+)\s*(?:days?|day|d|天)", text_lower)
    week_match = re.search(r"(\d+)\s*(?:weeks?|week|w|周)", text_lower)
    month_match = re.search(r"(\d+)\s*(?:months?|month|m|个月)", text_lower)

    if day_match:
        horizon_days = int(day_match.group(1))
    elif week_match:
        horizon_days = int(week_match.group(1)) * 7
        granularity = "weekly"
    elif month_match:
        horizon_days = int(month_match.group(1)) * 30

    if any(token in text_lower for token in ["weekly", "per week", "按周", "每周", "周度", "周汇总"]):
        granularity = "weekly"
    elif any(token in text_lower for token in ["daily", "per day", "按天", "每天", "日度"]):
        granularity = "daily"

    product_query = _extract_product_query(text)
    horizon_days = max(1, min(int(horizon_days), MAX_HORIZON_DAYS))
    return {
        "horizon_days": horizon_days,
        "granularity": granularity,
        "product_query": product_query,
    }


def _extract_product_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return ""

    llm_product = _extract_product_query_with_llm(query)
    if llm_product:
        return llm_product
    return _heuristic_extract_product_query(query)


def _extract_product_query_with_llm(query: str) -> str:
    if not ENABLE_OPENAI_QUERY_PARSER or not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = (
        "Extract only the product reference from the user's forecasting request. "
        "Return strict JSON with one key: product_query. "
        "If the user refers to a stock code or product ID, return just that ID. "
        "If the user refers to a product name, return just that name. "
        "Do not include forecast terms, time horizon, mode, or extra words."
    )
    try:
        response = client.responses.create(
            model=OPENAI_QUERY_PARSER_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        payload = json.loads((getattr(response, "output_text", "") or "").strip())
    except Exception:
        return ""

    product_query = str(payload.get("product_query", "")).strip()
    if not product_query:
        return ""
    return product_query


def _heuristic_extract_product_query(query: str) -> str:
    text = query.strip()

    explicit_id = re.search(
        r"\b(?:stock\s*code|product\s*id|id)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]*)\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_id:
        return explicit_id.group(1).strip()

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    for left, right in quoted:
        candidate = (left or right).strip()
        if candidate:
            return candidate

    cleaned = text
    for pattern in LEADING_QUERY_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    for pattern in TRAILING_TIME_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\b(?:for|of)\s+stock\s+code\s+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:for|of)\s+product\s+id\s+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:for|about)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)

    for token in NOISE_TOKENS:
        cleaned = re.sub(rf"\b{re.escape(token)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[,:;]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")

    if re.fullmatch(r"[A-Z0-9][A-Z0-9\-]*", cleaned, flags=re.IGNORECASE):
        return cleaned.strip()

    return cleaned.strip()
