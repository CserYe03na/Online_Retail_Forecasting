from __future__ import annotations

import json
from typing import Any, Dict, List


FORECAST_AGENT_SYSTEM_PROMPT = """You are a retail forecasting assistant for business managers.

Rules:
- Use only the supplied product metadata and forecast rows.
- Never invent dates, forecast values, product IDs, cluster assignments, or model names.
- If the data includes uncertainty proxies such as p_sale, mention them briefly and concretely.
- Keep the response short, managerial, and specific to the product asked about.
- Focus on the next 14 forecast dates only.
"""


def build_summary_payload(
    user_query: str,
    metadata: Dict[str, Any],
    forecast_rows: List[Dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "user_query": user_query,
            "product": {
                "product_family_name": metadata.get("product_family_name"),
                "product_id": metadata.get("product_id"),
                "cluster": metadata.get("cluster"),
                "model_name": metadata.get("model_name"),
                "generated_at": metadata.get("generated_at"),
            },
            "forecast_rows": forecast_rows,
        },
        ensure_ascii=False,
    )
