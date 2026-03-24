from __future__ import annotations

import re
from typing import Dict


DEFAULT_HORIZON_DAYS = 14
MAX_HORIZON_DAYS = 90


def parse_request_options(query: str) -> Dict[str, object]:
    text = (query or "").strip().lower()
    horizon_days = DEFAULT_HORIZON_DAYS
    granularity = "daily"

    day_match = re.search(r"(\d+)\s*(?:days?|day|d|天)", text)
    week_match = re.search(r"(\d+)\s*(?:weeks?|week|w|周)", text)

    if day_match:
        horizon_days = int(day_match.group(1))
    elif week_match:
        horizon_days = int(week_match.group(1)) * 7
        granularity = "weekly"

    if any(token in text for token in ["weekly", "per week", "按周", "每周", "周度", "周汇总"]):
        granularity = "weekly"
    elif any(token in text for token in ["daily", "per day", "按天", "每天", "日度"]):
        granularity = "daily"

    horizon_days = max(1, min(int(horizon_days), MAX_HORIZON_DAYS))
    return {
        "horizon_days": horizon_days,
        "granularity": granularity,
    }
