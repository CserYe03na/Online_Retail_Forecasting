from __future__ import annotations

from agent.query_parser import parse_request_options


def test_parse_request_options_extracts_days_and_defaults_daily() -> None:
    opts = parse_request_options("Show me the next 30 days for CHERRY LIGHTS")
    assert opts["horizon_days"] == 30
    assert opts["granularity"] == "daily"


def test_parse_request_options_extracts_weeks_and_switches_weekly() -> None:
    opts = parse_request_options("给我这个产品未来6周的weekly forecast")
    assert opts["horizon_days"] == 42
    assert opts["granularity"] == "weekly"
