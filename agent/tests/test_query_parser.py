from __future__ import annotations

from agent.query_parser import parse_request_options


def test_parse_request_options_extracts_days_and_defaults_daily() -> None:
    opts = parse_request_options("Show me the next 30 days for CHERRY LIGHTS")
    assert opts["horizon_days"] == 30
    assert opts["granularity"] == "daily"

def test_parse_request_options_extracts_product_query_from_stock_code_phrase() -> None:
    opts = parse_request_options("Show me the next 8 weeks for stock code 20973")
    assert opts["product_query"] == "20973"


def test_parse_request_options_extracts_product_query_from_named_request() -> None:
    opts = parse_request_options("Give me a weekly forecast for 12 PENCIL SMALL TUBE WOODLAND for 6 weeks")
    assert opts["product_query"] == "12 PENCIL SMALL TUBE WOODLAND"


def test_parse_request_options_drops_leading_article_in_product_query() -> None:
    opts = parse_request_options("Give me a forecast for KNICKKNACKTINSSET4")
    assert opts["product_query"] == "KNICKKNACKTINSSET4"
