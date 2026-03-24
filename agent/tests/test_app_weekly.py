from __future__ import annotations

import pandas as pd

from agent.app import _prepare_chart_frame, _prepare_display_frame


def test_prepare_display_frame_weekly_returns_period_start() -> None:
    forecast_df = pd.DataFrame(
        {
            "forecast_date": ["2024-01-01", "2024-01-02", "2024-01-09"],
            "forecast_value": [1.0, 2.0, 3.0],
            "actual_value": [2.0, 1.0, 4.0],
            "p_sale": [0.1, 0.2, 0.3],
        }
    )

    weekly = _prepare_display_frame(forecast_df=forecast_df, granularity="weekly")

    assert list(weekly.columns) == ["period_start", "forecast_value", "actual_value", "p_sale"]
    assert weekly["forecast_value"].tolist() == [3.0, 3.0]
    assert weekly["actual_value"].tolist() == [3.0, 4.0]


def test_prepare_display_frame_future_hides_actuals() -> None:
    forecast_df = pd.DataFrame(
        {
            "forecast_date": ["2024-01-01", "2024-01-02"],
            "forecast_value": [1.0, 2.0],
            "actual_value": [pd.NA, pd.NA],
            "p_sale": [0.1, 0.2],
        }
    )

    display_df = _prepare_display_frame(
        forecast_df=forecast_df,
        granularity="daily",
        include_actual=False,
    )

    assert list(display_df.columns) == ["forecast_date", "forecast_value", "p_sale"]


def test_prepare_chart_frame_future_weekly_combines_history_and_forecast() -> None:
    forecast_df = pd.DataFrame(
        {
            "forecast_date": ["2024-01-08", "2024-01-15"],
            "forecast_value": [5.0, 6.0],
            "actual_value": [pd.NA, pd.NA],
            "p_sale": [0.4, 0.5],
        }
    )
    history_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "actual_value": [2.0, 3.0],
        }
    )

    chart_df = _prepare_chart_frame(
        forecast_df=forecast_df,
        granularity="weekly",
        history_df=history_df,
    )

    assert list(chart_df.columns) == ["period_start", "forecast_value", "actual_value", "p_sale", "segment"]
    assert chart_df["segment"].tolist() == ["history", "forecast", "forecast"]
    assert chart_df["actual_value"].tolist()[0] == 5.0
    assert chart_df["forecast_value"].tolist()[1:] == [5.0, 6.0]
