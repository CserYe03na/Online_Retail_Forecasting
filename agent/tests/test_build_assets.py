from __future__ import annotations

import pandas as pd
import pytest

from agent.build_assets import normalize_prediction_frame, validate_canonical_forecast_frame


def test_validate_canonical_forecast_frame_rejects_missing_columns() -> None:
    df = pd.DataFrame({"product_key": ["a"]})
    with pytest.raises(ValueError):
        validate_canonical_forecast_frame(df)


def test_normalize_prediction_frame_cluster3_renames_sku_id() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2011-11-01", "2011-11-02"],
            "sku_id": ["CHERRY LIGHTS", "CHERRY LIGHTS"],
            "cluster": [3, 3],
            "y": [2.0, 4.0],
            "pred_two_stage": [2.5, 3.0],
            "p_sale": [0.8, 0.9],
        }
    )
    metadata = pd.DataFrame(
        {
            "product_family_name": ["CHERRY LIGHTS"],
            "product_id": ["85048"],
        }
    )

    out = normalize_prediction_frame(pred_df=pred_df, cluster_id=3, product_metadata=metadata)

    assert list(out.columns) == [
        "product_key",
        "product_family_name",
        "product_id",
        "cluster",
        "model_id",
        "model_name",
        "forecast_date",
        "forecast_value",
        "p_sale",
        "generated_at",
        "actual_value",
    ]
    assert out["product_family_name"].tolist() == ["CHERRY LIGHTS", "CHERRY LIGHTS"]
    assert out["product_id"].tolist() == ["85048", "85048"]
    assert out["forecast_value"].tolist() == [2.5, 3.0]
    assert out["actual_value"].tolist() == [2.0, 4.0]


def test_normalize_prediction_frame_cluster1_uses_p_sale_column_when_present() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2011-12-01", "2011-12-02"],
            "product_family_name": ["GLASS BALL LIGHTS", "GLASS BALL LIGHTS"],
            "cluster": [1, 1],
            "pred_zinb": [4.0, 5.0],
            "p_sale": [0.7, 0.8],
        }
    )

    out = normalize_prediction_frame(pred_df=pred_df, cluster_id=1)

    assert out["p_sale"].tolist() == [0.7, 0.8]
