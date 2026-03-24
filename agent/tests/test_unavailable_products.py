from __future__ import annotations

import pandas as pd

from agent.unavailable_products import resolve_unavailable_product


def test_resolve_unavailable_product_matches_cluster4_id(tmp_path, monkeypatch) -> None:
    unavailable_path = tmp_path / "cluster4_products.csv"
    pd.DataFrame(
        {
            "product_family_name": ["SMALL FAIRY CAKE FRIDGE MAGNETS"],
            "product_id": ["85220"],
            "reason_type": ["cluster_4_excluded"],
            "reason_message": ["cluster 4 excluded"],
        }
    ).to_csv(unavailable_path, index=False)

    monkeypatch.setattr(
        "agent.unavailable_products.DROPPED_PRODUCTS_PATH",
        tmp_path / "missing.csv",
    )
    monkeypatch.setattr(
        "agent.unavailable_products.CLUSTER4_PRODUCTS_PATH",
        unavailable_path,
    )
    from agent import unavailable_products as unavailable_module

    unavailable_module.load_unavailable_products.cache_clear()
    result = resolve_unavailable_product("85220")

    assert result is not None
    assert result["status"] == "unavailable"
    assert result["match"]["reason_type"] == "cluster_4_excluded"


def test_resolve_unavailable_product_matches_dropped_name(tmp_path, monkeypatch) -> None:
    unavailable_path = tmp_path / "dropped_before_clustering.csv"
    pd.DataFrame(
        {
            "product_family_name": ["KNICKKNACKTINSSET4"],
            "product_id": ["21400"],
            "train_sales_amount": [99.6],
            "reason_type": ["dropped_before_clustering"],
            "reason_message": ["dropped before clustering"],
        }
    ).to_csv(unavailable_path, index=False)

    monkeypatch.setattr(
        "agent.unavailable_products.DROPPED_PRODUCTS_PATH",
        unavailable_path,
    )
    monkeypatch.setattr(
        "agent.unavailable_products.CLUSTER4_PRODUCTS_PATH",
        tmp_path / "missing.csv",
    )
    from agent import unavailable_products as unavailable_module

    unavailable_module.load_unavailable_products.cache_clear()
    result = resolve_unavailable_product("Give me a forecast for KNICKKNACKTINSSET4")

    assert result is not None
    assert result["status"] == "unavailable"
    assert result["match"]["product_family_name"] == "KNICKKNACKTINSSET4"


def test_resolve_unavailable_product_does_not_fuzzy_match_normal_product(tmp_path, monkeypatch) -> None:
    unavailable_path = tmp_path / "dropped_before_clustering.csv"
    pd.DataFrame(
        {
            "product_family_name": ["Adjust bad debt", "Discount"],
            "product_id": ["BADD", "DISC"],
            "reason_type": ["dropped_before_clustering", "dropped_before_clustering"],
            "reason_message": ["dropped before clustering", "dropped before clustering"],
        }
    ).to_csv(unavailable_path, index=False)

    monkeypatch.setattr(
        "agent.unavailable_products.DROPPED_PRODUCTS_PATH",
        unavailable_path,
    )
    monkeypatch.setattr(
        "agent.unavailable_products.CLUSTER4_PRODUCTS_PATH",
        tmp_path / "missing.csv",
    )
    from agent import unavailable_products as unavailable_module

    unavailable_module.load_unavailable_products.cache_clear()
    result = resolve_unavailable_product("Give me a weekly forecast for 12 PENCIL SMALL TUBE WOODLAND for 6 weeks")

    assert result is None
