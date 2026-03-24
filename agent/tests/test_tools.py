from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import agent.tools as agent_tools
from agent.tools import get_product_forecast, resolve_product, resolve_product_strict


def _write_registry_and_artifact(tmp_path):
    artifact_path = tmp_path / "cluster_1_forecasts.csv"
    artifact_df = pd.DataFrame(
        {
            "product_key": ["c1-cherry-lights-12345678"] * 3 + ["c1-cherry-lamp-87654321"] * 2,
            "product_family_name": ["CHERRY LIGHTS"] * 3 + ["CHERRY LAMP"] * 2,
            "product_id": ["85048"] * 3 + ["85049"] * 2,
            "cluster": [1] * 5,
            "model_id": ["c1_zinb"] * 5,
            "model_name": ["ZINB"] * 5,
            "forecast_date": ["2011-12-01", "2011-12-02", "2011-12-03", "2011-12-01", "2011-12-02"],
            "forecast_value": [5.0, 6.0, 7.0, 2.0, 3.0],
            "p_sale": [0.8, 0.7, 0.9, 0.4, 0.5],
            "generated_at": ["2026-03-22T00:00:00Z"] * 5,
        }
    )
    artifact_df.to_csv(artifact_path, index=False)

    registry_path = tmp_path / "registry.csv"
    registry_df = pd.DataFrame(
        {
            "product_key": ["c1-cherry-lights-12345678", "c1-cherry-lamp-87654321"],
            "product_family_name": ["CHERRY LIGHTS", "CHERRY LAMP"],
            "normalized_product_name": ["CHERRY LIGHTS", "CHERRY LAMP"],
            "product_id": ["85048", "85049"],
            "product_id_norm": ["85048", "85049"],
            "cluster": [1, 1],
            "model_id": ["c1_zinb", "c1_zinb"],
            "model_name": ["ZINB", "ZINB"],
            "artifact_path": [str(artifact_path), str(artifact_path)],
            "generated_at": ["2026-03-22T00:00:00Z", "2026-03-22T00:00:00Z"],
            "search_aliases": [json.dumps(["CHERRY LIGHTS", "85048"]), json.dumps(["CHERRY LAMP", "85049"])],
            "search_aliases_norm": [json.dumps(["CHERRY LIGHTS", "85048"]), json.dumps(["CHERRY LAMP", "85049"])],
            "description_examples": [json.dumps(["PINK CHERRY LIGHTS"]), json.dumps(["CHERRY LAMP"])],
        }
    )
    registry_df.to_csv(registry_path, index=False)
    return registry_path


def test_resolve_product_supports_exact_id_and_ambiguous_and_not_found(tmp_path) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)

    exact = resolve_product("85048", registry_path=registry_path)
    assert exact["status"] == "resolved"
    assert exact["match"]["product_family_name"] == "CHERRY LIGHTS"

    ambiguous = resolve_product("cherry", registry_path=registry_path)
    assert ambiguous["status"] == "ambiguous"
    assert len(ambiguous["matches"]) == 2

    not_found = resolve_product("banana lantern", registry_path=registry_path)
    assert not_found["status"] == "not_found"


def test_resolve_product_strict_disables_fuzzy_resolution(tmp_path) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)

    strict = resolve_product_strict("banana lantern", registry_path=registry_path)
    assert strict["status"] == "not_found"
    assert strict["nearest_matches"] == []

    fuzzy = resolve_product("cherry", registry_path=registry_path)
    assert fuzzy["status"] == "ambiguous"


def test_get_product_forecast_returns_sorted_horizon(tmp_path) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)
    result = get_product_forecast(
        product_key="c1-cherry-lights-12345678",
        horizon_days=2,
        registry_path=registry_path,
    )

    forecast_df = result["forecast"]
    assert forecast_df["forecast_date"].dt.strftime("%Y-%m-%d").tolist() == ["2011-12-01", "2011-12-02"]
    assert forecast_df["forecast_value"].tolist() == [5.0, 6.0]


def test_get_product_forecast_resolves_project_relative_artifact_path(tmp_path) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)
    relative_artifact_path = Path("agent/artifacts/test_relative_forecasts.csv")
    absolute_artifact_path = agent_tools.PROJECT_ROOT / relative_artifact_path
    absolute_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "product_key": ["c1-cherry-lights-12345678"] * 2,
            "product_family_name": ["CHERRY LIGHTS"] * 2,
            "product_id": ["85048"] * 2,
            "cluster": [1, 1],
            "model_id": ["c1_zinb"] * 2,
            "model_name": ["ZINB"] * 2,
            "forecast_date": ["2011-12-01", "2011-12-02"],
            "forecast_value": [5.0, 6.0],
            "p_sale": [0.8, 0.7],
            "generated_at": ["2026-03-22T00:00:00Z"] * 2,
        }
    ).to_csv(absolute_artifact_path, index=False)

    registry_df = pd.read_csv(registry_path)
    registry_df["artifact_path"] = str(relative_artifact_path)
    registry_df.to_csv(registry_path, index=False)

    agent_tools.clear_caches()
    try:
        result = get_product_forecast(
            product_key="c1-cherry-lights-12345678",
            horizon_days=2,
            registry_path=registry_path,
        )
    finally:
        absolute_artifact_path.unlink(missing_ok=True)

    forecast_df = result["forecast"]
    assert forecast_df["forecast_value"].tolist() == [5.0, 6.0]


def test_get_product_forecast_future_mode_reads_future_artifact(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)
    future_artifact_path = tmp_path / "cluster_1_future_forecasts.csv"
    pd.DataFrame(
        {
            "product_key": ["c1-cherry-lights-12345678"] * 2,
            "product_family_name": ["CHERRY LIGHTS"] * 2,
            "product_id": ["85048"] * 2,
            "cluster": [1, 1],
            "model_id": ["c1_zinb"] * 2,
            "model_name": ["ZINB"] * 2,
            "forecast_date": ["2011-12-04", "2011-12-05"],
            "forecast_value": [8.0, 9.0],
            "p_sale": [0.6, 0.7],
            "generated_at": ["2026-03-22T00:00:00Z"] * 2,
        }
    ).to_csv(future_artifact_path, index=False)

    monkeypatch.setattr(
        agent_tools,
        "CLUSTER_MODEL_CONFIG",
        {1: SimpleNamespace(future_artifact_path=future_artifact_path)},
    )
    agent_tools.clear_caches()

    result = get_product_forecast(
        product_key="c1-cherry-lights-12345678",
        horizon_days=2,
        mode="future",
        registry_path=registry_path,
    )

    forecast_df = result["forecast"]
    assert forecast_df["forecast_date"].dt.strftime("%Y-%m-%d").tolist() == ["2011-12-04", "2011-12-05"]
    assert forecast_df["forecast_value"].tolist() == [8.0, 9.0]
    assert result["metadata"]["mode"] == "future"


def test_get_product_history_returns_empty_when_observed_history_is_unreadable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _write_registry_and_artifact(tmp_path)

    monkeypatch.setattr(
        agent_tools,
        "load_observed_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad parquet")),
    )

    history_df = agent_tools.get_product_history(
        product_key="c1-cherry-lights-12345678",
        registry_path=registry_path,
    )

    assert history_df.empty
    assert list(history_df.columns) == ["date", "actual_value"]
