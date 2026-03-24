from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from agent import AGENT_ROOT, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

ARTIFACTS_DIR = AGENT_ROOT / "artifacts"
RAW_ARTIFACTS_DIR = ARTIFACTS_DIR / "raw"
REGISTRY_PATH = ARTIFACTS_DIR / "product_registry.csv"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
CHAT_HISTORY_PATH = ARTIFACTS_DIR / "chat_history.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ENABLE_OPENAI_QUERY_PARSER = os.getenv("ENABLE_OPENAI_QUERY_PARSER", "0").strip().lower() in {"1", "true", "yes", "on"}
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
OPENAI_QUERY_PARSER_MODEL = os.getenv("OPENAI_QUERY_PARSER_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "forecasting" / "train_daily.parquet"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "forecasting" / "test_daily.parquet"
DEFAULT_CLEANED_XLSX_PATH = PROJECT_ROOT / "data" / "online_retail_II_cleaned.xlsx"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "desc2fam_checkpoint.jsonl"
FUTURE_FORECAST_HORIZON_DAYS = 90

FORECAST_SCHEMA: List[str] = [
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
]


@dataclass(frozen=True)
class ClusterModelConfig:
    cluster_id: int
    model_id: str
    model_name: str
    prediction_column: str
    probability_column: Optional[str]
    artifact_filename: str
    future_artifact_filename: str
    source_prediction_path: Optional[Path] = None
    enabled: bool = True

    @property
    def artifact_path(self) -> Path:
        return ARTIFACTS_DIR / self.artifact_filename

    @property
    def raw_output_path(self) -> Path:
        return RAW_ARTIFACTS_DIR / f"cluster_{self.cluster_id}_raw.parquet"

    @property
    def future_artifact_path(self) -> Path:
        return ARTIFACTS_DIR / self.future_artifact_filename


CLUSTER_MODEL_CONFIG: Dict[int, ClusterModelConfig] = {
    0: ClusterModelConfig(
        cluster_id=0,
        model_id="c0_m2_two_stage_lgbm",
        model_name="Two-stage LGBM",
        prediction_column="pred_two_stage_lgbm",
        probability_column="p_sale",
        artifact_filename="cluster_0_forecasts.csv",
        future_artifact_filename="cluster_0_future_forecasts.csv",
        source_prediction_path=PROJECT_ROOT / "data" / "forecasting" / "c0_prediction.parquet",
    ),
    1: ClusterModelConfig(
        cluster_id=1,
        model_id="c1_zinb",
        model_name="ZINB",
        prediction_column="pred_zinb",
        probability_column="p_sale",
        artifact_filename="cluster_1_forecasts.csv",
        future_artifact_filename="cluster_1_future_forecasts.csv",
        source_prediction_path=PROJECT_ROOT / "data" / "forecasting" / "c1_prediction.parquet",
    ),
    2: ClusterModelConfig(
        cluster_id=2,
        model_id="c2_m2_global_lgbm",
        model_name="Global LGBM",
        prediction_column="pred_global_lgbm",
        probability_column=None,
        artifact_filename="cluster_2_forecasts.csv",
        future_artifact_filename="cluster_2_future_forecasts.csv",
        source_prediction_path=PROJECT_ROOT / "data" / "forecasting" / "c2_prediction.parquet",
    ),
    3: ClusterModelConfig(
        cluster_id=3,
        model_id="c3_two_stage_hgb",
        model_name="Two-stage HGB",
        prediction_column="pred_two_stage",
        probability_column="p_sale",
        artifact_filename="cluster_3_forecasts.csv",
        future_artifact_filename="cluster_3_future_forecasts.csv",
        source_prediction_path=PROJECT_ROOT / "data" / "forecasting" / "c3_prediction.parquet",
    ),
}


def ensure_agent_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
