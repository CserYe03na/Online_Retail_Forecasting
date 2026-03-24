from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from agent import PROJECT_ROOT
from agent.build_assets import build_assets
from agent.config import CHAT_HISTORY_PATH
from agent.prompts import FORECAST_AGENT_SYSTEM_PROMPT, build_summary_payload
from agent.query_parser import parse_request_options
from agent.tools import get_product_forecast, get_product_history, load_manifest, resolve_product


st.set_page_config(page_title="Product Forecast Query Agent", layout="wide")

MAX_SAVED_QUERIES = 5

CLUSTER_GUIDE: List[Dict[str, str]] = [
    {
        "cluster": "C0",
        "title": "Frequent but Erratic",
        "model": "Two-stage LGBM",
        "description": "Moderately frequent demand with sharp volatility and weak structure. A two-stage model helps separate sale occurrence from sale size.",
    },
    {
        "cluster": "C1",
        "title": "Seasonal Intermittent",
        "model": "ZINB",
        "description": "Intermittent demand with strong recurring calendar patterns. The count process is structured enough for a leak-safe zero-inflated model.",
    },
    {
        "cluster": "C2",
        "title": "Dense, High-Volume",
        "model": "Global LGBM",
        "description": "Frequent demand with the highest sales scale. Stable enough for a global boosting model, but still not strongly seasonal.",
    },
    {
        "cluster": "C3",
        "title": "Sparse Long-Tail",
        "model": "Two-stage HGB",
        "description": "Low-density long-tail demand with limited seasonality and high randomness. A two-stage boosting model is used to capture both occurrence and magnitude.",
    },
]


def main() -> None:
    if "forecast_mode" not in st.session_state:
        st.session_state["forecast_mode"] = "future"
    st.title("Product Forecast Query Agent")
    st.caption("Ask for the latest validated forecast for a product by product name or stock code. Weekly queries are recommended.")

    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.9rem 1rem;border:2px solid #facc15;border-radius:0.9rem;background:#fff9db;margin-bottom:0.75rem;">
                <div style="font-size:0.82rem;font-weight:700;color:#92400e;letter-spacing:0.03em;text-transform:uppercase;">
                    Forecast Mode
                </div>
                <div style="font-size:0.95rem;color:#5b3b00;margin-top:0.2rem;">
                    Future mode is the default view and the recommended business-facing option.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        future_mode_enabled = st.toggle(
            "Future mode",
            value=st.session_state.get("forecast_mode", "future") == "future",
            help="Off: evaluation mode with test-set actuals. On: future mode using full history to forecast beyond the test set.",
        )
        st.session_state["forecast_mode"] = "future" if future_mode_enabled else "evaluation"
        st.caption(
            "Mode: Future (train+test to future)" if future_mode_enabled else "Mode: Evaluation (test set with actuals)"
        )

        st.subheader("Artifacts")
        manifest = load_manifest()
        if manifest is None:
            st.warning("No validated agent artifacts found yet.")
        else:
            st.success(f"Registry ready: {manifest.get('products', 0)} products")
            enabled_clusters = manifest.get("enabled_clusters", [])
            if enabled_clusters:
                st.caption(f"Enabled clusters: {', '.join(str(item) for item in enabled_clusters)}")
            if manifest.get("future_assets_built"):
                horizon = manifest.get("future_horizon_days") or "unknown"
                st.caption(f"Future assets ready: {horizon}-day horizon")
            else:
                st.warning("Future assets are not built yet. Use the build button below before querying in Future mode.")
            skipped_clusters = manifest.get("skipped_clusters", [])
            if skipped_clusters:
                skipped_ids = ", ".join(str(item.get("cluster_id")) for item in skipped_clusters)
                st.info(f"Evaluation artifacts are unavailable for clusters: {skipped_ids}. Future mode can still be built for supported clusters.")

        build_col1, build_col2 = st.columns(2)
        with build_col1:
            if st.button("Rebuild Evaluation", use_container_width=True):
                with st.spinner("Building evaluation artifacts..."):
                    try:
                        manifest = build_assets(include_future=False)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Build finished. Products indexed: {manifest.get('products', 0)}")
        with build_col2:
            if st.button("Rebuild Future", use_container_width=True):
                with st.spinner("Building future forecast artifacts..."):
                    try:
                        manifest = build_assets(include_future=True)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(
                            f"Future assets built for {manifest.get('future_horizon_days', 'unknown')} days."
                        )

        st.divider()
        st.subheader("Cluster Guide")
        for item in CLUSTER_GUIDE:
            st.markdown(
                (
                    f"**{item['cluster']} · {item['title']}**  \n"
                    f"Model: `{item['model']}`  \n"
                    f"{item['description']}"
                )
            )

        if st.button("Clear current chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_resolution = None
            st.rerun()

        transcript_bytes = _conversation_to_json_bytes(st.session_state.get("messages", []))
        st.download_button(
            "Download conversation JSON",
            data=transcript_bytes,
            file_name="forecast_agent_conversation.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Recent queries")
        history_items = _load_query_history()
        if not history_items:
            st.caption("No saved query history yet.")
        else:
            if st.button("Clear recent queries", key="clear_recent_queries", use_container_width=True):
                _clear_query_history()
                st.rerun()
        for idx, item in enumerate(reversed(history_items)):
            request_options = item.get("request_options") or {}
            mode = str(request_options.get("mode", "evaluation"))
            mode_prefix = "[Future]" if mode == "future" else "[Eval]"
            label = f"{mode_prefix} {item.get('user_query', 'query')} | {item.get('product_family_name', '')}"
            if st.button(label, key=f"history_query_{idx}", use_container_width=True):
                _replay_history_item(item)
                st.rerun()

        st.divider()
        st.markdown("Query examples")
        st.code("Give me a weekly forecast for 12 PENCIL SMALL TUBE WOODLAND for 6 weeks")
        st.code("Show me the next 8 weeks for stock code 20973")
        st.code("In Future mode, give me the next 12 weeks for 15CM CHRISTMAS GLASS BALL 20 LIGHTS")
        st.code("What is the weekly forecast for stock code 85048 for the next 2 months?")
        st.code("In Future mode, show me the next 10 weeks for CAKESTAND3TIER")
        st.code("Show me the weekly forecast for stock code 22423")
        st.code("Give me a weekly forecast for 12 COLOURED PARTY BALLOONS for 6 weeks")
        st.code("Show me the next 8 weeks for stock code 22436")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_resolution" not in st.session_state:
        st.session_state.pending_resolution = None

    for idx, message in enumerate(st.session_state.messages):
        _render_chat_message(message, idx)

    pending = st.session_state.pending_resolution
    if pending:
        _render_pending_resolution(pending)

    user_query = st.chat_input("Enter a product name or stock code, for example: weekly forecast for ALARMCLOCKBAKELIKE for 6 weeks")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        _handle_query(user_query)
        st.rerun()


def _handle_query(user_query: str) -> None:
    request_options = parse_request_options(user_query)
    request_options["mode"] = st.session_state.get("forecast_mode", "evaluation")
    try:
        resolution = resolve_product(user_query)
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": str(exc)})
        return

    if resolution["status"] == "resolved":
        try:
            response = _build_forecast_response(
                product_key=resolution["match"]["product_key"],
                user_query=user_query,
                request_options=request_options,
            )
        except FileNotFoundError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": _missing_artifact_message(request_options=request_options, error=exc),
                }
            )
            return
        st.session_state.messages.append(response)
        _append_query_history(response)
        st.session_state.pending_resolution = None
        return

    if resolution["status"] == "ambiguous":
        candidates = resolution["matches"][:5]
        content = "I found multiple matching products. Choose one below to load the forecast."
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.session_state.pending_resolution = {
            "query": user_query,
            "candidates": candidates,
        }
        return

    suggestions = resolution.get("nearest_matches", [])
    lines = ["I could not find a matching product in the validated forecast registry."]
    manifest = load_manifest() or {}
    enabled_clusters = manifest.get("enabled_clusters", [])
    if enabled_clusters:
        lines.append(f"The current registry includes products from clusters: {', '.join(str(item) for item in enabled_clusters)}.")
    if suggestions:
        lines.append("Closest matches:")
        lines.extend(f"- {item}" for item in suggestions)
    st.session_state.messages.append({"role": "assistant", "content": "\n".join(lines)})
    st.session_state.pending_resolution = None


def _render_pending_resolution(pending: Dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**Choose a product for:** `{pending['query']}`")
        candidates = pending["candidates"]
        labels = [_candidate_label(item) for item in candidates]
        selected_label = st.selectbox("Matching products", labels, key="candidate_selectbox")
        if st.button("Load selected forecast"):
            selected_idx = labels.index(selected_label)
            selected = candidates[selected_idx]
            request_options = parse_request_options(pending["query"])
            request_options["mode"] = st.session_state.get("forecast_mode", "evaluation")
            try:
                response = _build_forecast_response(
                    product_key=selected["product_key"],
                    user_query=pending["query"],
                    request_options=request_options,
                )
            except FileNotFoundError as exc:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": _missing_artifact_message(request_options=request_options, error=exc),
                    }
                )
                st.session_state.pending_resolution = None
                st.rerun()
                return
            st.session_state.messages.append(response)
            _append_query_history(response)
            st.session_state.pending_resolution = None
            st.rerun()


def _build_forecast_response(product_key: str, user_query: str, request_options: Dict[str, object]) -> Dict[str, Any]:
    horizon_days = int(request_options["horizon_days"])
    granularity = str(request_options["granularity"])
    mode = str(request_options.get("mode", "evaluation"))
    result = get_product_forecast(product_key=product_key, horizon_days=horizon_days, mode=mode)
    metadata = result["metadata"]
    forecast_df = result["forecast"].copy()
    display_df = _prepare_display_frame(
        forecast_df=forecast_df,
        granularity=granularity,
        include_actual=mode == "evaluation",
    )
    history_df = get_product_history(product_key=product_key) if mode == "future" else None
    chart_df = _prepare_chart_frame(
        forecast_df=forecast_df,
        granularity=granularity,
        history_df=history_df,
        include_actual=mode == "evaluation",
    )
    summary = _generate_summary(
        user_query=user_query,
        metadata=metadata,
        forecast_df=forecast_df,
        display_df=display_df,
        granularity=granularity,
    )

    info_lines = [
        f"**Product**: {_display_value(metadata.get('product_family_name'))}",
        f"**Product ID**: {_display_value(metadata.get('product_id'))}",
        f"**Cluster**: {_display_value(metadata.get('cluster'))}",
        f"**Model**: {_display_value(metadata.get('model_name'))}",
        f"**Mode**: {mode}",
        f"**Requested horizon**: {horizon_days} days",
        f"**Display granularity**: {granularity}",
    ]

    chart_rows = chart_df.to_dict(orient="records")
    serialized_rows = _serialize_records(chart_rows)

    return {
        "role": "assistant",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "product_key": product_key,
        "product_family_name": metadata.get("product_family_name"),
        "request_options": {
            "horizon_days": horizon_days,
            "granularity": granularity,
            "mode": mode,
        },
        "content": "  \n".join(info_lines) + "\n\n" + summary,
        "table": _serialize_records(display_df.to_dict(orient="records")),
        "chart": serialized_rows,
        "chart_granularity": granularity,
    }


def _generate_summary(
    user_query: str,
    metadata: Dict[str, Any],
    forecast_df: pd.DataFrame,
    display_df: pd.DataFrame,
    granularity: str,
) -> str:
    try:
        import os
        from openai import OpenAI
    except Exception:
        return _fallback_summary(metadata=metadata, forecast_df=display_df, granularity=granularity)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_summary(metadata=metadata, forecast_df=display_df, granularity=granularity)

    client = OpenAI(api_key=api_key)
    payload = build_summary_payload(
        user_query=user_query,
        metadata=metadata,
        forecast_rows=_forecast_rows_for_prompt(display_df, granularity=granularity),
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": FORECAST_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        text = getattr(response, "output_text", "") or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass

    return _fallback_summary(metadata=metadata, forecast_df=display_df, granularity=granularity)


def _prepare_display_frame(forecast_df: pd.DataFrame, granularity: str, include_actual: bool = True) -> pd.DataFrame:
    rows = forecast_df.copy()
    rows["forecast_date"] = pd.to_datetime(rows["forecast_date"], errors="coerce")

    if granularity == "weekly":
        agg_map = {"forecast_value": "sum"}
        if include_actual and "actual_value" in rows.columns and not rows["actual_value"].isna().all():
            agg_map["actual_value"] = "sum"
        if "p_sale" in rows.columns and not rows["p_sale"].isna().all():
            agg_map["p_sale"] = "mean"
        weekly = (
            rows.set_index("forecast_date")
            .resample("W-MON", label="left", closed="left")
            .agg(agg_map)
            .reset_index()
            .rename(columns={"forecast_date": "period_start"})
        )
        if include_actual and "actual_value" not in weekly.columns:
            weekly["actual_value"] = pd.NA
        if "p_sale" not in weekly.columns:
            weekly["p_sale"] = pd.NA
        weekly["period_start"] = pd.to_datetime(weekly["period_start"], errors="coerce").dt.date
        columns = ["period_start", "forecast_value"]
        if include_actual:
            columns.append("actual_value")
        columns.append("p_sale")
        return weekly[columns]

    display_df = rows[["forecast_date", "forecast_value"]].copy()
    if include_actual and "actual_value" in rows.columns and not rows["actual_value"].isna().all():
        display_df["actual_value"] = rows["actual_value"]
    elif include_actual:
        display_df["actual_value"] = pd.NA
    if "p_sale" in rows.columns and not rows["p_sale"].isna().all():
        display_df["p_sale"] = rows["p_sale"]
    else:
        display_df["p_sale"] = pd.NA
    display_df["forecast_date"] = pd.to_datetime(display_df["forecast_date"], errors="coerce").dt.date
    columns = ["forecast_date", "forecast_value"]
    if include_actual:
        columns.append("actual_value")
    columns.append("p_sale")
    return display_df[columns]


def _prepare_chart_frame(
    forecast_df: pd.DataFrame,
    granularity: str,
    history_df: pd.DataFrame | None = None,
    include_actual: bool = True,
) -> pd.DataFrame:
    forecast_display = _prepare_display_frame(
        forecast_df=forecast_df,
        granularity=granularity,
        include_actual=include_actual,
    ).copy()
    if history_df is None or history_df.empty:
        return forecast_display

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["actual_value"] = pd.to_numeric(history["actual_value"], errors="coerce").fillna(0.0)

    if granularity == "weekly":
        history_weekly = (
            history.set_index("date")
            .resample("W-MON", label="left", closed="left")
            .agg({"actual_value": "sum"})
            .reset_index()
            .rename(columns={"date": "period_start"})
        )
        history_weekly["period_start"] = pd.to_datetime(history_weekly["period_start"], errors="coerce").dt.date
        history_weekly["forecast_value"] = float("nan")
        history_weekly["p_sale"] = float("nan")
        history_weekly["segment"] = "history"

        forecast_weekly = forecast_display.copy()
        if "actual_value" not in forecast_weekly.columns:
            forecast_weekly["actual_value"] = float("nan")
        forecast_weekly["forecast_value"] = pd.to_numeric(forecast_weekly["forecast_value"], errors="coerce").astype(float)
        forecast_weekly["actual_value"] = pd.to_numeric(forecast_weekly["actual_value"], errors="coerce").astype(float)
        forecast_weekly["p_sale"] = pd.to_numeric(forecast_weekly["p_sale"], errors="coerce").astype(float)
        forecast_weekly["segment"] = "forecast"
        history_weekly["forecast_value"] = pd.to_numeric(history_weekly["forecast_value"], errors="coerce").astype(float)
        history_weekly["actual_value"] = pd.to_numeric(history_weekly["actual_value"], errors="coerce").astype(float)
        history_weekly["p_sale"] = pd.to_numeric(history_weekly["p_sale"], errors="coerce").astype(float)
        combined = pd.concat(
            [
                history_weekly[["period_start", "forecast_value", "actual_value", "p_sale", "segment"]],
                forecast_weekly[["period_start", "forecast_value", "actual_value", "p_sale", "segment"]],
            ],
            ignore_index=True,
        )
        return combined.sort_values(["period_start", "segment"]).reset_index(drop=True)

    history_daily = history.copy()
    history_daily["forecast_date"] = pd.to_datetime(history_daily["date"], errors="coerce").dt.date
    history_daily["forecast_value"] = float("nan")
    history_daily["p_sale"] = float("nan")
    history_daily["segment"] = "history"

    forecast_daily = forecast_display.copy()
    if "actual_value" not in forecast_daily.columns:
        forecast_daily["actual_value"] = float("nan")
    forecast_daily["forecast_value"] = pd.to_numeric(forecast_daily["forecast_value"], errors="coerce").astype(float)
    forecast_daily["actual_value"] = pd.to_numeric(forecast_daily["actual_value"], errors="coerce").astype(float)
    forecast_daily["p_sale"] = pd.to_numeric(forecast_daily["p_sale"], errors="coerce").astype(float)
    forecast_daily["segment"] = "forecast"
    history_daily["forecast_value"] = pd.to_numeric(history_daily["forecast_value"], errors="coerce").astype(float)
    history_daily["actual_value"] = pd.to_numeric(history_daily["actual_value"], errors="coerce").astype(float)
    history_daily["p_sale"] = pd.to_numeric(history_daily["p_sale"], errors="coerce").astype(float)
    combined = pd.concat(
        [
            history_daily[["forecast_date", "forecast_value", "actual_value", "p_sale", "segment"]],
            forecast_daily[["forecast_date", "forecast_value", "actual_value", "p_sale", "segment"]],
        ],
        ignore_index=True,
    )
    return combined.sort_values(["forecast_date", "segment"]).reset_index(drop=True)


def _forecast_rows_for_prompt(display_df: pd.DataFrame, granularity: str) -> List[Dict[str, Any]]:
    rows = display_df.copy()
    date_col = "period_start" if granularity == "weekly" else "forecast_date"
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return rows.fillna("").to_dict(orient="records")


def _fallback_summary(metadata: Dict[str, Any], forecast_df: pd.DataFrame, granularity: str) -> str:
    values = pd.to_numeric(forecast_df["forecast_value"], errors="coerce").fillna(0.0)
    start_value = float(values.iloc[0]) if not values.empty else 0.0
    end_value = float(values.iloc[-1]) if not values.empty else 0.0
    avg_value = float(values.mean()) if not values.empty else 0.0
    total_value = float(values.sum()) if not values.empty else 0.0
    zero_days = int((values <= 0).sum())
    periods = int(len(forecast_df))
    unit = "weeks" if granularity == "weekly" else "days"

    if end_value > start_value * 1.1:
        trend = f"The forecast trends upward over the {periods}-{unit[:-1]} window."
    elif start_value > end_value * 1.1:
        trend = f"The forecast trends downward over the {periods}-{unit[:-1]} window."
    else:
        trend = f"The forecast is broadly stable over the {periods}-{unit[:-1]} window."

    p_sale_line = ""
    if "p_sale" in forecast_df.columns and not forecast_df["p_sale"].isna().all():
        p_sale = pd.to_numeric(forecast_df["p_sale"], errors="coerce").dropna()
        if not p_sale.empty:
            p_sale_line = f" Average sale probability is {p_sale.mean():.2f}."

    zero_line = ""
    if granularity == "daily" and zero_days > 0:
        zero_line = (
            f" There are {zero_days} zero-forecast days in this daily view, which is normal for intermittent-demand products."
        )
    actual_line = ""
    if "actual_value" not in forecast_df.columns or pd.to_numeric(forecast_df["actual_value"], errors="coerce").isna().all():
        actual_line = " This mode does not include actual values because the forecast extends beyond the observed test period."

    return (
        f"Across the next {periods} {unit}, expected volume totals {total_value:.2f}, with an average of {avg_value:.2f} per {unit[:-1]}. "
        f"The first displayed period is {start_value:.2f} and the last displayed period is {end_value:.2f}. "
        f"{trend}{p_sale_line}{zero_line}{actual_line}"
    )


def _candidate_label(candidate: Dict[str, Any]) -> str:
    product_id = _display_value(candidate.get("product_id"))
    return f"{candidate.get('product_family_name')} | ID {product_id} | cluster {candidate.get('cluster')}"


def _display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(value)


def _missing_artifact_message(request_options: Dict[str, object], error: FileNotFoundError) -> str:
    mode = str(request_options.get("mode", "evaluation"))
    if mode == "future":
        return (
            "Future forecast artifacts are not built yet for the enabled clusters. "
            "Use the sidebar button `Build future assets` first, then rerun this query.\n\n"
            f"Details: {error}"
        )
    return f"Validated forecast artifacts are missing. Rebuild the evaluation assets first.\n\nDetails: {error}"


def _render_chat_message(message: Dict[str, Any], message_index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        table = message.get("table")
        if table:
            table_df = pd.DataFrame(table)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download table CSV",
                data=table_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_table_{message_index + 1}.csv",
                mime="text/csv",
                key=f"download_table_{message_index}",
            )
        chart = message.get("chart")
        if chart:
            chart_df = pd.DataFrame(chart)
            granularity = message.get("chart_granularity", "daily")
            fig = _build_chart_figure(
                chart_df=chart_df,
                granularity=granularity,
                product_name=message.get("product_family_name", "Product"),
            )
            st.pyplot(fig, use_container_width=True)
            png_bytes = _figure_to_png_bytes(fig)
            plt.close(fig)
            st.download_button(
                "Download chart PNG",
                data=png_bytes,
                file_name=f"forecast_chart_{message_index + 1}.png",
                mime="image/png",
                key=f"download_chart_{message_index}",
            )


def _serialize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in records:
        item: Dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                item[key] = None
            elif hasattr(value, "isoformat"):
                item[key] = value.isoformat()
            else:
                item[key] = value
        out.append(item)
    return out


def _build_chart_figure(chart_df: pd.DataFrame, granularity: str, product_name: str):
    date_col = "period_start" if granularity == "weekly" else "forecast_date"
    rows = chart_df.copy()
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce")
    has_actual = "actual_value" in rows.columns and not pd.to_numeric(rows["actual_value"], errors="coerce").isna().all()
    has_segment = "segment" in rows.columns and rows["segment"].notna().any()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if has_segment:
        history_rows = rows[rows["segment"] == "history"].copy()
        forecast_rows = rows[rows["segment"] == "forecast"].copy()
        if not history_rows.empty:
            ax.plot(
                history_rows[date_col],
                pd.to_numeric(history_rows["actual_value"], errors="coerce"),
                color="#1f77b4",
                linewidth=1.9,
                alpha=0.95,
                label="Observed history",
            )
        if not forecast_rows.empty:
            ax.plot(
                forecast_rows[date_col],
                pd.to_numeric(forecast_rows["forecast_value"], errors="coerce"),
                color="#d94801",
                linewidth=2.1,
                label="Future forecast",
            )
            forecast_start = pd.to_datetime(forecast_rows[date_col], errors="coerce").min()
            if pd.notna(forecast_start):
                ax.axvline(forecast_start, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.8, label="Forecast start")
        if not history_rows.empty and not forecast_rows.empty:
            connector_x = [
                pd.to_datetime(history_rows[date_col], errors="coerce").max(),
                pd.to_datetime(forecast_rows[date_col], errors="coerce").min(),
            ]
            connector_y = [
                float(pd.to_numeric(history_rows["actual_value"], errors="coerce").dropna().iloc[-1]),
                float(pd.to_numeric(forecast_rows["forecast_value"], errors="coerce").dropna().iloc[0]),
            ]
            ax.plot(
                connector_x,
                connector_y,
                color="#6b7280",
                linewidth=1.3,
                linestyle=":",
                alpha=0.9,
                label="History to forecast transition",
            )
    else:
        ax.plot(rows[date_col], rows["forecast_value"], color="#d94801", linewidth=2.0, label="Forecast")
    if has_actual and not has_segment:
        ax.plot(
            rows[date_col],
            pd.to_numeric(rows["actual_value"], errors="coerce"),
            color="#1f77b4",
            linewidth=1.8,
            alpha=0.85,
            label="Actual",
        )
    if has_segment:
        title = "Observed History + Future Forecast"
    elif has_actual:
        title = "Actual vs Forecast"
    else:
        title = "Forecast"
    ax.set_title(f"{product_name}: {title}")
    ax.set_xlabel("Week" if granularity == "weekly" else "Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf.getvalue()


def _conversation_to_json_bytes(messages: List[Dict[str, Any]]) -> bytes:
    return json.dumps(messages, ensure_ascii=False, indent=2).encode("utf-8")


def _load_query_history() -> List[Dict[str, Any]]:
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        with CHAT_HISTORY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data[-MAX_SAVED_QUERIES:]


def _append_query_history(response: Dict[str, Any]) -> None:
    history = _load_query_history()
    history.append(
        {
            "timestamp": response.get("timestamp"),
            "user_query": response.get("user_query"),
            "product_key": response.get("product_key"),
            "product_family_name": response.get("product_family_name"),
            "request_options": response.get("request_options", {}),
        },
    )
    history = history[-MAX_SAVED_QUERIES:]
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


def _clear_query_history() -> None:
    if CHAT_HISTORY_PATH.exists():
        CHAT_HISTORY_PATH.unlink()


def _replay_history_item(item: Dict[str, Any]) -> None:
    user_query = str(item.get("user_query") or "")
    request_options = item.get("request_options") or parse_request_options(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    try:
        response = _build_forecast_response(
            product_key=str(item["product_key"]),
            user_query=user_query,
            request_options=request_options,
        )
        st.session_state.messages.append(response)
    except FileNotFoundError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": _missing_artifact_message(request_options=request_options, error=exc)}
        )
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": str(exc)})


if __name__ == "__main__":
    main()
