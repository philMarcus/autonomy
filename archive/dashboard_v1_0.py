# dashboard_v1_0.py
# Streamlit dashboard for Autonomy telemetry (DuckDB views over Parquet)
#
# Prereq (run once in DuckDB CLI):
#   .read sql/views.sql
#
# Run:
#   streamlit run dashboard_v1_0.py

from __future__ import annotations

import os
from datetime import date

import duckdb
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


DB_PATH = "warehouse/telemetry.duckdb"


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Missing {DB_PATH}. Create it and run `.read sql/views.sql` in DuckDB first."
        )
    return duckdb.connect(DB_PATH)


def qdf(sql: str, params: tuple | None = None) -> pd.DataFrame:
    con = get_con()
    if params is None:
        return con.execute(sql).df()
    return con.execute(sql, params).df()


def ensure_views_exist() -> None:
    # Minimal check: events view exists
    df = qdf("SELECT count(*) AS n FROM information_schema.views WHERE table_name='events';")
    if int(df.loc[0, "n"]) == 0:
        raise RuntimeError(
            "DuckDB views not found. In DuckDB CLI, run:\n"
            "  .read sql/views.sql\n"
            "and re-run the dashboard."
        )


def main():
    st.set_page_config(page_title="Autonomy Dashboard v1.0", layout="wide")
    st.title("Autonomy Dashboard v1.0")

    ensure_views_exist()

    # ---- Determine min/max dt and brains for filters
    meta = qdf("SELECT min(dt) AS min_dt, max(dt) AS max_dt FROM events;")
    min_dt = meta.loc[0, "min_dt"]
    max_dt = meta.loc[0, "max_dt"]

    if min_dt is None or max_dt is None:
        st.warning("No data found in events view.")
        st.stop()

    brains_df = qdf(
        """
        SELECT DISTINCT brain
        FROM events
        WHERE brain IS NOT NULL
        ORDER BY brain
        """
    )
    brains = ["(all)"] + brains_df["brain"].tolist()

    # ---- Sidebar filters
    st.sidebar.header("Filters")

    start_dt, end_dt = st.sidebar.date_input(
        "Date range (dt)",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
    )
    if isinstance(start_dt, (list, tuple)):
        start_dt, end_dt = start_dt[0], start_dt[1]

    brain = st.sidebar.selectbox("Brain", brains, index=0)

    # Event types/tags depend on the date range + brain
    if brain == "(all)":
        et_df = qdf(
            """
            SELECT DISTINCT event_type
            FROM events
            WHERE dt >= ? AND dt <= ? AND event_type IS NOT NULL
            ORDER BY event_type
            """,
            (start_dt, end_dt),
        )
        tag_df = qdf(
            """
            SELECT DISTINCT tag
            FROM events
            WHERE dt >= ? AND dt <= ? AND tag IS NOT NULL
            ORDER BY tag
            """,
            (start_dt, end_dt),
        )
    else:
        et_df = qdf(
            """
            SELECT DISTINCT event_type
            FROM events
            WHERE dt >= ? AND dt <= ? AND brain = ? AND event_type IS NOT NULL
            ORDER BY event_type
            """,
            (start_dt, end_dt, brain),
        )
        tag_df = qdf(
            """
            SELECT DISTINCT tag
            FROM events
            WHERE dt >= ? AND dt <= ? AND brain = ? AND tag IS NOT NULL
            ORDER BY tag
            """,
            (start_dt, end_dt, brain),
        )

    event_types = ["(all)"] + et_df["event_type"].tolist()
    event_type = st.sidebar.selectbox("Event type", event_types, index=0)

    tags = ["(all)"] + tag_df["tag"].tolist()
    tag = st.sidebar.selectbox("Tag", tags, index=0)

    show_payload = st.sidebar.checkbox("Show payload_json in tables", value=False)

    # ---- WHERE builder
    def build_where() -> tuple[str, tuple]:
        clauses = ["dt >= ?", "dt <= ?"]
        params = [start_dt, end_dt]

        if brain != "(all)":
            clauses.append("brain = ?")
            params.append(brain)

        if event_type != "(all)":
            clauses.append("event_type = ?")
            params.append(event_type)

        if tag != "(all)":
            clauses.append("tag = ?")
            params.append(tag)

        return " AND ".join(clauses), tuple(params)

    where_sql, where_params = build_where()

    # ---- KPIs (FIX: cycles counted by distinct (run_id, cycle_num))
    kpi = qdf(
        f"""
        SELECT
          count(*) AS events,
          count(DISTINCT (run_id, cycle_num)) AS cycles,
          sum(CASE WHEN event_type IN ('llm_call','llm_request') THEN 1 ELSE 0 END) AS llm_events,
          sum(CASE WHEN event_type LIKE '%_api_call' THEN 1 ELSE 0 END) AS api_events,
          sum(CASE WHEN event_type IN ('action_executed','action_blocked','action_skipped') THEN 1 ELSE 0 END) AS action_events,
          sum(CASE WHEN event_type IN ('error','llm_exception','external_api_error') THEN 1 ELSE 0 END) AS error_events,
          sum(CASE WHEN http_status = 429 THEN 1 ELSE 0 END) AS rate_limited_429
        FROM events
        WHERE {where_sql}
        """,
        where_params,
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Events", int(kpi.loc[0, "events"] or 0))
    c2.metric("Cycles", int(kpi.loc[0, "cycles"] or 0))
    c3.metric("LLM events", int(kpi.loc[0, "llm_events"] or 0))
    c4.metric("API events", int(kpi.loc[0, "api_events"] or 0))
    c5.metric("Action events", int(kpi.loc[0, "action_events"] or 0))
    c6.metric("Errors", int(kpi.loc[0, "error_events"] or 0))
    c7.metric("429s", int(kpi.loc[0, "rate_limited_429"] or 0))

    st.divider()

    # ---- Charts row
    left, right = st.columns(2)

    # Chart 1: LLM prompt/response chars over time (hourly) — from events
    llm_ts = qdf(
        f"""
        SELECT
          date_trunc('hour', ts) AS hour,
          sum(coalesce(prompt_chars,0)) AS prompt_chars,
          sum(coalesce(response_chars,0)) AS response_chars
        FROM events
        WHERE {where_sql}
          AND event_type = 'llm_call'
        GROUP BY 1
        ORDER BY 1
        """,
        where_params,
    )

    with left:
        st.subheader("LLM volume over time (hourly)")
        if len(llm_ts) == 0:
            st.info("No llm_call rows in this filter window.")
        else:
            fig = plt.figure()
            plt.plot(llm_ts["hour"], llm_ts["prompt_chars"], label="prompt_chars")
            plt.plot(llm_ts["hour"], llm_ts["response_chars"], label="response_chars")
            plt.xticks(rotation=30, ha="right")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)

    # Chart 2: API status breakdown — from api_calls view
    # We apply the same filters by joining back to events-like columns (api_calls already includes them)
    api_status = qdf(
        f"""
        SELECT
          http_status,
          count(*) AS n
        FROM api_calls
        WHERE {where_sql}
          AND http_status IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        """,
        where_params,
    )

    with right:
        st.subheader("API status codes")
        if len(api_status) == 0:
            st.info("No API calls in this filter window.")
        else:
            fig = plt.figure()
            plt.bar(api_status["http_status"].astype(str), api_status["n"])
            plt.tight_layout()
            st.pyplot(fig)

    st.divider()

    # ---- Events per cycle (recent) — use cycle_summary (already grouped by run_id+cycle_num)
    st.subheader("Events per cycle (recent)")
    e_per_cycle = qdf(
        f"""
        SELECT
          run_id,
          cycle_num,
          events_in_cycle
        FROM cycle_summary
        WHERE CAST(cycle_start AS DATE) >= ? AND CAST(cycle_start AS DATE) <= ?

        {'' if brain == '(all)' else 'AND brain = ?'}
        ORDER BY cycle_start DESC
        LIMIT 50
        """,
        (start_dt, end_dt, brain) if brain != "(all)" else (start_dt, end_dt),
    )

    if len(e_per_cycle) == 0:
        st.info("No cycles in this filter window.")
    else:
        # For plotting, create an x label that is unique: last 6 of run_id + cycle_num
        e_per_cycle = e_per_cycle.sort_values(["run_id", "cycle_num"])
        x = [f"{rid[-6:]}:{int(cn)}" for rid, cn in zip(e_per_cycle["run_id"], e_per_cycle["cycle_num"])]
        fig = plt.figure()
        plt.plot(x, e_per_cycle["events_in_cycle"])
        plt.xticks(rotation=60, ha="right")
        plt.xlabel("run_id suffix : cycle_num")
        plt.ylabel("events_in_cycle")
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ---- Recent cycles summary table — from cycle_summary (correct uniqueness)
    st.subheader("Recent cycles (summary)")
    cycles_select = """
      run_id, brain, cycle_num, cycle_start, cycle_end, events_in_cycle,
      llm_calls, api_calls, actions_executed, rate_limited_429,
      prompt_chars, response_chars
    """
    cycles = qdf(
        f"""
        SELECT {cycles_select}
        FROM cycle_summary
        WHERE cycle_dt >= ? AND cycle_dt <= ?
        {'' if brain == '(all)' else 'AND brain = ?'}
        ORDER BY cycle_start DESC
        LIMIT 30
        """,
        (start_dt, end_dt, brain) if brain != "(all)" else (start_dt, end_dt),
    )
    st.dataframe(cycles, use_container_width=True, hide_index=True)

    # ---- Recent errors — from errors view
    st.subheader("Recent errors")
    err_select = """
      ts, dt, brain, run_id, cycle_num, event_type,
      error_type, error_message, http_status, http_path
    """
    if show_payload:
        err_select += ", payload_json"

    errors_df = qdf(
        f"""
        SELECT {err_select}
        FROM errors
        WHERE {where_sql}
        ORDER BY ts DESC
        LIMIT 50
        """,
        where_params,
    )
    st.dataframe(errors_df, use_container_width=True, hide_index=True)

    # ---- Recent events — from events
    st.subheader("Recent events (drilldown)")
    evt_select = """
      ts, dt, brain, run_id, cycle_num, event_type, tag,
      model, latency_ms, http_method, http_path, http_status, action_type
    """
    if show_payload:
        evt_select += ", payload_json"

    recent = qdf(
        f"""
        SELECT {evt_select}
        FROM events
        WHERE {where_sql}
        ORDER BY ts DESC
        LIMIT 200
        """,
        where_params,
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

    if show_payload and len(recent) > 0:
        st.caption("Tip: payload_json can be large—toggle it off for faster rendering.")


if __name__ == "__main__":
    main()
