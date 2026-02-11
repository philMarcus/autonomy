# dashboard_v1_1.py
# Streamlit dashboard for Autonomy telemetry (DuckDB views over Parquet)
# + Dry Run & Feeds viewer (reads JSONL directly, no warehouse needed)
#
# Prereq for Telemetry tab (run once in DuckDB CLI from project root):
#   .read sql/views.sql
#
# Run:
#   streamlit run dashboard_v1_1.py
#
# Notes:
# - Cycles are counted by (run_id, cycle_num) because cycle_num resets each run.
# - Uses cycle_summary.cycle_dt (DATE) for cycle filtering (no dt() function in DuckDB).
# - Robust to NULL run_id rows in cycle_summary (will label as "no_run_id").

from __future__ import annotations

import os
import json
import glob
from typing import Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = "warehouse/telemetry.duckdb"
BRAINS_DIR = os.environ.get("BRAINS_DIR", "brains")


# ============================================================
# DuckDB helpers (for Telemetry tab)
# ============================================================
def _get_duckdb():
    """Lazy import duckdb — only needed for Telemetry tab."""
    import duckdb
    return duckdb


@st.cache_resource
def get_con():
    duckdb = _get_duckdb()
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Missing {DB_PATH}. Create it and run `.read sql/views.sql` in DuckDB first."
        )
    return duckdb.connect(DB_PATH)


def qdf(sql: str, params: Tuple | None = None) -> pd.DataFrame:
    con = get_con()
    if params is None:
        return con.execute(sql).df()
    return con.execute(sql, params).df()


def ensure_views_exist() -> None:
    df = qdf(
        "SELECT count(*) AS n FROM information_schema.views WHERE table_name='events';"
    )
    if int(df.loc[0, "n"]) == 0:
        raise RuntimeError(
            "DuckDB views not found. In DuckDB CLI, run:\n"
            "  .read sql/views.sql\n"
            "and re-run the dashboard."
        )


# ============================================================
# Telemetry tab
# ============================================================
def render_telemetry_tab():
    try:
        ensure_views_exist()
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(str(e))
        return

    meta = qdf("SELECT min(dt) AS min_dt, max(dt) AS max_dt FROM events;")
    min_dt = meta.loc[0, "min_dt"]
    max_dt = meta.loc[0, "max_dt"]

    if min_dt is None or max_dt is None:
        st.warning("No data found in events view.")
        return

    brains_df = qdf(
        """
        SELECT DISTINCT brain
        FROM events
        WHERE brain IS NOT NULL
        ORDER BY brain
        """
    )
    brains = ["(all)"] + brains_df["brain"].tolist()

    st.sidebar.header("Telemetry Filters")

    start_dt, end_dt = st.sidebar.date_input(
        "Date range (dt)",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
    )
    if isinstance(start_dt, (list, tuple)):
        start_dt, end_dt = start_dt[0], start_dt[1]

    brain = st.sidebar.selectbox("Brain", brains, index=0)

    # Event types / tags depend on date range + brain
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

    # ---- KPIs
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

    left, right = st.columns(2)

    # ---- Chart: LLM volume (hourly)
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

    # ---- Chart: API status codes
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

    # ---- Events per cycle (recent)
    st.subheader("Events per cycle (recent)")

    params = [start_dt, end_dt]
    brain_clause = ""
    if brain != "(all)":
        brain_clause = "AND brain = ?"
        params.append(brain)

    e_per_cycle = qdf(
        f"""
        SELECT
          coalesce(run_id, 'no_run_id') AS run_id,
          cycle_num,
          cycle_start,
          events_in_cycle
        FROM cycle_summary
        WHERE cycle_dt >= ? AND cycle_dt <= ?
          {brain_clause}
        ORDER BY cycle_start DESC
        LIMIT 50
        """,
        tuple(params),
    )

    if len(e_per_cycle) == 0:
        st.info("No cycles in this filter window.")
    else:
        e_plot = e_per_cycle.sort_values("cycle_start", ascending=True)
        x_labels = []
        for rid, cn in zip(e_plot["run_id"], e_plot["cycle_num"]):
            rid_s = str(rid) if rid is not None else "no_run_id"
            try:
                cn_i = int(cn) if cn is not None else -1
            except Exception:
                cn_i = -1
            x_labels.append(f"{rid_s[-6:]}:{cn_i}")

        fig = plt.figure()
        plt.plot(x_labels, e_plot["events_in_cycle"])
        plt.xticks(rotation=60, ha="right")
        plt.xlabel("run_id suffix : cycle_num")
        plt.ylabel("events_in_cycle")
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ---- Recent cycles (summary)
    st.subheader("Recent cycles (summary)")
    cycles_select = """
      coalesce(run_id, 'no_run_id') AS run_id,
      brain,
      cycle_num,
      cycle_start,
      cycle_end,
      cycle_dt,
      events_in_cycle,
      llm_calls,
      api_calls,
      actions_executed,
      rate_limited_429,
      prompt_chars,
      response_chars
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

    # ---- Recent errors
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

    # ---- Recent events (drilldown)
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


# ============================================================
# Dry Run & Feeds tab
# ============================================================
def render_dryrun_tab():
    st.header("Dry Run Outputs & Feed Snapshots")

    dryrun_files = sorted(glob.glob(os.path.join(BRAINS_DIR, "*_dryrun.jsonl")))

    if not dryrun_files:
        st.info("No dry-run logs found. Run with `--dry-run` to generate them.")
        return

    # Brain selector (extract brain name from filename)
    file_labels = []
    for f in dryrun_files:
        name = os.path.basename(f).replace("_dryrun.jsonl", "")
        file_labels.append(name)

    selected_idx = st.selectbox("Brain", range(len(file_labels)), format_func=lambda i: file_labels[i])
    selected_file = dryrun_files[selected_idx]

    # Read and parse JSONL
    entries = []
    with open(selected_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        st.info("Log file is empty.")
        return

    st.caption(f"Loaded {len(entries)} entries from `{selected_file}`")

    # Filter by type
    available_types = sorted(set(e.get("type", "unknown") for e in entries))
    type_options = ["(all)"] + available_types
    type_filter = st.selectbox("Entry type", type_options)

    filtered = entries if type_filter == "(all)" else [e for e in entries if e.get("type") == type_filter]

    # KPIs
    action_count = sum(1 for e in entries if e.get("type") == "action")
    feed_count = sum(1 for e in entries if e.get("type") == "feed")
    kernel_count = sum(1 for e in entries if e.get("type") == "kernel_update")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entries", len(entries))
    c2.metric("Actions", action_count)
    c3.metric("Feed snapshots", feed_count)
    c4.metric("Kernel updates", kernel_count)

    st.divider()

    # Overview table
    if filtered:
        df = pd.json_normalize(filtered)
        # Reorder columns: ts, type first
        cols = list(df.columns)
        priority = ["ts", "type", "brain", "cycle", "action"]
        ordered = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        df = df[ordered]
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Action details (expandable)
    actions = [e for e in filtered if e.get("type") == "action"]
    if actions:
        st.subheader(f"Action Details ({len(actions)} total)")
        for a in reversed(actions[-30:]):
            plan = a.get("plan", {})
            action_type = a.get("action") or plan.get("action", "?")
            title = plan.get("title") or plan.get("content", "")
            preview = str(title)[:80] if title else a.get("summary", "")
            with st.expander(f"{a.get('ts', '')} | {action_type} | {preview}"):
                st.json(a)

    # Feed snapshots (expandable)
    feeds = [e for e in filtered if e.get("type") == "feed"]
    if feeds:
        st.subheader(f"Feed Snapshots ({len(feeds)} total)")
        for f_entry in reversed(feeds[-15:]):
            with st.expander(f"Cycle {f_entry.get('cycle', '?')} — {f_entry.get('ts', '')}"):
                st.text(f_entry.get("feed_brief", "(empty)"))

    # Kernel updates (expandable)
    kernels = [e for e in filtered if e.get("type") == "kernel_update"]
    if kernels:
        st.subheader(f"Kernel Updates ({len(kernels)} total)")
        for k in reversed(kernels[-10:]):
            with st.expander(f"Cycle {k.get('cycle', '?')} — {k.get('ts', '')} — {k.get('reason', '')}"):
                st.text(k.get("new_kernel", "(empty)"))


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="Autonomy Dashboard v1.1", layout="wide")
    st.title("Autonomy Dashboard v1.1")

    tab1, tab2 = st.tabs(["Telemetry", "Dry Run & Feeds"])

    with tab1:
        render_telemetry_tab()

    with tab2:
        render_dryrun_tab()


if __name__ == "__main__":
    main()
