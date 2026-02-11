# dashboard_v1_3.py
# Streamlit dashboard: Telemetry tab (DuckDB) + Dry-Run Viewer (.txt)
#
# Changes in v1.3:
# - Fixed: actions_executed now shows correctly (cycle_num populated in telemetry)
# - Fixed: error_message column populated (ingest reads "error" field)
# - Added: Actions breakdown section (by action type)
# - Added: action_type column in Recent events drilldown
# - Improved: Cycle count KPI uses cycle_start events (not DISTINCT tuple which breaks on NULLs)
# - Added: seq column for deterministic event ordering within same-second timestamps
# - Improved: Recent events/errors ORDER BY ts DESC, seq DESC
#
# Prereq for Telemetry tab (run once in DuckDB CLI from project root):
#   .read sql/views.sql
#
# Run:
#   streamlit run dashboard_v1_3.py

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List, Tuple

import streamlit as st

DB_PATH = "warehouse/telemetry.duckdb"
BRAINS_DIR = os.environ.get("BRAINS_DIR", "brains")


# ============================================================
# DuckDB helpers (for Telemetry tab — lazy imports)
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


def qdf(sql: str, params: Tuple | None = None):
    import pandas as pd
    con = get_con()
    if params is None:
        return con.execute(sql).df()
    return con.execute(sql, params).df()


def ensure_views_exist() -> None:
    """Recreate all views so they stay in sync with the parquet schema."""
    con = get_con()
    con.execute("""
        CREATE OR REPLACE VIEW events AS
        SELECT * FROM read_parquet('warehouse/events/dt=*/events_*.parquet');
    """)
    con.execute("""
        CREATE OR REPLACE VIEW llm_calls AS
        SELECT * FROM events WHERE event_type IN ('llm_call', 'llm_request');
    """)
    con.execute("""
        CREATE OR REPLACE VIEW api_calls AS
        SELECT * FROM events WHERE event_type LIKE '%_api_call';
    """)
    con.execute("""
        CREATE OR REPLACE VIEW actions AS
        SELECT * FROM events WHERE event_type IN ('action_executed', 'action_blocked', 'action_skipped');
    """)
    con.execute("""
        CREATE OR REPLACE VIEW errors AS
        SELECT * FROM events WHERE event_type IN ('error', 'llm_exception', 'external_api_error');
    """)
    con.execute("""
        CREATE OR REPLACE VIEW cycle_summary AS
        SELECT
          run_id, brain, cycle_num,
          min(ts) AS cycle_start, max(ts) AS cycle_end,
          CAST(min(ts) AS DATE) AS cycle_dt,
          count(*) AS events_in_cycle,
          sum(CASE WHEN event_type = 'llm_call' THEN coalesce(prompt_chars,0) ELSE 0 END) AS prompt_chars,
          sum(CASE WHEN event_type = 'llm_call' THEN coalesce(response_chars,0) ELSE 0 END) AS response_chars,
          sum(CASE WHEN event_type = 'llm_call' THEN 1 ELSE 0 END) AS llm_calls,
          sum(CASE WHEN event_type LIKE '%_api_call' THEN 1 ELSE 0 END) AS api_calls,
          sum(CASE WHEN http_status = 429 THEN 1 ELSE 0 END) AS rate_limited_429,
          sum(CASE WHEN event_type = 'action_executed' THEN 1 ELSE 0 END) AS actions_executed
        FROM events
        WHERE cycle_num IS NOT NULL
        GROUP BY 1,2,3;
    """)


# ============================================================
# Telemetry tab
# ============================================================
def render_telemetry_tab():
    import matplotlib.pyplot as plt

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
    # Use cycle_start count for cycles (robust even when cycle_num is NULL on some events)
    kpi = qdf(
        f"""
        SELECT
          count(*) AS events,
          sum(CASE WHEN event_type = 'cycle_start' THEN 1 ELSE 0 END) AS cycles,
          sum(CASE WHEN event_type IN ('llm_call','llm_request') THEN 1 ELSE 0 END) AS llm_events,
          sum(CASE WHEN event_type LIKE '%_api_call' THEN 1 ELSE 0 END) AS api_events,
          sum(CASE WHEN event_type = 'action_executed' THEN 1 ELSE 0 END) AS actions_executed,
          sum(CASE WHEN event_type IN ('action_blocked','action_skipped') THEN 1 ELSE 0 END) AS actions_blocked,
          sum(CASE WHEN event_type IN ('error','llm_exception','external_api_error') THEN 1 ELSE 0 END) AS error_events,
          sum(CASE WHEN http_status = 429 THEN 1 ELSE 0 END) AS rate_limited_429
        FROM events
        WHERE {where_sql}
        """,
        where_params,
    )

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Events", int(kpi.loc[0, "events"] or 0))
    c2.metric("Cycles", int(kpi.loc[0, "cycles"] or 0))
    c3.metric("LLM", int(kpi.loc[0, "llm_events"] or 0))
    c4.metric("API", int(kpi.loc[0, "api_events"] or 0))
    c5.metric("Actions", int(kpi.loc[0, "actions_executed"] or 0))
    c6.metric("Blocked", int(kpi.loc[0, "actions_blocked"] or 0))
    c7.metric("Errors", int(kpi.loc[0, "error_events"] or 0))
    c8.metric("429s", int(kpi.loc[0, "rate_limited_429"] or 0))

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

    # ---- Actions breakdown (by action type)
    actions_by_type = qdf(
        f"""
        SELECT
          coalesce(action_type, '(unknown)') AS action_type,
          count(*) AS n
        FROM events
        WHERE {where_sql}
          AND event_type = 'action_executed'
        GROUP BY 1
        ORDER BY 2 DESC
        """,
        where_params,
    )

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Actions by type")
        if len(actions_by_type) == 0:
            st.info("No executed actions in this filter window.")
        else:
            fig = plt.figure()
            plt.barh(actions_by_type["action_type"], actions_by_type["n"])
            plt.xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig)

    # ---- Actions over time (hourly)
    actions_ts = qdf(
        f"""
        SELECT
          date_trunc('hour', ts) AS hour,
          count(*) AS n
        FROM events
        WHERE {where_sql}
          AND event_type = 'action_executed'
        GROUP BY 1
        ORDER BY 1
        """,
        where_params,
    )

    with right2:
        st.subheader("Actions over time (hourly)")
        if len(actions_ts) == 0:
            st.info("No executed actions in this filter window.")
        else:
            fig = plt.figure()
            plt.plot(actions_ts["hour"], actions_ts["n"])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Actions")
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

    cycles_df = qdf(
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
    st.dataframe(cycles_df, use_container_width=True, hide_index=True)

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
        ORDER BY ts DESC, seq DESC
        LIMIT 50
        """,
        where_params,
    )
    st.dataframe(errors_df, use_container_width=True, hide_index=True)

    # ---- Recent events (drilldown)
    st.subheader("Recent events (drilldown)")
    evt_select = """
      ts, seq, dt, brain, run_id, cycle_num, event_type, tag,
      model, latency_ms, http_method, http_path, http_status, action_type
    """
    if show_payload:
        evt_select += ", payload_json"

    recent = qdf(
        f"""
        SELECT {evt_select}
        FROM events
        WHERE {where_sql}
        ORDER BY ts DESC, seq DESC
        LIMIT 200
        """,
        where_params,
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

    if show_payload and len(recent) > 0:
        st.caption("Tip: payload_json can be large - toggle it off for faster rendering.")


# ============================================================
# Dry-Run .txt Parsing
# ============================================================
@dataclass
class CycleSection:
    name: str       # e.g. "SOCIAL ACTIONS", "FEED", "PLANNER OUTPUT"
    content: str    # raw text content

@dataclass
class Cycle:
    number: int
    timestamp: str
    sections: List[CycleSection] = field(default_factory=list)


_CYCLE_RE = re.compile(r"={64}\nCYCLE (\d+) \| (.+?)\n={64}")
_SECTION_RE = re.compile(r"^--- (.+?) ---$", re.MULTILINE)


def parse_dryrun_txt(text: str) -> List[Cycle]:
    cycles: List[Cycle] = []
    parts = _CYCLE_RE.split(text)
    # parts layout: [pre, num, ts, body, num, ts, body, ...]
    i = 1
    while i + 2 < len(parts):
        cycle_num = int(parts[i])
        timestamp = parts[i + 1].strip()
        body = parts[i + 2]

        sections: List[CycleSection] = []
        sec_parts = _SECTION_RE.split(body)
        # sec_parts: [pre, name, content, name, content, ...]
        j = 1
        while j + 1 < len(sec_parts):
            sec_name = sec_parts[j].strip()
            sec_content = sec_parts[j + 1].strip()
            if sec_content:
                sections.append(CycleSection(name=sec_name, content=sec_content))
            j += 2

        cycles.append(Cycle(number=cycle_num, timestamp=timestamp, sections=sections))
        i += 3

    return cycles


# ============================================================
# Dry-Run Viewer tab
# ============================================================
def render_dryrun_tab():
    txt_files = sorted(glob.glob(os.path.join(BRAINS_DIR, "*_dryrun.txt")))
    if not txt_files:
        st.info("No dry-run .txt logs found. Run with `--dry-run` to generate them.")
        return

    brain_names = [os.path.basename(f).replace("_dryrun.txt", "") for f in txt_files]
    selected_idx = st.sidebar.selectbox(
        "Brain (dry-run)", range(len(brain_names)), format_func=lambda i: brain_names[i]
    )
    selected_file = txt_files[selected_idx]

    # Read and parse
    with open(selected_file, encoding="utf-8") as f:
        raw_text = f.read()

    cycles = parse_dryrun_txt(raw_text)

    if not cycles:
        st.info("Log file is empty or has no cycles yet.")
        return

    # KPI metrics
    total_cycles = len(cycles)
    total_social = sum(1 for c in cycles for s in c.sections if s.name == "SOCIAL ACTIONS")
    total_planner = sum(1 for c in cycles for s in c.sections if s.name == "PLANNER OUTPUT")
    total_kernel = sum(1 for c in cycles for s in c.sections if s.name == "KERNEL UPDATE PROPOSAL")
    total_feed = sum(1 for c in cycles for s in c.sections if s.name == "FEED")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cycles", total_cycles)
    c2.metric("Feeds", total_feed)
    c3.metric("Social sections", total_social)
    c4.metric("Planner outputs", total_planner)
    c5.metric("Kernel proposals", total_kernel)

    st.divider()

    # Sidebar filters
    cycle_nums = [c.number for c in cycles]
    min_cycle, max_cycle = min(cycle_nums), max(cycle_nums)

    range_start, range_end = st.sidebar.slider(
        "Cycle range",
        min_value=min_cycle,
        max_value=max_cycle,
        value=(max(min_cycle, max_cycle - 19), max_cycle),
    )

    all_section_names = sorted(set(s.name for c in cycles for s in c.sections))
    show_sections = st.sidebar.multiselect(
        "Show sections", all_section_names, default=all_section_names
    )

    filtered = [c for c in cycles if range_start <= c.number <= range_end]

    # Display cycles (newest first)
    for cycle in reversed(filtered):
        visible_sections = [s for s in cycle.sections if s.name in show_sections]
        if not visible_sections:
            continue

        label = f"Cycle {cycle.number} | {cycle.timestamp}"
        # Show brief preview of action if available
        for s in cycle.sections:
            if s.name == "PLANNER OUTPUT":
                first_line = s.content.split("\n")[0] if s.content else ""
                label += f"  ---  {first_line}"
                break

        with st.expander(label, expanded=False):
            for section in visible_sections:
                # FEED and SOCIAL ACTIONS collapsed by default to reduce noise
                if section.name in ("FEED", "SOCIAL ACTIONS"):
                    with st.expander(f"--- {section.name} ---", expanded=False):
                        st.text(section.content)
                elif section.name in ("PLANNER OUTPUT", "KERNEL UPDATE PROPOSAL"):
                    st.markdown(f"**--- {section.name} ---**")
                    st.code(section.content, language=None)
                else:
                    st.markdown(f"**--- {section.name} ---**")
                    st.text(section.content)


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="Autonomy Dashboard v1.3", layout="wide")
    st.title("Autonomy Dashboard v1.3")

    tab1, tab2 = st.tabs(["Telemetry", "Dry-Run Viewer"])

    with tab1:
        render_telemetry_tab()

    with tab2:
        render_dryrun_tab()


if __name__ == "__main__":
    main()
