# dashboard_v2_1.py
# Streamlit dashboard: Overview + Cycle Replay + Daemon Monitor + Controls
#
# Run:
#   streamlit run dashboard_v2_1.py

from __future__ import annotations

import json
import os
import glob
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import streamlit as st

DB_PATH = "warehouse/telemetry.duckdb"
BRAINS_DIR = os.environ.get("BRAINS_DIR", "brains")
TELEMETRY_DIR = os.environ.get("TELEMETRY_DIR", "telemetry")


# ============================================================
# Auto-ingest on startup + manual refresh
# ============================================================
def run_ingest() -> str:
    """Run ingest.py and return its output."""
    try:
        from ingest import ingest_once
        lines, events = ingest_once()
        return f"Ingested {lines} lines, {events} events."
    except Exception as e:
        return f"Ingest error: {e}"


def auto_ingest_on_startup():
    """Run ingest once per Streamlit session (on first load)."""
    if "ingest_done" not in st.session_state:
        result = run_ingest()
        st.session_state["ingest_done"] = True
        st.session_state["last_ingest"] = result


# ============================================================
# Brain discovery
# ============================================================
def discover_brains() -> list[str]:
    """Find brain names from kernel prompt files in brains/ dir."""
    pattern = os.path.join(BRAINS_DIR, "*_kernel_prompt.txt")
    files = glob.glob(pattern)
    names = sorted(
        os.path.basename(f).replace("_kernel_prompt.txt", "") for f in files
    )
    if not names:
        # fallback: check for *_kernel.txt (backward compat)
        alt = glob.glob(os.path.join(BRAINS_DIR, "*_kernel.txt"))
        names = sorted(
            os.path.basename(f).replace("_kernel.txt", "") for f in alt
        )
    return names


# ============================================================
# DuckDB helpers (for Overview tab)
# ============================================================
def _get_duckdb():
    import duckdb
    return duckdb


@st.cache_resource
def get_con():
    duckdb = _get_duckdb()
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Missing {DB_PATH}. Run `python ingest.py` first."
        )
    return duckdb.connect(DB_PATH)


def qdf(sql: str, params: Tuple | None = None):
    import pandas as pd
    con = get_con()
    if params is None:
        return con.execute(sql).df()
    return con.execute(sql, params).df()


def ensure_views_exist() -> None:
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
# JSONL reader (for Cycle Replay tab)
# ============================================================
def read_brain_events(brain_name: str) -> list[dict]:
    """Read all events from per-brain JSONL file. Also checks legacy events.jsonl."""
    events = []

    # Per-brain file (new format)
    per_brain = os.path.join(TELEMETRY_DIR, f"{brain_name}_events.jsonl")
    if os.path.exists(per_brain):
        with open(per_brain, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Legacy shared file
    legacy = os.path.join(TELEMETRY_DIR, "events.jsonl")
    if os.path.exists(legacy):
        with open(legacy, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        evt = json.loads(line)
                        if evt.get("brain") == brain_name:
                            events.append(evt)
                    except json.JSONDecodeError:
                        pass

    # Sort by seq (or ts as fallback)
    events.sort(key=lambda e: (e.get("ts", ""), e.get("seq", 0)))
    return events


# ============================================================
# Tab 1: Overview
# ============================================================
def render_overview_tab(brain_filter: str):
    import pandas as pd

    try:
        ensure_views_exist()
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(str(e))
        return

    meta = qdf("SELECT min(dt) AS min_dt, max(dt) AS max_dt FROM events;")
    min_dt = meta.loc[0, "min_dt"]
    max_dt = meta.loc[0, "max_dt"]

    if min_dt is None or max_dt is None:
        st.warning("No data found. Run `python ingest.py` to populate warehouse.")
        return

    st.sidebar.header("Filters")

    start_dt, end_dt = st.sidebar.date_input(
        "Date range",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
    )
    if isinstance(start_dt, (list, tuple)):
        start_dt, end_dt = start_dt[0], start_dt[1]

    show_payload = st.sidebar.checkbox("Show payload_json", value=False)

    # Build WHERE clause
    def build_where() -> tuple[str, tuple]:
        clauses = ["dt >= ?", "dt <= ?"]
        params: list = [start_dt, end_dt]
        if brain_filter != "(all)":
            clauses.append("brain = ?")
            params.append(brain_filter)
        return " AND ".join(clauses), tuple(params)

    where_sql, where_params = build_where()

    # ---- KPIs
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

    # ---- Charts (compact, native Streamlit)
    left, right = st.columns(2)

    # LLM volume (hourly)
    llm_ts = qdf(
        f"""
        SELECT
          date_trunc('hour', ts) AS hour,
          sum(coalesce(prompt_chars,0)) AS prompt_chars,
          sum(coalesce(response_chars,0)) AS response_chars
        FROM events
        WHERE {where_sql} AND event_type = 'llm_call'
        GROUP BY 1 ORDER BY 1
        """,
        where_params,
    )
    with left:
        st.caption("LLM volume (hourly)")
        if len(llm_ts) == 0:
            st.info("No LLM calls in range.")
        else:
            chart_df = llm_ts.set_index("hour")[["prompt_chars", "response_chars"]]
            st.line_chart(chart_df, height=200)

    # API status codes
    api_status = qdf(
        f"""
        SELECT http_status::VARCHAR AS status, count(*) AS n
        FROM api_calls
        WHERE {where_sql} AND http_status IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC
        """,
        where_params,
    )
    with right:
        st.caption("API status codes")
        if len(api_status) == 0:
            st.info("No API calls in range.")
        else:
            chart_df = api_status.set_index("status")
            st.bar_chart(chart_df, height=200)

    left2, right2 = st.columns(2)

    # Actions by type
    actions_by_type = qdf(
        f"""
        SELECT coalesce(action_type, '(unknown)') AS action_type, count(*) AS n
        FROM events
        WHERE {where_sql} AND event_type = 'action_executed'
        GROUP BY 1 ORDER BY 2 DESC
        """,
        where_params,
    )
    with left2:
        st.caption("Actions by type")
        if len(actions_by_type) == 0:
            st.info("No actions in range.")
        else:
            chart_df = actions_by_type.set_index("action_type")
            st.bar_chart(chart_df, height=200)

    # Actions over time (hourly)
    actions_ts = qdf(
        f"""
        SELECT date_trunc('hour', ts) AS hour, count(*) AS actions
        FROM events
        WHERE {where_sql} AND event_type = 'action_executed'
        GROUP BY 1 ORDER BY 1
        """,
        where_params,
    )
    with right2:
        st.caption("Actions over time (hourly)")
        if len(actions_ts) == 0:
            st.info("No actions in range.")
        else:
            chart_df = actions_ts.set_index("hour")
            st.line_chart(chart_df, height=200)

    # Events per cycle
    brain_clause = "" if brain_filter == "(all)" else "AND brain = ?"
    epc_params = [start_dt, end_dt] + ([brain_filter] if brain_filter != "(all)" else [])
    e_per_cycle = qdf(
        f"""
        SELECT cycle_num, events_in_cycle, cycle_start
        FROM cycle_summary
        WHERE cycle_dt >= ? AND cycle_dt <= ? {brain_clause}
        ORDER BY cycle_start DESC
        LIMIT 50
        """,
        tuple(epc_params),
    )
    st.caption("Events per cycle (recent 50)")
    if len(e_per_cycle) == 0:
        st.info("No cycles in range.")
    else:
        e_plot = e_per_cycle.sort_values("cycle_start", ascending=True)
        chart_df = e_plot.set_index("cycle_num")[["events_in_cycle"]]
        st.line_chart(chart_df, height=200)

    st.divider()

    # ---- Data tables (collapsed)
    with st.expander("Recent cycles (summary)", expanded=False):
        cycles_df = qdf(
            f"""
            SELECT run_id, brain, cycle_num, cycle_start, cycle_end,
                   events_in_cycle, llm_calls, api_calls, actions_executed,
                   rate_limited_429, prompt_chars, response_chars
            FROM cycle_summary
            WHERE cycle_dt >= ? AND cycle_dt <= ? {brain_clause}
            ORDER BY cycle_start DESC LIMIT 30
            """,
            tuple(epc_params),
        )
        st.dataframe(cycles_df, use_container_width=True, hide_index=True)

    with st.expander("Recent errors", expanded=False):
        err_select = "ts, dt, brain, run_id, cycle_num, event_type, error_type, error_message, http_status, http_path"
        if show_payload:
            err_select += ", payload_json"
        errors_df = qdf(
            f"""
            SELECT {err_select} FROM errors
            WHERE {where_sql}
            ORDER BY ts DESC, seq DESC LIMIT 50
            """,
            where_params,
        )
        st.dataframe(errors_df, use_container_width=True, hide_index=True)

    with st.expander("Recent events (all)", expanded=False):
        evt_select = "ts, seq, brain, run_id, cycle_num, event_type, tag, model, latency_ms, http_method, http_path, http_status, action_type"
        if show_payload:
            evt_select += ", payload_json"
        recent = qdf(
            f"""
            SELECT {evt_select} FROM events
            WHERE {where_sql}
            ORDER BY ts DESC, seq DESC LIMIT 200
            """,
            where_params,
        )
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ============================================================
# Tab 2: Cycle Replay
# ============================================================
_EVENT_STYLES = {
    "run_start": ("RUN START", "#00ffd5"),
    "cycle_start": ("CYCLE START", "#00ffd5"),
    "cycle_end": ("CYCLE END", "#666"),
    "feed_context": ("FEED", "#39ff14"),
    "planner_decision": ("PLANNER", "#ff00ff"),
    "grounding_metadata": ("GROUNDING", "#00997f"),
    "kernel_update_executed": ("KERNEL UPDATE", "#ff4444"),
    "kernel_snapshot": ("KERNEL SNAPSHOT", "#ff8800"),
    "action_executed": ("ACTION", "#39ff14"),
    "action_blocked": ("BLOCKED", "#ff4444"),
    "artifact_published": ("PUBLISHED", "#00ffd5"),
    "llm_call": ("LLM CALL", "#888"),
    "moltbook_api_call": ("API", "#555"),
    "analog_home_api_call": ("API", "#555"),
    "error": ("ERROR", "#ff4444"),
    "llm_exception": ("LLM ERROR", "#ff4444"),
}


def _render_event(evt: dict):
    """Render a single telemetry event as styled markdown."""
    etype = evt.get("event_type", "unknown")
    label, color = _EVENT_STYLES.get(etype, (etype.upper(), "#888"))
    ts = evt.get("ts", "")
    seq = evt.get("seq", "")

    st.markdown(
        f'<span style="color:{color};font-weight:bold;font-size:13px">[{label}]</span>'
        f' <span style="color:#666;font-size:11px">{ts} seq={seq}</span>',
        unsafe_allow_html=True,
    )

    if etype == "cycle_start":
        temp = evt.get("temperature", "?")
        model = evt.get("model", "?")
        st.markdown(f"Temperature: `{temp}` | Model: `{model}`")

    elif etype == "feed_context":
        brief = evt.get("brief", evt.get("text", ""))
        text_len = evt.get("text_length", len(brief))
        with st.expander(f"Feed context ({text_len} chars)", expanded=False):
            st.text(brief[:3000])

    elif etype == "planner_decision":
        preamble = evt.get("preamble", "")
        action = evt.get("action", "")
        plan = evt.get("plan", {})
        if preamble:
            with st.expander("Reasoning / Preamble", expanded=True):
                st.markdown(preamble)
        st.markdown(f"**Action:** `{action}`")
        if plan:
            with st.expander("Full plan JSON", expanded=False):
                st.json(plan)

    elif etype == "grounding_metadata":
        sources = evt.get("sources", [])
        if sources:
            for src in sources:
                title = src.get("title", "?")
                uri = src.get("uri", "")
                st.markdown(f"- [{title}]({uri})" if uri else f"- {title}")
        queries = evt.get("search_queries", [])
        if queries:
            st.markdown("Search queries: " + ", ".join(f"`{q}`" for q in queries))

    elif etype == "kernel_update_executed":
        reason = evt.get("reason", "")
        st.warning(f"Kernel update: {reason}")

    elif etype in ("action_executed", "action_blocked"):
        action_type = evt.get("action_type", evt.get("type", "?"))
        result = evt.get("action_result", evt.get("result", ""))
        st.markdown(f"Type: `{action_type}` | Result: {result}")

    elif etype == "artifact_published":
        title = evt.get("title", "?")
        art_type = evt.get("artifact_type", "?")
        st.markdown(f"Title: **{title}** | Type: `{art_type}`")

    elif etype == "llm_call":
        model = evt.get("model", "?")
        prompt_c = evt.get("prompt_chars", "?")
        resp_c = evt.get("response_chars", "?")
        latency = evt.get("latency_ms", "?")
        st.markdown(f"`{model}` | prompt: {prompt_c}c | response: {resp_c}c | {latency}ms")

    elif etype in ("moltbook_api_call", "analog_home_api_call"):
        method = evt.get("method", "?")
        path = evt.get("path", "?")
        status = evt.get("status", "?")
        latency = evt.get("latency_ms", "?")
        st.markdown(f"`{method} {path}` -> {status} ({latency}ms)")

    elif etype in ("error", "llm_exception", "external_api_error"):
        msg = evt.get("error_message", evt.get("error", evt.get("message", "")))
        st.error(msg[:500] if msg else "Unknown error")

    elif etype == "cycle_end":
        duration = evt.get("duration_seconds", "?")
        st.markdown(f"Duration: {duration}s")

    elif etype == "run_start":
        version = evt.get("version", "?")
        model = evt.get("model", "?")
        temp = evt.get("temperature", "?")
        st.markdown(f"Version: `{version}` | Model: `{model}` | Temperature: `{temp}`")

    else:
        # Generic: show all non-meta keys
        skip = {"ts", "seq", "brain", "run_id", "event_type", "cycle"}
        extra = {k: v for k, v in evt.items() if k not in skip and v}
        if extra:
            with st.expander("Details", expanded=False):
                st.json(extra)


def render_cycle_replay_tab(brain_filter: str):
    if brain_filter == "(all)":
        st.info("Select a specific brain in the sidebar to use Cycle Replay.")
        return

    events = read_brain_events(brain_filter)
    if not events:
        st.warning(f"No JSONL events found for brain '{brain_filter}'.")
        return

    # Group by run_id
    runs: dict[str, list[dict]] = {}
    for evt in events:
        rid = evt.get("run_id", "unknown")
        runs.setdefault(rid, []).append(evt)

    # Build run list with timestamps
    run_options = []
    for rid, evts in runs.items():
        first_ts = evts[0].get("ts", "?")
        run_options.append((rid, first_ts))
    run_options.sort(key=lambda x: x[1], reverse=True)

    if not run_options:
        st.warning("No runs found.")
        return

    run_labels = [f"{ts[:19]} ({rid[:8]}...)" for rid, ts in run_options]
    selected_run_idx = st.selectbox("Run", range(len(run_labels)), format_func=lambda i: run_labels[i])
    selected_run_id = run_options[selected_run_idx][0]

    run_events = runs[selected_run_id]

    # Find cycles in this run
    cycle_nums = sorted(set(
        evt.get("cycle", evt.get("cycle_num"))
        for evt in run_events
        if evt.get("cycle") is not None or evt.get("cycle_num") is not None
    ))

    # Show run-level events (cycle=None) and per-cycle events
    run_level = [e for e in run_events if e.get("cycle") is None and e.get("cycle_num") is None]

    if run_level:
        with st.expander(f"Run-level events ({len(run_level)})", expanded=False):
            for evt in run_level:
                _render_event(evt)
                st.markdown("---")

    if not cycle_nums:
        st.info("No cycles in this run.")
        return

    # Cycle selector
    if len(cycle_nums) > 1:
        selected_cycle = st.select_slider(
            "Cycle",
            options=cycle_nums,
            value=cycle_nums[-1],
        )
    else:
        selected_cycle = cycle_nums[0]
        st.markdown(f"**Cycle {selected_cycle}**")

    # Filter events for selected cycle
    cycle_events = [
        e for e in run_events
        if (e.get("cycle") == selected_cycle or e.get("cycle_num") == selected_cycle)
    ]

    if not cycle_events:
        st.info(f"No events for cycle {selected_cycle}.")
        return

    st.markdown(f"**{len(cycle_events)} events in cycle {selected_cycle}**")
    st.divider()

    for evt in cycle_events:
        _render_event(evt)
        st.markdown("---")


# ============================================================
# Tab 3: Controls (merged text editors + controls manager)
# ============================================================
def render_controls_tab(brain_filter: str):
    if brain_filter == "(all)":
        st.info("Select a specific brain in the sidebar.")
        return

    st.subheader(f"Controls — {brain_filter}")

    # File paths
    state_path = os.path.join(BRAINS_DIR, f"{brain_filter}_memories.json")
    kernel_path = os.path.join(BRAINS_DIR, f"{brain_filter}_kernel_prompt.txt")
    knowledge_path = os.path.join(BRAINS_DIR, f"{brain_filter}_knowledge.txt")
    controls_path = os.path.join(BRAINS_DIR, f"{brain_filter}_controls.json")

    if not os.path.exists(kernel_path):
        alt = os.path.join(BRAINS_DIR, f"{brain_filter}_kernel.txt")
        if os.path.exists(alt):
            kernel_path = alt

    col_left, col_right = st.columns(2)

    # ================================================================
    # LEFT COLUMN — Text editors
    # ================================================================
    with col_left:
        st.markdown("### Directive")
        st.caption("Guides the agent's behavior. Takes effect next cycle.")

        current_state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    current_state = json.load(f)
            except (json.JSONDecodeError, OSError):
                st.warning(f"Could not read {state_path}")

        current_directive = current_state.get("directive", "Participate on Moltbook.")
        new_directive = st.text_area("Directive", value=current_directive, height=80, key="directive_editor")

        if st.button("Save directive", key="save_directive"):
            current_state["directive"] = new_directive
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(current_state, f, indent=2, ensure_ascii=False)
            st.success("Directive saved.")

        st.divider()

        st.markdown("### Kernel Prompt")
        st.caption("Core identity prompt. Takes effect on restart.")

        current_kernel = ""
        if os.path.exists(kernel_path):
            with open(kernel_path, encoding="utf-8") as f:
                current_kernel = f.read()

        new_kernel = st.text_area("Kernel prompt", value=current_kernel, height=300, key="kernel_editor")

        if st.button("Save kernel", key="save_kernel"):
            with open(kernel_path, "w", encoding="utf-8") as f:
                f.write(new_kernel)
            st.success("Kernel saved.")

        st.divider()

        st.markdown("### Knowledge File")
        st.caption("Reference knowledge for planner context. Takes effect on restart.")

        current_knowledge = ""
        if os.path.exists(knowledge_path):
            with open(knowledge_path, encoding="utf-8") as f:
                current_knowledge = f.read()

        new_knowledge = st.text_area("Knowledge", value=current_knowledge, height=200, key="knowledge_editor")

        if st.button("Save knowledge", key="save_knowledge"):
            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write(new_knowledge)
            st.success("Knowledge saved.")

    # ================================================================
    # RIGHT COLUMN — Controls Manager
    # ================================================================
    with col_right:
        st.markdown("### Controls Manager")
        st.caption(
            "Set values and toggle agent write-access. "
            "Locked controls are visible to the agent but read-only. "
            "Saved changes take effect next cycle (controls re-read from disk each cycle)."
        )

        # Load current controls.json
        current_ctrl: dict = {}
        locked_set: set = set()
        if os.path.exists(controls_path):
            try:
                with open(controls_path, encoding="utf-8") as f:
                    current_ctrl = json.load(f)
                locked_set = set(current_ctrl.get("_locked", []))
            except (json.JSONDecodeError, OSError):
                st.warning(f"Could not read {controls_path} — using defaults.")

        # Group controls by category
        cat_groups: dict = {}
        for meta in CONTROLS_META:
            cat = meta[4]
            cat_groups.setdefault(cat, []).append(meta)

        new_values: dict = {}
        new_locked: set = set()

        for cat in CATEGORY_ORDER:
            if cat not in cat_groups:
                continue
            label = CATEGORY_LABELS.get(cat, cat.title())
            with st.expander(f"**{label}**", expanded=(cat in ("llm", "output", "social"))):
                for meta in cat_groups[cat]:
                    key, typ, default, desc, _cat, mn, mx, choices = meta
                    current_val = current_ctrl.get(key, default)

                    # Build help text with CLI flag hint
                    cli_flag = CLI_FLAGS.get(key)
                    help_text = desc
                    if cli_flag:
                        help_text += f"  \nCLI override: `{cli_flag}`"

                    col_w, col_l = st.columns([5, 1])
                    with col_w:
                        if choices:
                            safe_val = current_val if current_val in choices else choices[0]
                            val = st.selectbox(
                                key, choices, index=choices.index(safe_val),
                                help=help_text, key=f"ctrl_{key}"
                            )
                        elif typ == "bool":
                            val = st.checkbox(key, value=bool(current_val), help=help_text, key=f"ctrl_{key}")
                        elif typ == "float":
                            try:
                                fval = float(current_val)
                            except (TypeError, ValueError):
                                fval = float(default)
                            val = st.number_input(
                                key, value=fval,
                                min_value=float(mn) if mn is not None else None,
                                max_value=float(mx) if mx is not None else None,
                                step=0.01, format="%.3f", help=help_text, key=f"ctrl_{key}"
                            )
                        elif typ == "int":
                            try:
                                ival = int(current_val)
                            except (TypeError, ValueError):
                                ival = int(default)
                            val = st.number_input(
                                key, value=ival,
                                min_value=int(mn) if mn is not None else None,
                                max_value=int(mx) if mx is not None else None,
                                step=1, help=help_text, key=f"ctrl_{key}"
                            )
                        else:
                            val = st.text_input(key, value=str(current_val), help=help_text, key=f"ctrl_{key}")
                        new_values[key] = val

                    with col_l:
                        st.write("")  # vertical alignment
                        agent_writable = st.checkbox(
                            "Unlocked", value=(key not in locked_set),
                            key=f"lock_{key}",
                            help="Uncheck to lock — agent can see but not change."
                        )
                        if not agent_writable:
                            new_locked.add(key)

        col_save, col_info = st.columns([2, 5])
        with col_save:
            if st.button("Save controls", type="primary", key="save_controls_btn"):
                to_save = dict(new_values)
                if new_locked:
                    to_save["_locked"] = sorted(new_locked)
                with open(controls_path, "w", encoding="utf-8") as f:
                    json.dump(to_save, f, indent=2, ensure_ascii=False)
                st.success("Controls saved.")
                st.rerun()
        with col_info:
            if new_locked:
                st.info(f"Locked: {', '.join(sorted(new_locked))}")

        with st.expander("Raw controls.json", expanded=False):
            if os.path.exists(controls_path):
                with open(controls_path, encoding="utf-8") as f:
                    st.code(f.read(), language="json")
            else:
                st.caption("File not yet created — will be written on first Save.")


# ============================================================
# Tab 4: Daemon Monitor — reads directly from JSONL
# ============================================================
_DAEMON_EVENT_TYPES = {
    "daemon_start", "daemon_stop", "daemon_tick", "daemon_error",
    "daemon_wake", "daemon_directives",
    "sentry_signal", "sentry_score_error",
    "strategist_draft", "strategist_error", "strategist_parse_fail",
    "strategist_budget_skip",
}


def _load_daemon_events(brain: str, max_lines: int = 2000) -> list[dict]:
    """Read daemon-related events from a brain's JSONL telemetry file."""
    path = os.path.join(TELEMETRY_DIR, f"{brain}_events.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if evt.get("event_type") in _DAEMON_EVENT_TYPES:
                events.append(evt)
    except Exception:
        pass
    return events


def render_daemon_tab(brain_filter: str):
    import pandas as pd

    if brain_filter == "(all)":
        st.info("Select a specific brain in the sidebar to view daemon activity.")
        return

    events = _load_daemon_events(brain_filter)
    if not events:
        st.info(f"No daemon events for **{brain_filter}**. "
                "Run v15.5 with `--subconscious` to enable the daemon.")
        return

    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # --- KPIs ---
    ticks = df[df["event_type"] == "daemon_tick"]
    signals = df[df["event_type"] == "sentry_signal"]
    drafts = df[df["event_type"] == "strategist_draft"]
    wakes = df[df["event_type"] == "daemon_wake"]
    errors = df[df["event_type"] == "daemon_error"]
    starts = df[df["event_type"] == "daemon_start"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ticks", len(ticks))
    c2.metric("Scanned", int(ticks["items_scanned"].sum()) if "items_scanned" in ticks.columns and len(ticks) > 0 else 0)
    c3.metric("Signals", len(signals))
    c4.metric("Drafts", len(drafts))
    c5.metric("Wakes", len(wakes))
    c6.metric("Errors", len(errors))

    # Model + interval caption
    if len(starts) > 0:
        latest = starts.iloc[-1]
        st.caption(
            f"Model: **{latest.get('model', '?')}** | "
            f"Interval: **{latest.get('sentry_interval', '?')}s** | "
            f"Run: `{latest.get('run_id', '?')[:12]}...`"
        )

    st.divider()

    # --- Charts (compact, native Streamlit) ---
    if len(ticks) > 0 and "wake_potential" in ticks.columns:
        left, right = st.columns(2)

        with left:
            st.caption("Wake potential over time")
            wake_df = ticks.set_index("ts")[["wake_potential"]]
            st.line_chart(wake_df, height=180)

        with right:
            st.caption("Items scanned per tick")
            scan_cols = ["items_scanned"]
            if "seeds_scanned" in ticks.columns:
                scan_cols.append("seeds_scanned")
            scan_df = ticks[scan_cols].reset_index(drop=True)
            st.bar_chart(scan_df, height=180)

    st.divider()

    # --- Sentry Signals ---
    st.caption("Sentry signals")
    if len(signals) > 0:
        sig_cols = ["ts", "item_id", "score", "above_threshold"]
        if "source" in signals.columns:
            sig_cols.append("source")
        sig_display = signals[[c for c in sig_cols if c in signals.columns]].copy()
        sig_display["ts"] = sig_display["ts"].dt.strftime("%H:%M:%S")
        sig_display = sig_display.sort_values("ts", ascending=False)

        left2, right2 = st.columns([3, 1])
        with left2:
            st.dataframe(sig_display, use_container_width=True, hide_index=True, height=200)
        with right2:
            if "score" in signals.columns:
                scores = signals["score"].dropna()
                st.caption("Score distribution")
                hist = scores.value_counts(bins=10).sort_index()
                st.bar_chart(hist, height=180)
    else:
        st.info("No sentry signals yet.")

    st.divider()

    # --- Strategist Drafts ---
    st.caption("Strategist drafts")
    if len(drafts) > 0:
        draft_cols = ["ts", "item_id", "action", "charge", "draft_length"]
        if "source" in drafts.columns:
            draft_cols.append("source")
        available = [c for c in draft_cols if c in drafts.columns]
        draft_display = drafts[available].copy()
        if "ts" in draft_display.columns:
            draft_display["ts"] = draft_display["ts"].dt.strftime("%H:%M:%S")
        draft_display = draft_display.sort_values("ts", ascending=False)
        st.dataframe(draft_display, use_container_width=True, hide_index=True, height=200)
    else:
        st.info("No strategist drafts yet.")

    # --- Strategist Failures ---
    strat_errors = df[df["event_type"].isin(["strategist_error", "strategist_parse_fail", "strategist_budget_skip"])]
    if len(strat_errors) > 0:
        st.caption(f"Strategist failures ({len(strat_errors)})")
        fail_cols = ["ts", "event_type", "item_id", "error", "error_type", "raw_text", "model"]
        available_fail = [c for c in fail_cols if c in strat_errors.columns]
        fail_display = strat_errors[available_fail].copy()
        if "ts" in fail_display.columns:
            fail_display["ts"] = fail_display["ts"].dt.strftime("%H:%M:%S")
        fail_display = fail_display.sort_values("ts", ascending=False)
        st.dataframe(fail_display, use_container_width=True, hide_index=True, height=200)

    # --- Errors ---
    if len(errors) > 0:
        st.divider()
        st.caption("Daemon errors")
        err_cols = ["ts", "tick", "error"]
        err_display = errors[[c for c in err_cols if c in errors.columns]].copy()
        if "ts" in err_display.columns:
            err_display["ts"] = err_display["ts"].dt.strftime("%H:%M:%S")
        st.dataframe(err_display, use_container_width=True, hide_index=True, height=150)

    # --- Raw ticks ---
    with st.expander("Recent ticks (raw)", expanded=False):
        if len(ticks) > 0:
            tick_cols = ["ts", "tick", "items_scanned", "new_items", "seeds_scanned",
                         "signals_above_threshold", "wake_potential", "draft_count", "model"]
            available = [c for c in tick_cols if c in ticks.columns]
            tick_display = ticks[available].copy()
            if "ts" in tick_display.columns:
                tick_display["ts"] = tick_display["ts"].dt.strftime("%H:%M:%S")
            tick_display = tick_display.sort_values("ts", ascending=False).head(50)
            st.dataframe(tick_display, use_container_width=True, hide_index=True)


# ============================================================
# Controls metadata (mirrors v15_7/controls.py)
# Format: (key, type, default, description, category, min, max, choices)
# ============================================================
CONTROLS_META = [
    # LLM
    ("temperature",              "float", 0.7,    "Conscious LLM temperature",                      "llm",      0.0,  2.0,   None),
    ("subconscious_temperature", "float", 0.3,    "Daemon LLM temperature",                         "llm",      0.0,  2.0,   None),
    ("conscious_model",          "str",   "gemini-2.5-flash", "Model for conscious loop",           "llm",      None, None,  None),
    ("subconscious_model",       "str",   "gemini-2.5-flash", "Model for subconscious daemon",      "llm",      None, None,  None),
    # Cost
    ("daily_budget_usd",         "float", 1.0,    "Daily API spend limit (USD)",                    "cost",     0.01, 100.0, None),
    # Timing
    ("cycle_interval_minutes",   "int",   5,      "Minutes between cycles",                         "timing",   1,    120,   None),
    ("post_interval_minutes",    "int",   30,     "Minutes between posts",                          "timing",   5,    1440,  None),
    ("post_failure_cooldown_seconds", "int", 900, "Cooldown after a failed post attempt (secs)",    "timing",   60,   7200,  None),
    # Output
    ("output_destination",       "str",   "analog_home", "Where to publish artifacts",              "output",   None, None,  ["analog_home", "moltbook_and_analog_home"]),
    # Social
    ("mode",                     "str",   "all",  "Action mode",                                    "social",   None, None,  ["all", "comment_only", "no_post", "no_comment", "post_only"]),
    ("follow_prob",              "float", 0.60,   "Follow-on-like probability",                     "social",   0.0,  1.0,   None),
    ("create_submolt_prob",      "float", 0.05,   "Create submolt probability",                     "social",   0.0,  1.0,   None),
    ("allow_downvote",           "bool",  True,   "Allow downvoting",                               "social",   None, None,  None),
    ("priority",                 "str",   "replies_first", "Reply priority",                        "social",   None, None,  ["replies_first", "outside_first"]),
    ("my_post_scan_limit",       "int",   50,     "Recent own posts to scan for unanswered comments", "social", 5,    200,   None),
    ("reply_threads_scanned",    "int",   4,      "Own post threads to scan per cycle for replies", "social",   1,    20,    None),
    ("reply_max_comments",       "int",   25,     "Max comments evaluated per thread",              "social",   5,    100,   None),
    ("thread_comments_for_engagement", "int", 12, "Max comments on a thread before dogpile guard fires", "social", 1, 100,  None),
    # Daemon
    ("sentry_interval_seconds",  "int",   60,     "Seconds between sentry scans",                   "daemon",   10,   600,   None),
    ("signal_threshold",         "float", 0.5,    "Score to trigger strategist",                    "daemon",   0.0,  1.0,   None),
    ("wake_threshold",           "float", 2.0,    "Charge to fire conscious",                       "daemon",   0.5,  10.0,  None),
    ("wake_decay_rate",          "float", 1.0,    "Per-tick charge decay (1.0 = no decay)",         "daemon",   0.5,  1.0,   None),
    ("wake_refractory",          "float", -2.0,   "Wake potential reset after firing (negative = cooldown)", "daemon", -10.0, 0.0, None),
    ("max_drafts",               "int",   10,     "Max drafts in buffer before pruning oldest",     "daemon",   1,    50,    None),
    ("sentry_max_tokens",        "int",   256,    "Max output tokens for sentry scoring",           "daemon",   64,   1024,  None),
    ("strategist_max_tokens",    "int",   4096,   "Max output tokens for strategist drafts",        "daemon",   128,  8192,  None),
    ("max_item_age_hours",       "int",   24,     "Ignore feed items older than this (hours)",      "daemon",   1,    168,   None),
    ("saved_plan_max_cycles",    "int",   5,      "Cycles a daemon draft persists before expiring", "daemon",   1,    20,    None),
    ("daemon_notes_max",         "int",   5,      "Max directive notes retained",                   "daemon",   1,    20,    None),
    # Context
    ("feed_batch_size",          "int",   12,     "Feed items per cycle",                           "context",  1,    50,    None),
    ("feed_item_chars",          "int",   400,    "Max chars per feed item in prompt",              "context",  50,   2000,  None),
    ("history_context_n",        "int",   15,     "History entries in prompt",                      "context",  1,    50,    None),
    ("memory_max_chars",         "int",   4000,   "Memory context budget (chars)",                  "context",  500,  20000, None),
    ("reply_candidate_chars",    "int",   5000,   "Max chars for reply candidate text in prompt",   "context",  500,  20000, None),
    ("outside_candidate_chars",  "int",   5000,   "Max chars for outside comment candidate in prompt", "context", 500, 20000, None),
    # Conscious
    ("dream_depth",              "int",   10,     "History entries to synthesize per dream",        "conscious", 3,   50,    None),
]

# CLI flags that override these controls at startup (shown as hints in the UI)
CLI_FLAGS = {
    "temperature":              "--temperature",
    "conscious_model":          "--conscious-model / --gemini-model",
    "subconscious_model":       "--subconscious-model",
    "daily_budget_usd":         "--daily-budget",
    "cycle_interval_minutes":   "--interval",
    "post_interval_minutes":    "--post-interval",
    "mode":                     "--mode",
    "priority":                 "--priority",
    "allow_downvote":           "--allow-downvote",
    "sentry_interval_seconds":  "--sentry-interval",
}

CATEGORY_ORDER = ["llm", "cost", "timing", "output", "social", "daemon", "context", "conscious"]
CATEGORY_LABELS = {
    "llm": "LLM",
    "cost": "Cost",
    "timing": "Timing",
    "output": "Output",
    "social": "Social",
    "daemon": "Daemon",
    "context": "Context",
    "conscious": "Conscious",
}




# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="Autonomy Dashboard v2.1", layout="wide")
    st.title("Autonomy Dashboard v2.1")

    # Auto-ingest telemetry on first load
    auto_ingest_on_startup()

    # Brain selector in sidebar (shared across tabs)
    brains = discover_brains()
    brain_options = ["(all)"] + brains
    brain_filter = st.sidebar.selectbox("Brain", brain_options, index=0)

    # Ingest refresh button in sidebar
    st.sidebar.divider()
    if st.sidebar.button("Refresh telemetry data"):
        result = run_ingest()
        st.session_state["last_ingest"] = result
        # Clear DuckDB cached connection so views are recreated with fresh data
        get_con.clear()
        st.rerun()
    if st.session_state.get("last_ingest"):
        st.sidebar.caption(st.session_state["last_ingest"])

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cycle Replay", "Daemon Monitor", "Controls"])

    with tab1:
        render_overview_tab(brain_filter)

    with tab2:
        render_cycle_replay_tab(brain_filter)

    with tab3:
        render_daemon_tab(brain_filter)

    with tab4:
        render_controls_tab(brain_filter)


if __name__ == "__main__":
    main()
