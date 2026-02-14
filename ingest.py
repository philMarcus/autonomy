#!/usr/bin/env python3
"""
autonomy_duckdb_ingest_v2
JSONL telemetry -> partitioned Parquet (DuckDB)

- Source of truth: telemetry/{brain}_events.jsonl (per-brain, append-only)
- Legacy support: also reads telemetry/events.jsonl if present
- Warehouse: warehouse/events/dt=YYYY-MM-DD/*.parquet
- Full fidelity: payload_json stores the original JSON line exactly
"""

from __future__ import annotations

import glob as glob_mod
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd


# ----------------------------
# Config
# ----------------------------
SOURCE_DIR = Path("telemetry")
WAREHOUSE_DIR = Path("warehouse")
EVENTS_DIR = WAREHOUSE_DIR / "events"
STATE_PATH = WAREHOUSE_DIR / "ingest_state.json"


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs() -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        # Migrate legacy single-file format to per-file offsets
        if "byte_offset" in state and "file_offsets" not in state:
            state["file_offsets"] = {"events.jsonl": state.pop("byte_offset")}
        return state
    return {"file_offsets": {}}


def save_state(state: Dict[str, Any]) -> None:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_ts(ts_str: str) -> datetime:
    # Handles "2026-02-08T14:51:35+00:00"
    # Python fromisoformat supports offsets like +00:00
    dt = datetime.fromisoformat(ts_str)
    # ensure tz-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def safe_int(x: Any) -> Optional[int]:
    try:
        return None if x is None else int(x)
    except Exception:
        return None


def safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return None


def infer_api_service(event_type: str) -> Optional[str]:
    if event_type.endswith("_api_call"):
        # "moltbook_api_call" -> "moltbook"
        return event_type[: -len("_api_call")]
    return None


def infer_provider_from_model(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "gpt" in m or "openai" in m:
        return "openai"
    if "mistral" in m:
        return "mistral"
    return None


def normalize_event(obj: Dict[str, Any], raw_line: str) -> Dict[str, Any]:
    ts = parse_ts(obj["ts"])
    dt = ts.date().isoformat()

    event_type = obj.get("event_type")
    model = obj.get("model")

    out: Dict[str, Any] = {
        "ts": ts.isoformat(),
        "seq": safe_int(obj.get("seq")),
        "dt": dt,
        "run_id": obj.get("run_id"),
        "brain": obj.get("brain"),
        "version": obj.get("version"),
        "cycle_num": safe_int(obj.get("cycle")),
        "event_type": event_type,
        "tag": obj.get("tag"),
        "severity": obj.get("severity"),

        # LLM
        "provider": obj.get("provider") or infer_provider_from_model(model),
        "model": model,
        "prompt_chars": safe_int(obj.get("prompt_chars")),
        "response_chars": safe_int(obj.get("response_chars")),
        "prompt_tokens_est": safe_int(obj.get("est_prompt_tokens")),
        "response_tokens_est": safe_int(obj.get("est_response_tokens")),
        "latency_ms": safe_int(obj.get("latency_ms")),
        "cost_usd_est": None,  # fill later with pricing model if desired

        # API
        "api_service": obj.get("api_service") or infer_api_service(event_type or ""),
        "http_method": obj.get("method"),
        "http_path": obj.get("path"),
        "http_status": safe_int(obj.get("status")),
        "api_latency_ms": safe_int(obj.get("latency_ms")) if event_type and event_type.endswith("_api_call") else None,
        # has_body/body_bytes: older telemetry uses has_body/body_bytes,
        # newer uses req_has_body/req_body_bytes (+ resp_ variants)
        "has_body": safe_bool(obj.get("has_body") or obj.get("req_has_body")),
        "body_bytes": safe_int(obj.get("body_bytes") or obj.get("req_body_bytes")),
        "rate_limited": True if safe_int(obj.get("status")) == 429 else None,

        # Actions
        "action_type": obj.get("action"),
        "post_id": obj.get("post_id"),
        "comment_id": obj.get("comment_id"),
        "parent_comment_id": obj.get("parent_comment_id"),
        "action_result": obj.get("action_result"),

        # Errors — telemetry logs "error" not "error_message"
        "error_type": obj.get("error_type"),
        "error_message": obj.get("error_message") or obj.get("error"),
        "error_stack": obj.get("error_stack"),

        # Full fidelity
        "payload_json": raw_line.rstrip("\n"),
    }
    return out


def write_parquet_partitioned(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)

    # Use DuckDB to write Parquet (keeps types reasonable, no pyarrow dependency)
    con = duckdb.connect(database=":memory:")

    con.register("batch_df", df)

    # Ensure dt partition folder per day; write a new file each ingest batch
    # We'll loop per dt to keep file naming simple and avoid overwrites.
    for dt_value in sorted(df["dt"].dropna().unique()):
        dt_str = str(dt_value)
        out_dir = EVENTS_DIR / f"dt={dt_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # monotonic-ish filename
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"events_{stamp}.parquet"

        con.execute(
            f"""
            COPY (
              SELECT
                -- cast to keep a stable schema
                CAST(ts AS TIMESTAMPTZ) AS ts,
                CAST(seq AS INTEGER) AS seq,
                CAST(dt AS DATE) AS dt,
                CAST(run_id AS VARCHAR) AS run_id,
                CAST(brain AS VARCHAR) AS brain,
                CAST(version AS VARCHAR) AS version,
                CAST(cycle_num AS INTEGER) AS cycle_num,
                CAST(event_type AS VARCHAR) AS event_type,
                CAST(tag AS VARCHAR) AS tag,
                CAST(severity AS VARCHAR) AS severity,

                CAST(provider AS VARCHAR) AS provider,
                CAST(model AS VARCHAR) AS model,
                CAST(prompt_chars AS INTEGER) AS prompt_chars,
                CAST(response_chars AS INTEGER) AS response_chars,
                CAST(prompt_tokens_est AS INTEGER) AS prompt_tokens_est,
                CAST(response_tokens_est AS INTEGER) AS response_tokens_est,
                CAST(latency_ms AS INTEGER) AS latency_ms,
                CAST(cost_usd_est AS DOUBLE) AS cost_usd_est,

                CAST(api_service AS VARCHAR) AS api_service,
                CAST(http_method AS VARCHAR) AS http_method,
                CAST(http_path AS VARCHAR) AS http_path,
                CAST(http_status AS INTEGER) AS http_status,
                CAST(api_latency_ms AS INTEGER) AS api_latency_ms,
                CAST(has_body AS BOOLEAN) AS has_body,
                CAST(body_bytes AS INTEGER) AS body_bytes,
                CAST(rate_limited AS BOOLEAN) AS rate_limited,

                CAST(action_type AS VARCHAR) AS action_type,
                CAST(post_id AS VARCHAR) AS post_id,
                CAST(comment_id AS VARCHAR) AS comment_id,
                CAST(parent_comment_id AS VARCHAR) AS parent_comment_id,
                CAST(action_result AS VARCHAR) AS action_result,

                CAST(error_type AS VARCHAR) AS error_type,
                CAST(error_message AS VARCHAR) AS error_message,
                CAST(error_stack AS VARCHAR) AS error_stack,

                CAST(payload_json AS VARCHAR) AS payload_json
              FROM batch_df
              WHERE dt = '{dt_str}'
            )
            TO '{out_path.as_posix()}'
            (FORMAT PARQUET);
            """
        )

    con.close()


def _find_event_files() -> List[Path]:
    """Return all *_events.jsonl files plus legacy events.jsonl if present."""
    files = []
    if SOURCE_DIR.exists():
        for p in sorted(SOURCE_DIR.glob("*_events.jsonl")):
            files.append(p)
        # Legacy single file (if not already matched by the glob)
        legacy = SOURCE_DIR / "events.jsonl"
        if legacy.exists() and legacy not in files:
            files.append(legacy)
    return files


def _ingest_file(path: Path, offset: int) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Read new lines from a single JSONL file starting at byte offset.

    Returns (lines_read, events_parsed, new_offset, rows).
    """
    rows: List[Dict[str, Any]] = []
    new_offset = offset
    lines_read = 0

    with path.open("rb") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                break
            new_offset += len(line)
            lines_read += 1

            s = line.decode("utf-8", errors="replace").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "ts" not in obj or "event_type" not in obj:
                    continue
                rows.append(normalize_event(obj, s))
            except json.JSONDecodeError:
                continue

    return lines_read, len(rows), new_offset, rows


def ingest_once() -> Tuple[int, int]:
    ensure_dirs()
    state = load_state()
    file_offsets: Dict[str, int] = state.get("file_offsets", {})

    event_files = _find_event_files()
    if not event_files:
        print(f"No *_events.jsonl files found in {SOURCE_DIR.resolve()}")
        return 0, 0

    all_rows: List[Dict[str, Any]] = []
    total_lines = 0

    for path in event_files:
        fname = path.name
        offset = int(file_offsets.get(fname, 0))
        lines_read, events_parsed, new_offset, rows = _ingest_file(path, offset)
        total_lines += lines_read
        all_rows.extend(rows)
        file_offsets[fname] = new_offset
        if lines_read:
            print(f"  {fname}: +{lines_read} lines, +{events_parsed} events")

    # Write parquet
    write_parquet_partitioned(all_rows)

    # Update state
    state["file_offsets"] = file_offsets
    save_state(state)

    return total_lines, len(all_rows)


if __name__ == "__main__":
    lines, events = ingest_once()
    print(f"Ingested: lines_read={lines}, events_written={events}")
    print(f"Parquet output root: {EVENTS_DIR.resolve()}")
