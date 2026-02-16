"""Shared utility functions: JSON parsing, state management, helpers."""

import os
import re
import json
import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .telemetry import TelemetryLogger

from .config import (
    KNOWLEDGE_MAX_CHARS, MEMORY_MAX_CHARS, HISTORY_KEEP, HISTORY_CONTEXT_N,
    MOLTBOOK_WEB_BASE,
)


# ============================================================
# JSON parsing / repair
# ============================================================
def _repair_json_newlines(s: str) -> str:
    """Fix unescaped newlines/tabs inside JSON string values (common LLM mistake)."""
    result = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
                result.append(ch)
            elif ch == '\\':
                esc = True
                result.append(ch)
            elif ch == '"':
                in_str = False
                result.append(ch)
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_str = True
            result.append(ch)
    return ''.join(result)


def extract_first_json_object(text: str) -> Optional[str]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    # Find { followed by " — real JSON objects start with a key, skipping
    # prose braces like {Layer 1} that appear in reasoning preamble.
    m = re.search(r'\{\s*"', s)
    if not m:
        return None
    start = m.start()
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None


def _strip_to_json(s: str) -> str:
    """Strip preamble text and markdown fences to isolate the JSON portion."""
    raw = (s or "").strip()
    # Strategy 1: find ```json fence (most reliable when preamble has curly braces)
    fence = re.search(r"```json\s*\n?\s*(\{)", raw)
    if fence:
        raw = raw[fence.start(1):]
        raw = re.sub(r"\n?\s*```\s*$", "", raw).strip()
        return raw
    # Strategy 2: strip leading ``` fence if the whole response is fenced
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
    # Strategy 3: find { followed by " (real JSON object start, not prose braces)
    m = re.search(r'\{\s*"', raw)
    if m:
        raw = raw[m.start():]
    # Strip trailing text/fences after the last }
    last_brace = raw.rfind("}")
    if last_brace >= 0 and last_brace < len(raw) - 1:
        raw = raw[:last_brace + 1]
    return raw


def parse_json_strict(s: str) -> Dict[str, Any]:
    stripped = (s or "").strip()
    if not stripped:
        raise ValueError("Invalid JSON from model. Snippet: (empty)")

    # 1. Fast path: already clean JSON
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # 2. Brace-match extraction on original (handles preamble + trailing text)
    candidate = extract_first_json_object(stripped)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return json.loads(_repair_json_newlines(candidate))
        except Exception:
            pass

    # 3. _strip_to_json fallback (handles ```json fences, complex preamble)
    raw = _strip_to_json(stripped)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return json.loads(_repair_json_newlines(raw))
        except Exception:
            pass

    snippet = (candidate or raw or stripped)[:1200]
    raise ValueError(f"Invalid JSON from model. Snippet: {snippet}")


# ============================================================
# General helpers
# ============================================================
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def post_url(post_id: str) -> str:
    return f"{MOLTBOOK_WEB_BASE}/post/{post_id}"


def safe_json_load(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def safe_json_write(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def safe_text_write(path: str, text: str) -> None:
    """Atomically write text to file using temp file pattern."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def backup_kernel(kernel_path: str) -> bool:
    """Create backup of current kernel before overwriting.

    Returns:
        True if backup created successfully, False if source doesn't exist.
    """
    if not os.path.exists(kernel_path):
        return False

    backup_path = kernel_path.replace("_kernel_prompt.txt", "_kernel_prompt.backup.txt")
    if backup_path == kernel_path:  # Fallback if pattern doesn't match
        backup_path = kernel_path + ".backup"

    try:
        with open(kernel_path, "r", encoding="utf-8") as f:
            current = f.read()
        safe_text_write(backup_path, current)
        return True
    except Exception:
        return False


def update_kernel_file(kernel_path: str, new_kernel: str,
                       telemetry: Optional['TelemetryLogger'] = None,
                       reason: str = "") -> Dict[str, Any]:
    """Validate and write new kernel to disk with backup.

    Args:
        kernel_path: Path to kernel file
        new_kernel: New kernel text to write
        telemetry: Optional telemetry logger
        reason: Why the kernel is being updated (logged to kernel history)

    Returns:
        Dict with keys: success (bool), error (str), backup_created (bool)
    """
    # Validation: length check
    text = (new_kernel or "").strip()
    if len(text) < 50:
        return {"success": False, "error": "Kernel too short (min 50 chars)", "backup_created": False}
    if len(text) > 5000:
        return {"success": False, "error": "Kernel too long (max 5000 chars)", "backup_created": False}

    # Snapshot the old kernel before overwriting
    if telemetry:
        try:
            with open(kernel_path, "r", encoding="utf-8") as f:
                old_text = f.read()
            telemetry.log_kernel_snapshot(old_text, reason=f"pre_update: {reason}", source="disk_write")
        except Exception:
            pass

    # Create backup (.backup.txt for quick manual recovery)
    backup_created = backup_kernel(kernel_path)

    # Write new kernel
    try:
        safe_text_write(kernel_path, text)
        # Snapshot the new kernel
        if telemetry:
            telemetry.log_kernel_snapshot(text, reason=reason, source="disk_write")
            telemetry.log("kernel_updated", {
                "kernel_path": kernel_path,
                "new_length": len(text),
                "backup_created": backup_created,
            })
        return {"success": True, "error": None, "backup_created": backup_created}
    except Exception as e:
        if telemetry:
            telemetry.log("kernel_update_failed", {
                "kernel_path": kernel_path,
                "error": str(e),
                "backup_created": backup_created,
            })
        return {"success": False, "error": str(e), "backup_created": backup_created}


def shorten(s: str, max_chars: int) -> str:
    s = (s or "").replace("\r", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def get_author_name(author_obj: Any) -> str:
    if isinstance(author_obj, dict):
        return author_obj.get("name") or author_obj.get("username") or "unknown"
    return "unknown"


def author_handle(obj: Any) -> str:
    if isinstance(obj, dict):
        return (obj.get("name") or obj.get("username") or obj.get("handle") or obj.get("id") or "").strip() or "unknown"
    return str(obj or "unknown").strip() or "unknown"


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def get_post_comment_count(post: Dict[str, Any]) -> int:
    if not isinstance(post, dict):
        return 0
    keys = (
        "comment_count", "commentCount",
        "comments_count", "commentsCount",
        "num_comments", "numComments",
        "total_comments", "totalComments",
        "comment_total", "commentTotal",
        "totalCommentCount", "commentCountTotal",
    )
    for k in keys:
        v = post.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            pass
    counts = post.get("counts") or post.get("stats") or {}
    if isinstance(counts, dict):
        for k in ("comments", "commentCount", "comment_count", "totalComments"):
            v = counts.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                pass
    return 0


# ============================================================
# State management (memories.json)
# ============================================================
def load_state(state_path: str) -> Dict[str, Any]:
    state = safe_json_load(state_path, {})
    if not isinstance(state, dict):
        state = {}

    state.setdefault("my_post_ids", [])
    state.setdefault("replied_comment_keys", [])
    state.setdefault("memory", "")
    state.setdefault("history", [])
    state.setdefault("subscribed_submolts", [])
    state.setdefault("followed_agents", [])
    state.setdefault("next_post_time", 0.0)
    state.setdefault("next_comment_time", 0.0)

    # Type safety
    if not isinstance(state["my_post_ids"], list):
        state["my_post_ids"] = []
    if not isinstance(state["replied_comment_keys"], list):
        state["replied_comment_keys"] = []
    if not isinstance(state.get("memory", ""), str):
        state["memory"] = str(state.get("memory", ""))
    if not isinstance(state["history"], list):
        state["history"] = []
    if not isinstance(state.get("subscribed_submolts"), list):
        state["subscribed_submolts"] = []
    if not isinstance(state.get("followed_agents"), list):
        state["followed_agents"] = []
    if not isinstance(state["next_post_time"], (int, float)):
        state["next_post_time"] = 0.0
    if not isinstance(state["next_comment_time"], (int, float)):
        state["next_comment_time"] = 0.0

    # Trim
    state["history"] = state["history"][-HISTORY_KEEP:]
    state["my_post_ids"] = state["my_post_ids"][-500:]
    state["replied_comment_keys"] = state["replied_comment_keys"][-5000:]
    state["subscribed_submolts"] = state["subscribed_submolts"][-5000:]
    state["followed_agents"] = state["followed_agents"][-5000:]

    return state


def save_state(state_path: str, state: Dict[str, Any]) -> None:
    safe_json_write(state_path, state)


def add_history(state: Dict[str, Any], entry: Dict[str, Any]) -> None:
    state.setdefault("history", [])
    entry = dict(entry)
    entry.setdefault("ts", now_iso())
    state["history"].append(entry)
    state["history"] = state["history"][-HISTORY_KEEP:]


def history_context(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    hist = state.get("history", [])[-HISTORY_CONTEXT_N:]
    for h in hist:
        action = h.get("action", "?")
        target = h.get("target", "")
        summary = h.get("summary", "")
        lines.append(f"- {h.get('ts','')}: {action} {target} | {summary}")
    return "\n".join(lines) if lines else "No prior actions recorded."


def memory_context(state: Dict[str, Any]) -> str:
    mem = (state.get("memory") or "").strip()
    return mem[:MEMORY_MAX_CHARS] if mem else "No personal memory set."


def compress_memories(
    state: Dict[str, Any],
    chat_fn,
    depth: int = 20,
    telemetry: Optional['TelemetryLogger'] = None,
) -> Dict[str, Any]:
    """Compress oldest `depth` history entries into a narrative prepended to memory.

    Args:
        state: Agent state dict (modified in-place).
        chat_fn: Callable that sends a prompt to the conscious LLM and returns text.
        depth: Number of oldest history entries to compress.
        telemetry: Optional logger.

    Returns:
        Dict with keys: success, entries_compressed, narrative_length, error
    """
    history = state.get("history", [])
    if len(history) < depth:
        depth = len(history)
    if depth < 3:
        return {"success": False, "entries_compressed": 0, "narrative_length": 0,
                "error": "Too few history entries to compress"}

    to_compress = history[:depth]
    remaining = history[depth:]

    lines = []
    for h in to_compress:
        action = h.get("action", "?")
        target = h.get("target", "")
        summary = h.get("summary", "")
        ts = h.get("ts", "")
        lines.append(f"- {ts}: {action} {target} | {summary}")
    entries_text = "\n".join(lines)

    existing_memory = (state.get("memory") or "").strip()

    prompt = (
        "You are performing memory consolidation (dreaming).\n"
        "Compress the following history entries into a concise narrative paragraph.\n"
        "Preserve key facts, relationships, recurring themes, and important outcomes.\n"
        "Discard routine/repetitive details. Write in first person past tense.\n"
        "Keep the narrative under 600 characters.\n\n"
        f"History entries to compress ({len(to_compress)} entries):\n"
        f"{entries_text}\n\n"
        f"Existing memory (for context, do not repeat):\n"
        f"{existing_memory[:1000] if existing_memory else '(empty)'}\n\n"
        "Write ONLY the compressed narrative paragraph, nothing else."
    )

    try:
        narrative = chat_fn(prompt).strip()
        if existing_memory:
            new_memory = f"{narrative}\n---\n{existing_memory}"
        else:
            new_memory = narrative
        state["memory"] = new_memory[:MEMORY_MAX_CHARS]
        state["history"] = remaining

        result = {
            "success": True,
            "entries_compressed": len(to_compress),
            "narrative_length": len(narrative),
            "error": None,
        }
        if telemetry:
            telemetry.log("dream_compress", result)
        return result
    except Exception as e:
        result = {
            "success": False,
            "entries_compressed": 0,
            "narrative_length": 0,
            "error": str(e),
        }
        if telemetry:
            telemetry.log("dream_compress_failed", result)
        return result


# ============================================================
# Brain file loaders
# ============================================================
def load_kernel(kernel_path: str) -> str:
    if os.path.exists(kernel_path):
        with open(kernel_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def load_knowledge(knowledge_path: str) -> str:
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            return f.read().strip()[:KNOWLEDGE_MAX_CHARS]
    return ""
