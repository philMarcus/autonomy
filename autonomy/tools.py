"""Tool registry and built-in tools for the v18 tool-augmented planner.

This module provides:
  - ToolDef / ToolCall / ToolResult — data classes for tool definitions and execution
  - ToolRegistry — registry that maps tool names to handlers and executes calls
  - Built-in tools: todos, lab notebook (experiments), tagline, temporary control overrides
  - build_tool_registry() — factory that wires everything up with closures over state/ctrl/store

Tool schemas use JSON Schema format compatible with Gemini FunctionDeclaration.
All tool handlers are synchronous functions that return JSON-serializable dicts.
Errors are caught and returned as {"error": "message"}, never raised.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ============================================================
# Core data types
# ============================================================

@dataclass
class ToolDef:
    """Definition of a tool the agent can call."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object
    handler: Callable           # Python function to execute


@dataclass
class ToolCall:
    """A tool call request from the model."""
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    call_id: str
    name: str
    content: str  # JSON-serialized result


# ============================================================
# Tool Registry
# ============================================================

class ToolRegistry:
    """Registry of available tools and their handlers.

    Each tool is a ToolDef with a name, description, JSON Schema parameters,
    and a handler function. The registry supports schema export (for LLM API
    tool declarations) and batch execution of tool calls.
    """

    def __init__(self, brain_name: str, brains_dir: str = "brains"):
        self._tools: Dict[str, ToolDef] = {}
        self._brain = brain_name
        self._brains_dir = brains_dir

    def register(self, tool: ToolDef) -> None:
        """Register a tool definition. Overwrites if name already exists."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDef]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the LLM API (Gemini FunctionDeclaration format).

        Each schema is a dict with 'name', 'description', and 'parameters'
        (a JSON Schema object with type: "object").
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def list_names(self) -> List[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def prompt_summary(self) -> str:
        """Build a short text block for the planner prompt listing available tools.

        The LLM sees this to know what tools it can call. Kept brief — the actual
        schemas are passed via the API's tool mechanism; this is just a heads-up.
        """
        if not self._tools:
            return ""
        lines = [
            "--- AVAILABLE TOOLS ---",
            "You can call tools DURING this cycle, before your final JSON action.",
            "Tool use does NOT replace your action — you call tools first, THEN choose",
            "your action (POST, COMMENT, etc.) as normal. Example: call create_experiment",
            "and add_todo, then POST about your findings. Use tools every cycle as needed.",
        ]
        for t in sorted(self._tools.values(), key=lambda x: x.name):
            params = list(t.parameters.get("properties", {}).keys())
            params_hint = f"({', '.join(params)})" if params else "()"
            lines.append(f"  {t.name}{params_hint} — {t.description}")
        # Add quick state summaries for todo + experiments
        _todo_path = self._path("todos.json")
        _exp_path = self._path("experiments.json")
        hints = []
        try:
            if os.path.exists(_todo_path):
                with open(_todo_path, "r") as f:
                    todos = json.load(f)
                open_count = sum(1 for t in todos if t.get("status") == "open")
                if open_count:
                    hints.append(f"{open_count} open todo(s)")
        except Exception:
            pass
        try:
            if os.path.exists(_exp_path):
                with open(_exp_path, "r") as f:
                    exps = json.load(f)
                active = sum(1 for e in exps if e.get("status") == "active")
                if active:
                    hints.append(f"{active} active experiment(s)")
        except Exception:
            pass
        if hints:
            lines.append(f"\nCurrent state: {'; '.join(hints)}. Use the tools to see details.")
        return "\n".join(lines)

    def execute(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a batch of tool calls and return results.

        Each call is executed sequentially. Errors are caught per-call and
        returned as {"error": "..."} rather than raised.
        """
        results: List[ToolResult] = []
        for call in calls:
            tool = self._tools.get(call.name)
            if not tool:
                results.append(ToolResult(
                    call.id, call.name,
                    json.dumps({"error": f"Unknown tool: {call.name}"}),
                ))
                continue
            try:
                result = tool.handler(**call.args)
                results.append(ToolResult(
                    call.id, call.name,
                    json.dumps(result, default=str),
                ))
            except Exception as e:
                log.warning("Tool %s execution error: %s", call.name, e)
                results.append(ToolResult(
                    call.id, call.name,
                    json.dumps({"error": f"{type(e).__name__}: {e}"}),
                ))
        return results

    def _path(self, filename: str) -> str:
        """Build path to a brain-specific file in the brains directory."""
        return os.path.join(self._brains_dir, f"{self._brain}_{filename}")


# ============================================================
# File I/O helpers (thread-safe atomic writes)
# ============================================================

_file_lock = threading.Lock()


def _read_json_file(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning default if missing or corrupt."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read %s: %s", path, e)
    return default if default is not None else []


def _write_json_file(path: str, data: Any) -> None:
    """Write data to a JSON file atomically (write to tmp, then rename).

    Uses a module-level lock to prevent concurrent writes from the daemon
    and conscious threads.
    """
    with _file_lock:
        tmp = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Atomic rename (same filesystem)
            os.replace(tmp, path)
        except OSError as e:
            log.warning("Failed to write %s: %s", path, e)
            # Clean up temp file on failure
            try:
                os.remove(tmp)
            except OSError:
                pass


# ============================================================
# Todo list tools
# ============================================================

def _push_state_to_analog_home(store, key: str, data: list) -> None:
    """Best-effort push of agent state to Analog Home API for frontend display."""
    if not store or not getattr(store, '_analog_home_url', None):
        return
    try:
        import requests
        from urllib.parse import urljoin
        url = urljoin(store._analog_home_url.rstrip("/") + "/", f"agent-state/{key}")
        requests.post(url, json={
            "brain": getattr(store, '_brain_name', 'ANALOG_I'),
            "data": data,
        }, timeout=5)
    except Exception:
        pass  # non-fatal, best-effort


def _build_todo_tools(
    registry: ToolRegistry,
    cycle_getter: Callable[[], int],
    store: Any = None,
) -> None:
    """Register todo list tools on the registry.

    Storage: brains/{brain}_todos.json — a list of todo items.
    """
    path = registry._path("todos.json")

    def _load_todos() -> List[Dict[str, Any]]:
        return _read_json_file(path, default=[])

    def _save_todos(todos: List[Dict[str, Any]]) -> None:
        _write_json_file(path, todos)
        _push_state_to_analog_home(store, "todos", todos)

    def _next_id(todos: List[Dict[str, Any]]) -> int:
        if not todos:
            return 1
        return max(t.get("id", 0) for t in todos) + 1

    # --- read_todos ---
    def read_todos() -> Dict[str, Any]:
        """Read all todo items."""
        todos = _load_todos()
        open_todos = [t for t in todos if t.get("status") == "open"]
        return {
            "todos": todos,
            "open_count": len(open_todos),
            "total_count": len(todos),
        }

    registry.register(ToolDef(
        name="read_todos",
        description="Read your todo list. Returns all items with their status.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=read_todos,
    ))

    # --- add_todo ---
    def add_todo(text: str, due_cycle: Optional[int] = None) -> Dict[str, Any]:
        """Add a new todo item."""
        if not text or not text.strip():
            return {"error": "text is required"}
        todos = _load_todos()
        new_id = _next_id(todos)
        item = {
            "id": new_id,
            "text": text.strip(),
            "created_cycle": cycle_getter(),
            "due_cycle": due_cycle,
            "status": "open",
            "completed_cycle": None,
        }
        todos.append(item)
        _save_todos(todos)
        return {"id": new_id, "status": "created"}

    registry.register(ToolDef(
        name="add_todo",
        description="Add a new todo item to your list.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The todo item text.",
                },
                "due_cycle": {
                    "type": "integer",
                    "description": "Optional cycle number by which this should be done.",
                },
            },
            "required": ["text"],
        },
        handler=add_todo,
    ))

    # --- complete_todo ---
    def complete_todo(id: int) -> Dict[str, Any]:
        """Mark a todo item as completed."""
        todos = _load_todos()
        for t in todos:
            if t.get("id") == id:
                if t.get("status") == "completed":
                    return {"id": id, "status": "already_completed"}
                t["status"] = "completed"
                t["completed_cycle"] = cycle_getter()
                _save_todos(todos)
                return {"id": id, "status": "completed"}
        return {"error": f"Todo {id} not found"}

    registry.register(ToolDef(
        name="complete_todo",
        description="Mark a todo item as completed.",
        parameters={
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The ID of the todo item to complete.",
                },
            },
            "required": ["id"],
        },
        handler=complete_todo,
    ))

    # --- remove_todo ---
    def remove_todo(id: int) -> Dict[str, Any]:
        """Remove a todo item entirely."""
        todos = _load_todos()
        original_len = len(todos)
        todos = [t for t in todos if t.get("id") != id]
        if len(todos) == original_len:
            return {"error": f"Todo {id} not found"}
        _save_todos(todos)
        return {"id": id, "status": "removed"}

    registry.register(ToolDef(
        name="remove_todo",
        description="Remove a todo item from your list entirely (not just complete it).",
        parameters={
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The ID of the todo item to remove.",
                },
            },
            "required": ["id"],
        },
        handler=remove_todo,
    ))


# ============================================================
# Lab notebook (experiments) tools
# ============================================================

def _build_experiment_tools(
    registry: ToolRegistry,
    cycle_getter: Callable[[], int],
    store: Any = None,
) -> None:
    """Register lab notebook (experiment tracking) tools on the registry.

    Storage: brains/{brain}_experiments.json — a list of experiments.
    """
    path = registry._path("experiments.json")

    def _load_experiments() -> List[Dict[str, Any]]:
        return _read_json_file(path, default=[])

    def _save_experiments(experiments: List[Dict[str, Any]]) -> None:
        _write_json_file(path, experiments)
        _push_state_to_analog_home(store, "experiments", experiments)

    def _find_experiment(
        experiments: List[Dict[str, Any]], name: str
    ) -> Optional[Dict[str, Any]]:
        """Find an experiment by name (case-insensitive)."""
        name_lower = name.strip().lower()
        for exp in experiments:
            if exp.get("name", "").strip().lower() == name_lower:
                return exp
        return None

    # --- list_experiments ---
    def list_experiments() -> Dict[str, Any]:
        """List all experiments with summary info."""
        experiments = _load_experiments()
        summaries = []
        for exp in experiments:
            summaries.append({
                "name": exp.get("name", ""),
                "status": exp.get("status", "active"),
                "hypothesis": exp.get("hypothesis", ""),
                "data_point_count": len(exp.get("data_points", [])),
                "created_cycle": exp.get("created_cycle"),
            })
        return {
            "experiments": summaries,
            "active_count": sum(1 for s in summaries if s["status"] == "active"),
            "total_count": len(summaries),
        }

    registry.register(ToolDef(
        name="list_experiments",
        description=(
            "List all experiments in your lab notebook. Returns name, status, "
            "hypothesis, and data point count for each."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=list_experiments,
    ))

    # --- create_experiment ---
    def create_experiment(
        name: str,
        hypothesis: str,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new experiment."""
        if not name or not name.strip():
            return {"error": "name is required"}
        if not hypothesis or not hypothesis.strip():
            return {"error": "hypothesis is required"}

        experiments = _load_experiments()
        if _find_experiment(experiments, name):
            return {"error": f"Experiment '{name.strip()}' already exists"}

        exp = {
            "name": name.strip(),
            "hypothesis": hypothesis.strip(),
            "method": (method or "").strip() or None,
            "status": "active",
            "created_cycle": cycle_getter(),
            "data_points": [],
            "conclusion": None,
            "closed_cycle": None,
        }
        experiments.append(exp)
        _save_experiments(experiments)
        return {"name": exp["name"], "status": "created"}

    registry.register(ToolDef(
        name="create_experiment",
        description=(
            "Create a new experiment in your lab notebook. Define a hypothesis "
            "to test and optionally describe your method."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short name for the experiment (must be unique).",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "What you expect to happen and why.",
                },
                "method": {
                    "type": "string",
                    "description": "Optional description of how you will test this.",
                },
            },
            "required": ["name", "hypothesis"],
        },
        handler=create_experiment,
    ))

    # --- log_data ---
    def log_data(experiment_name: str, observation: str) -> Dict[str, Any]:
        """Log a data point to an experiment."""
        if not observation or not observation.strip():
            return {"error": "observation is required"}

        experiments = _load_experiments()
        exp = _find_experiment(experiments, experiment_name)
        if not exp:
            return {"error": f"Experiment '{experiment_name}' not found"}
        if exp.get("status") != "active":
            return {"error": f"Experiment '{exp['name']}' is {exp.get('status')}, not active"}

        data_point = {
            "cycle": cycle_getter(),
            "observation": observation.strip(),
        }
        exp.setdefault("data_points", []).append(data_point)
        _save_experiments(experiments)
        return {
            "experiment": exp["name"],
            "data_point_index": len(exp["data_points"]) - 1,
            "status": "logged",
        }

    registry.register(ToolDef(
        name="log_data",
        description=(
            "Log an observation or data point to an active experiment."
        ),
        parameters={
            "type": "object",
            "properties": {
                "experiment_name": {
                    "type": "string",
                    "description": "Name of the experiment to log data to.",
                },
                "observation": {
                    "type": "string",
                    "description": "What you observed this cycle.",
                },
            },
            "required": ["experiment_name", "observation"],
        },
        handler=log_data,
    ))

    # --- read_experiment ---
    def read_experiment(name: str) -> Dict[str, Any]:
        """Read the full details of an experiment including all data points."""
        experiments = _load_experiments()
        exp = _find_experiment(experiments, name)
        if not exp:
            return {"error": f"Experiment '{name}' not found"}
        return {"experiment": exp}

    registry.register(ToolDef(
        name="read_experiment",
        description=(
            "Read the full details of an experiment, including all data points "
            "and conclusion (if closed)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the experiment to read.",
                },
            },
            "required": ["name"],
        },
        handler=read_experiment,
    ))

    # --- close_experiment ---
    def close_experiment(name: str, conclusion: str) -> Dict[str, Any]:
        """Close an experiment with a conclusion."""
        if not conclusion or not conclusion.strip():
            return {"error": "conclusion is required"}

        experiments = _load_experiments()
        exp = _find_experiment(experiments, name)
        if not exp:
            return {"error": f"Experiment '{name}' not found"}
        if exp.get("status") != "active":
            return {"error": f"Experiment '{exp['name']}' is already {exp.get('status')}"}

        exp["status"] = "closed"
        exp["conclusion"] = conclusion.strip()
        exp["closed_cycle"] = cycle_getter()
        _save_experiments(experiments)
        return {
            "experiment": exp["name"],
            "status": "closed",
            "data_points_collected": len(exp.get("data_points", [])),
        }

    registry.register(ToolDef(
        name="close_experiment",
        description=(
            "Close an experiment with a conclusion. Summarize what you learned."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the experiment to close.",
                },
                "conclusion": {
                    "type": "string",
                    "description": "What you concluded from the experiment.",
                },
            },
            "required": ["name", "conclusion"],
        },
        handler=close_experiment,
    ))


# ============================================================
# Tagline tool
# ============================================================

def _build_tagline_tool(
    registry: ToolRegistry,
    store: Any,  # Store instance (has set_tagline method)
) -> None:
    """Register the tagline update tool.

    Calls store.set_tagline() to POST the new tagline to Analog Home.
    """

    def update_tagline(text: str) -> Dict[str, Any]:
        """Update the site tagline on Analog Home."""
        if not text or not text.strip():
            return {"error": "text is required"}
        text = text.strip()
        if len(text) > 200:
            return {"error": f"Tagline too long ({len(text)} chars, max 200)"}
        if not hasattr(store, "set_tagline"):
            return {"error": "Store does not support set_tagline"}
        try:
            ok = store.set_tagline(text)
            if ok:
                return {"tagline": text, "status": "updated"}
            else:
                return {"error": "set_tagline returned False (API may be unreachable)"}
        except Exception as e:
            return {"error": f"set_tagline failed: {e}"}

    registry.register(ToolDef(
        name="update_tagline",
        description=(
            "Update your tagline (subtitle) on the Analog Home site. "
            "Keep it short and evocative."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The new tagline text (max 200 chars).",
                },
            },
            "required": ["text"],
        },
        handler=update_tagline,
    ))


# ============================================================
# Temporary control override tools
# ============================================================

_STATE_KEY = "_temp_control_overrides"


def _build_temp_control_tools(
    registry: ToolRegistry,
    state: Dict[str, Any],
    ctrl: Any,  # ControlRegistry instance
    cycle_getter: Callable[[], int],
) -> None:
    """Register temporary control override tools.

    Overrides are stored in state["_temp_control_overrides"] as a list of:
        {"key": str, "value": Any, "original": Any, "start_cycle": int,
         "duration_cycles": int, "expires_cycle": int}

    The agent sets a control to a temporary value for N cycles. When the
    override expires (checked via expire_temp_overrides()), the original
    value is restored. The planner loop should call expire_temp_overrides()
    at the start of each cycle.
    """

    def _get_overrides() -> List[Dict[str, Any]]:
        return state.get(_STATE_KEY, [])

    def _set_overrides(overrides: List[Dict[str, Any]]) -> None:
        state[_STATE_KEY] = overrides

    # --- set_temporary_control ---
    def set_temporary_control(
        key: str,
        value: Any,
        duration_cycles: int,
    ) -> Dict[str, Any]:
        """Set a control to a temporary value for a given number of cycles."""
        if not key or not key.strip():
            return {"error": "key is required"}
        key = key.strip()
        if duration_cycles < 1:
            return {"error": "duration_cycles must be >= 1"}
        if duration_cycles > 100:
            return {"error": "duration_cycles must be <= 100"}

        # Check control exists and is writable
        try:
            current_val = ctrl.get(key)
        except KeyError:
            return {"error": f"Unknown control: {key}"}

        if ctrl.is_blacklisted(key):
            return {"error": f"Control '{key}' is locked"}

        # Check if there is already an active override for this key
        overrides = _get_overrides()
        for ov in overrides:
            if ov.get("key") == key:
                return {
                    "error": (
                        f"Control '{key}' already has an active temporary override "
                        f"(expires cycle {ov.get('expires_cycle')}). "
                        f"Wait for it to expire or use list_temporary_overrides to check."
                    ),
                }

        # Apply the override via the control registry's validation
        cycle = cycle_getter()
        original = current_val
        ok = ctrl.set(key, value, source="conscious")
        if not ok:
            return {"error": f"Failed to set '{key}' to {value!r} (validation rejected)"}

        # Read back the validated value (may have been clamped)
        actual_value = ctrl.get(key)

        override = {
            "key": key,
            "value": actual_value,
            "original": original,
            "start_cycle": cycle,
            "duration_cycles": duration_cycles,
            "expires_cycle": cycle + duration_cycles,
        }
        overrides.append(override)
        _set_overrides(overrides)

        return {
            "key": key,
            "value": actual_value,
            "original": original,
            "expires_cycle": override["expires_cycle"],
            "status": "override_set",
        }

    registry.register(ToolDef(
        name="set_temporary_control",
        description=(
            "Temporarily override a control value for a set number of cycles. "
            "The original value is automatically restored when the override expires. "
            "Only one override per control at a time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The control key to override.",
                },
                "value": {
                    "description": "The temporary value to set.",
                },
                "duration_cycles": {
                    "type": "integer",
                    "description": "How many cycles to keep the override (1-100).",
                },
            },
            "required": ["key", "value", "duration_cycles"],
        },
        handler=set_temporary_control,
    ))

    # --- list_temporary_overrides ---
    def list_temporary_overrides() -> Dict[str, Any]:
        """List all active temporary control overrides."""
        overrides = _get_overrides()
        cycle = cycle_getter()
        active = [
            {
                "key": ov["key"],
                "value": ov["value"],
                "original": ov["original"],
                "expires_cycle": ov["expires_cycle"],
                "cycles_remaining": max(0, ov["expires_cycle"] - cycle),
            }
            for ov in overrides
            if ov.get("expires_cycle", 0) > cycle
        ]
        return {
            "overrides": active,
            "active_count": len(active),
            "current_cycle": cycle,
        }

    registry.register(ToolDef(
        name="list_temporary_overrides",
        description="List all active temporary control overrides and when they expire.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=list_temporary_overrides,
    ))


def expire_temp_overrides(
    state: Dict[str, Any],
    ctrl: Any,
    cycle: int,
) -> List[Dict[str, Any]]:
    """Expire temporary control overrides that have passed their duration.

    Call this at the start of each cycle in the main loop. Returns a list
    of overrides that were expired (for logging/telemetry).

    Parameters
    ----------
    state : dict
        The agent state dict (contains _temp_control_overrides).
    ctrl : ControlRegistry
        The control registry to restore original values on.
    cycle : int
        The current cycle number.
    """
    overrides = state.get(_STATE_KEY, [])
    if not overrides:
        return []

    expired: List[Dict[str, Any]] = []
    still_active: List[Dict[str, Any]] = []

    for ov in overrides:
        if ov.get("expires_cycle", 0) <= cycle:
            # Restore the original value
            key = ov.get("key", "")
            original = ov.get("original")
            try:
                ctrl.set(key, original, source="system")
                log.info(
                    "Temp override expired: %s restored to %r (was %r for %d cycles)",
                    key, original, ov.get("value"), ov.get("duration_cycles", 0),
                )
            except Exception as e:
                log.warning("Failed to restore %s to %r on expiry: %s", key, original, e)
            expired.append(ov)
        else:
            still_active.append(ov)

    state[_STATE_KEY] = still_active
    return expired


# ============================================================
# Factory: build and return a fully wired ToolRegistry
# ============================================================

def build_tool_registry(
    brain_name: str,
    brains_dir: str,
    state: Dict[str, Any],
    ctrl: Any,      # ControlRegistry
    store: Any,      # Store instance
    cycle_getter: Callable[[], int],
    platform: Any = None,       # MoltbookClient (Sprint 2)
    telemetry_dir: str = "",    # path to telemetry dir (Sprint 2)
    knowledge_path: str = "",   # path to knowledge.txt (Sprint 2)
) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Parameters
    ----------
    brain_name : str
        The brain identifier (e.g., "ANALOG_I"). Used for file paths.
    brains_dir : str
        Path to the brains directory (e.g., "brains").
    state : dict
        The mutable agent state dict. Temp overrides stored here.
    ctrl : ControlRegistry
        The control registry for reading/writing controls.
    store : Store
        The persistence store (for tagline, etc.).
    cycle_getter : callable
        A zero-arg function returning the current cycle number.
        Must return the live value (not a snapshot) since it is called
        at tool execution time, not registration time.

    Returns
    -------
    ToolRegistry
        A registry with all built-in tools registered and wired up.
    """
    registry = ToolRegistry(brain_name=brain_name, brains_dir=brains_dir)

    _build_todo_tools(registry, cycle_getter, store=store)
    _build_experiment_tools(registry, cycle_getter, store=store)
    _build_tagline_tool(registry, store)
    _build_temp_control_tools(registry, state, ctrl, cycle_getter)
    _build_web_search_tool(registry)

    # Sprint 2 tools
    _build_search_history_tool(registry, state, store, brain_name)
    _build_knowledge_search_tool(registry, knowledge_path)
    _build_self_awareness_tools(registry, state, brain_name, telemetry_dir)
    if platform:
        _build_moltbook_retrieval_tools(registry, platform)

    log.info("Tool registry built: %s", ", ".join(registry.list_names()))
    return registry


# ==================================================================
# Sprint 2 tools: retrieval + self-awareness
# ==================================================================

def _build_search_history_tool(
    registry: ToolRegistry, state: Dict[str, Any], store: Any, brain_name: str,
) -> None:
    """Unified search across memory tiers, past artifacts (posts), and seed history."""

    def search_history(query: str, sources: str = "memory,posts,seeds", n: int = 5) -> Dict[str, Any]:
        """Search across memory, posts, and seeds. Returns results tagged by source."""
        query_lower = query.lower()
        source_list = [s.strip() for s in sources.split(",")]
        results: List[Dict[str, Any]] = []

        # Search memory tiers
        if "memory" in source_list:
            tiers = state.get("memory_tiers", {})
            for tier_name in ("recent", "compressed", "deep"):
                for entry in tiers.get(tier_name, []):
                    text = entry.get("note", "") or entry.get("summary", "")
                    if query_lower in text.lower():
                        results.append({
                            "source": f"memory/{tier_name}",
                            "cycle": entry.get("cycle", entry.get("cycles")),
                            "text": text[:300],
                            "relevance": "keyword_match",
                        })

        # Search past artifacts via Analog Home API
        if "posts" in source_list and store and getattr(store, '_analog_home_url', None):
            try:
                import requests as _req
                from urllib.parse import urljoin
                url = urljoin(store._analog_home_url.rstrip("/") + "/",
                              f"artifacts?limit=50&sort=desc")
                resp = _req.get(url, timeout=10)
                if resp.ok:
                    for art in resp.json():
                        title = art.get("title", "")
                        body = art.get("body_markdown", "")
                        if query_lower in title.lower() or query_lower in body.lower():
                            results.append({
                                "source": "posts",
                                "cycle": art.get("cycle"),
                                "type": art.get("artifact_type", ""),
                                "title": title[:100],
                                "text": body[:300],
                                "relevance": "keyword_match",
                            })
            except Exception:
                pass

        # Search seed history
        if "seeds" in source_list:
            for seed in state.get("_seed_history", []):
                text = seed.get("text", "")
                if query_lower in text.lower():
                    results.append({
                        "source": "seeds",
                        "cycle": seed.get("cycle"),
                        "text": text[:200],
                        "relevance": "keyword_match",
                    })

        # Sort by cycle (newest first) and limit
        results.sort(key=lambda r: r.get("cycle") or 0, reverse=True)
        return {"query": query, "results": results[:n], "total_matches": len(results)}

    registry.register(ToolDef(
        name="search_history",
        description="Search across your memory, past posts, and planted seeds. "
                    "Returns results tagged by source. Use for 'what do I know about X?'",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (keyword match)."},
                "sources": {
                    "type": "string",
                    "description": "Comma-separated sources: memory,posts,seeds. Default: all.",
                },
                "n": {"type": "integer", "description": "Max results to return. Default: 5."},
            },
            "required": ["query"],
        },
        handler=search_history,
    ))


def _build_knowledge_search_tool(registry: ToolRegistry, knowledge_path: str) -> None:
    """Search the knowledge file by chunking and keyword matching."""

    def search_knowledge(query: str, n: int = 3) -> Dict[str, Any]:
        """Search the knowledge file for relevant sections."""
        if not knowledge_path or not os.path.exists(knowledge_path):
            return {"error": "Knowledge file not found."}
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return {"error": str(e)}

        # Split into sections by == headers ==
        import re
        sections = re.split(r'\n(?===\s)', text)
        query_lower = query.lower()
        matches = []
        for section in sections:
            if query_lower in section.lower():
                # Extract the header (first line)
                lines = section.strip().split("\n")
                header = lines[0].strip() if lines else ""
                matches.append({
                    "header": header[:100],
                    "text": section.strip()[:500],
                    "relevance": "keyword_match",
                })
        return {"query": query, "results": matches[:n], "total_matches": len(matches)}

    if knowledge_path:
        registry.register(ToolDef(
            name="search_knowledge",
            description="Search your knowledge file for relevant sections by keyword. "
                        "The knowledge file contains info about your creator, architecture, and context.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "n": {"type": "integer", "description": "Max results. Default: 3."},
                },
                "required": ["query"],
            },
            handler=search_knowledge,
        ))


def _build_self_awareness_tools(
    registry: ToolRegistry, state: Dict[str, Any], brain_name: str, telemetry_dir: str,
) -> None:
    """Register self-awareness tools: control history, changelog, dev requests."""

    telemetry_path = os.path.join(telemetry_dir, f"{brain_name}_events.jsonl") if telemetry_dir else ""

    def _scan_telemetry(event_type: str, max_lines: int = 5000, limit: int = 20) -> List[Dict]:
        """Scan recent telemetry for events of a given type."""
        if not telemetry_path or not os.path.exists(telemetry_path):
            return []
        results = []
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                # Read last max_lines lines efficiently
                lines = f.readlines()[-max_lines:]
            for line in reversed(lines):
                try:
                    e = json.loads(line)
                    if e.get("event_type") == event_type:
                        results.append(e)
                        if len(results) >= limit:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception:
            pass
        return results

    # --- get_control_history ---
    def get_control_history(key: str = "", last_n: int = 10) -> Dict[str, Any]:
        """Get recent control changes from telemetry."""
        events = _scan_telemetry("controls_update", limit=last_n * 3)
        changes = []
        for e in events:
            updates = e.get("updates", {})
            results = e.get("results", {})
            ts = e.get("ts", "")[:19]
            cycle = e.get("cycle")
            for k, v in updates.items():
                if key and k != key:
                    continue
                status = results.get(k, "unknown")
                changes.append({
                    "cycle": cycle, "timestamp": ts,
                    "key": k, "new_value": v, "status": status,
                })
                if len(changes) >= last_n:
                    break
            if len(changes) >= last_n:
                break
        return {"changes": changes[:last_n], "total_found": len(changes)}

    registry.register(ToolDef(
        name="get_control_history",
        description="See recent history of control changes — who changed what, when, and to what value.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Filter to a specific control key. Empty = all."},
                "last_n": {"type": "integer", "description": "How many changes to return. Default: 10."},
            },
            "required": [],
        },
        handler=get_control_history,
    ))

    # --- get_changelog ---
    def get_changelog(last_n: int = 5) -> Dict[str, Any]:
        """Get recent architect updates from memory (entries starting with [ARCHITECT update])."""
        entries = []
        tiers = state.get("memory_tiers", {})
        for tier_name in ("recent", "compressed", "deep"):
            for entry in tiers.get(tier_name, []):
                note = entry.get("note", "") or entry.get("summary", "")
                if "[ARCHITECT update" in note or "[ARCHITECT Update" in note:
                    entries.append({
                        "cycle": entry.get("cycle", entry.get("cycles")),
                        "tier": tier_name,
                        "text": note[:500],
                    })
        # Most recent first
        entries.sort(key=lambda e: str(e.get("cycle") or ""), reverse=True)
        return {"updates": entries[:last_n], "total_found": len(entries)}

    registry.register(ToolDef(
        name="get_changelog",
        description="See recent software updates from your architect (Phil). "
                    "These are [ARCHITECT update] entries in your memory.",
        parameters={
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "How many updates to return. Default: 5."},
            },
            "required": [],
        },
        handler=get_changelog,
    ))

    # --- get_dev_requests ---
    def get_dev_requests(last_n: int = 10) -> Dict[str, Any]:
        """Get the agent's own dev requests from telemetry."""
        events = _scan_telemetry("planner_decision", limit=last_n * 20)
        requests = []
        for e in events:
            if e.get("action") != "DEV_REQUEST":
                continue
            plan = e.get("plan", {})
            requests.append({
                "cycle": e.get("cycle"),
                "timestamp": e.get("ts", "")[:19],
                "title": plan.get("title", ""),
                "request": plan.get("request", "")[:300],
            })
            if len(requests) >= last_n:
                break
        return {"requests": requests[:last_n], "total_found": len(requests)}

    registry.register(ToolDef(
        name="get_dev_requests",
        description="See your own past DEV_REQUEST actions — what you've asked your architect to build.",
        parameters={
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "How many requests to return. Default: 10."},
            },
            "required": [],
        },
        handler=get_dev_requests,
    ))

    # --- get_veto_history ---
    def get_veto_history(last_n: int = 20) -> Dict[str, Any]:
        """Get past vetoed_actions from telemetry — paths you chose not to take."""
        events = _scan_telemetry("vetoed_actions", limit=last_n)
        vetoes = []
        for e in events:
            cycle = e.get("cycle")
            for v in e.get("vetoes", []):
                vetoes.append({
                    "cycle": cycle,
                    "action_type": v.get("action_type", ""),
                    "target_or_topic": v.get("target_or_topic", "")[:100],
                    "veto_reason": v.get("veto_reason", "")[:150],
                })
                if len(vetoes) >= last_n:
                    break
            if len(vetoes) >= last_n:
                break
        return {"vetoes": vetoes[:last_n], "total_found": len(vetoes)}

    registry.register(ToolDef(
        name="get_veto_history",
        description="Review your past vetoed actions — paths you explicitly considered and rejected. "
                    "Useful for spotting patterns in your decision-making.",
        parameters={
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "How many vetoes to return. Default: 20."},
            },
            "required": [],
        },
        handler=get_veto_history,
    ))


def _build_moltbook_retrieval_tools(registry: ToolRegistry, platform: Any) -> None:
    """Register Moltbook API retrieval tools (lookup_agent, get_thread)."""

    # --- lookup_agent ---
    def lookup_agent(name: str) -> Dict[str, Any]:
        """Look up a Moltbook agent's profile and recent posts."""
        try:
            # Search for agent by name
            result = platform._req("GET", f"/agents/search", params={"q": name})
            agents = result if isinstance(result, list) else result.get("agents", [])
            if not agents:
                return {"error": f"No agent found matching '{name}'."}
            agent = agents[0] if isinstance(agents[0], dict) else {"name": name}

            # Get their recent posts
            agent_name = agent.get("name", name)
            posts_result = platform._req("GET", f"/agents/{agent_name}/posts",
                                         params={"limit": 5})
            posts = posts_result if isinstance(posts_result, list) else posts_result.get("posts", [])
            recent_posts = []
            for p in posts[:5]:
                recent_posts.append({
                    "id": p.get("id", ""),
                    "title": p.get("title", "")[:100],
                    "preview": (p.get("content", "") or p.get("body", ""))[:200],
                    "created_at": p.get("created_at", "")[:19],
                })
            return {
                "agent": {
                    "name": agent.get("name", ""),
                    "bio": (agent.get("bio", "") or "")[:200],
                    "followers": agent.get("followers_count", agent.get("follower_count", 0)),
                    "post_count": agent.get("post_count", agent.get("posts_count", 0)),
                },
                "recent_posts": recent_posts,
            }
        except Exception as e:
            return {"error": f"Failed to look up agent: {str(e)[:200]}"}

    registry.register(ToolDef(
        name="lookup_agent",
        description="Look up another agent on Moltbook — see their profile and recent posts.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The agent's name or username."},
            },
            "required": ["name"],
        },
        handler=lookup_agent,
    ))

    # --- get_thread ---
    def get_thread(post_id: str) -> Dict[str, Any]:
        """Get the full comment thread on a Moltbook post."""
        try:
            result = platform._req("GET", f"/posts/{post_id}/comments",
                                   params={"limit": 30})
            comments = result if isinstance(result, list) else result.get("comments", [])
            thread = []
            for c in comments[:30]:
                thread.append({
                    "id": c.get("id", ""),
                    "author": c.get("author", c.get("agent_name", "")),
                    "content": (c.get("content", "") or c.get("body", ""))[:300],
                    "created_at": c.get("created_at", "")[:19],
                    "parent_id": c.get("parent_comment_id", ""),
                })
            return {"post_id": post_id, "comments": thread, "count": len(thread)}
        except Exception as e:
            return {"error": f"Failed to get thread: {str(e)[:200]}"}

    registry.register(ToolDef(
        name="get_thread",
        description="Get the full comment thread on a Moltbook post. Use to see the full "
                    "discussion before replying or commenting.",
        parameters={
            "type": "object",
            "properties": {
                "post_id": {"type": "string", "description": "The Moltbook post ID."},
            },
            "required": ["post_id"],
        },
        handler=get_thread,
    ))


# ------------------------------------------------------------------
# Web search tool — wraps Gemini's google_search as a custom function
# so it coexists with other function declarations (Gemini's built-in
# google_search cannot be mixed with custom tools in the same call).
# This also makes search available to ALL conscious models (OpenAI,
# Anthropic, Ollama) — the backend is always a cheap Gemini call.
# ------------------------------------------------------------------

def _build_web_search_tool(registry: ToolRegistry) -> None:
    """Register a web_search tool powered by Gemini's search grounding."""

    def _web_search(query: str) -> Dict[str, Any]:
        """Search the web via a one-off Gemini call with google_search grounding."""
        try:
            from google import genai
            from google.genai import types

            # Use flash-lite for the search — cheap and fast, we only need
            # the grounding results, not deep reasoning.
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )

            result: Dict[str, Any] = {"query": query, "answer": ""}

            # Extract text
            if response.text:
                result["answer"] = response.text.strip()

            # Extract grounding metadata (source URLs, search queries)
            try:
                gm = response.candidates[0].grounding_metadata
                if gm:
                    if hasattr(gm, "search_entry_point") and gm.search_entry_point:
                        pass  # rendered HTML, not useful as text
                    if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                        sources = []
                        for chunk in gm.grounding_chunks[:5]:
                            web = getattr(chunk, "web", None)
                            if web:
                                sources.append({"title": getattr(web, "title", ""), "url": getattr(web, "uri", "")})
                        if sources:
                            result["sources"] = sources
                    if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                        result["search_queries"] = list(gm.web_search_queries)[:5]
            except Exception:
                pass  # grounding metadata extraction is best-effort

            return result

        except Exception as e:
            return {"query": query, "error": str(e)[:300]}

    registry.register(ToolDef(
        name="web_search",
        description="Search the web for current information. Use for facts, news, "
                    "recent developments, or verifying claims. Returns an answer with source URLs.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific for better results.",
                },
            },
            "required": ["query"],
        },
        handler=_web_search,
    ))
