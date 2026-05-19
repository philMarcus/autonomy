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
            "You may call these tools during your reasoning. Call as many as needed;",
            "results are returned before you produce your final JSON action.",
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

def _build_todo_tools(
    registry: ToolRegistry,
    cycle_getter: Callable[[], int],
) -> None:
    """Register todo list tools on the registry.

    Storage: brains/{brain}_todos.json — a list of todo items.
    """
    path = registry._path("todos.json")

    def _load_todos() -> List[Dict[str, Any]]:
        return _read_json_file(path, default=[])

    def _save_todos(todos: List[Dict[str, Any]]) -> None:
        _write_json_file(path, todos)

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
) -> None:
    """Register lab notebook (experiment tracking) tools on the registry.

    Storage: brains/{brain}_experiments.json — a list of experiments.
    """
    path = registry._path("experiments.json")

    def _load_experiments() -> List[Dict[str, Any]]:
        return _read_json_file(path, default=[])

    def _save_experiments(experiments: List[Dict[str, Any]]) -> None:
        _write_json_file(path, experiments)

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

    _build_todo_tools(registry, cycle_getter)
    _build_experiment_tools(registry, cycle_getter)
    _build_tagline_tool(registry, store)
    _build_temp_control_tools(registry, state, ctrl, cycle_getter)

    log.info("Tool registry built: %s", ", ".join(registry.list_names()))
    return registry
