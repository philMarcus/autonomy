"""Thin persistence interface for state, artifacts, and controls.

Swap implementations (LocalFileStore → DuckDBStore → PostgresStore) to
change the backend without touching the agent loop or action logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from . import utils


class Store(ABC):
    """Persistence interface. All state reads/writes go through this."""

    # --- State (memories / rate-limits / history) ---
    @abstractmethod
    def load_state(self) -> Dict[str, Any]: ...

    @abstractmethod
    def save_state(self, state: Dict[str, Any]) -> None: ...

    # --- Analog I Phase 2 stubs (no-op until integration) ---

    def write_artifact(self, cycle: int, artifact: Dict[str, Any]) -> None:
        """Write a cycle artifact (title, body_markdown, monologue_public)."""

    def read_controls(self) -> Dict[str, Any]:
        """Read external controls (temperature, focus keyword, etc.)."""
        return {}

    def increment_vote(self, vote_type: str, target_id: str) -> None:
        """Record a vote event (upvote_post, downvote_post, etc.)."""


class LocalFileStore(Store):
    """File-backed store using JSON — wraps existing utils functions."""

    def __init__(self, state_path: str):
        self._state_path = state_path

    @property
    def state_path(self) -> str:
        return self._state_path

    def load_state(self) -> Dict[str, Any]:
        return utils.load_state(self._state_path)

    def save_state(self, state: Dict[str, Any]) -> None:
        utils.save_state(self._state_path, state)
