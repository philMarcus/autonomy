"""Thin persistence interface for state, artifacts, and controls.

Swap implementations (LocalFileStore → DuckDBStore → PostgresStore) to
change the backend without touching the agent loop or action logic.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from urllib.parse import urljoin

from . import utils

log = logging.getLogger(__name__)


class Store(ABC):
    """Persistence interface. All state reads/writes go through this."""

    # --- State (memories / rate-limits / history) ---
    @abstractmethod
    def load_state(self) -> Dict[str, Any]: ...

    @abstractmethod
    def save_state(self, state: Dict[str, Any]) -> None: ...

    # --- Artifacts ---
    def write_artifact(self, cycle: int, artifact: Dict[str, Any]) -> None:
        """Write a cycle artifact to the canonical archive."""

    # --- Controls (Analog I Phase 2) ---
    def read_controls(self) -> Dict[str, Any]:
        """Read external controls (temperature, focus keyword, etc.)."""
        return {}

    def increment_vote(self, vote_type: str, target_id: str) -> None:
        """Record a vote event (explore, exploit, reflect, etc.)."""


class LocalFileStore(Store):
    """File-backed state + HTTP artifact publishing to Analog_Home API."""

    def __init__(self, state_path: str, analog_home_url: str = ""):
        self._state_path = state_path
        self._analog_home_url = analog_home_url

    @property
    def state_path(self) -> str:
        return self._state_path

    def load_state(self) -> Dict[str, Any]:
        return utils.load_state(self._state_path)

    def save_state(self, state: Dict[str, Any]) -> None:
        utils.save_state(self._state_path, state)

    def write_artifact(self, cycle: int, artifact: Dict[str, Any]) -> None:
        """Publish artifact to Analog_Home API. Fails silently (log + continue)."""
        if not self._analog_home_url:
            return
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "publish")
            payload = {
                "id": artifact.get("id", int(time.time())),
                "brain": artifact.get("brain", ""),
                "cycle": cycle,
                "artifact_type": artifact.get("artifact_type", "post"),
                "title": artifact.get("title", ""),
                "body_markdown": artifact.get("body_markdown", ""),
                "monologue_public": artifact.get("monologue_public", ""),
                "channel": artifact.get("channel", ""),
                "source_platform": artifact.get("source_platform", ""),
                "source_id": artifact.get("source_id", ""),
                "source_parent_id": artifact.get("source_parent_id", ""),
                "source_url": artifact.get("source_url", ""),
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home publish failed (%s): %s", resp.status_code, resp.text[:200])
        except Exception as e:
            log.warning("analog_home publish error: %s", e)
