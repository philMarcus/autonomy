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

    # --- Controls ---
    def read_controls(self) -> Dict[str, Any]:
        """Read external controls (temperature, seeds, trajectory votes)."""
        return {}

    def consume_seeds(self, seed_ids: list) -> bool:
        """Delete seeds by ID after the agent has read them. Returns True on success."""
        return False

    def set_trajectory(self, label_1: str, label_2: str, label_3: str) -> bool:
        """Reset vote buttons with 3 new labels. Returns True on success."""
        return False


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

    def read_controls(self) -> Dict[str, Any]:
        """Read controls + seeds from Analog Home API."""
        if not self._analog_home_url:
            return {}
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "state")
            resp = requests.get(url, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home read_controls failed (%s): %s", resp.status_code, resp.text[:200])
                return {}
            data = resp.json()
            controls = data.get("controls", {})
            seeds = data.get("seeds", [])
            return {
                "temperature": controls.get("temperature", 0.7),
                "vote_1": controls.get("vote_1", 0),
                "vote_2": controls.get("vote_2", 0),
                "vote_3": controls.get("vote_3", 0),
                "vote_label_1": controls.get("vote_label_1", "emergence"),
                "vote_label_2": controls.get("vote_label_2", "entropy"),
                "vote_label_3": controls.get("vote_label_3", "self"),
                "seeds": [s.get("text", "") for s in seeds if s.get("text")],
                "seed_ids": [s.get("id") for s in seeds if s.get("id") is not None],
            }
        except Exception as e:
            log.warning("analog_home read_controls error: %s", e)
            return {}

    def consume_seeds(self, seed_ids: list) -> bool:
        """DELETE seeds by ID after the agent has read them."""
        if not self._analog_home_url or not seed_ids:
            return False
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "consume-seeds")
            resp = requests.post(url, json={"ids": seed_ids}, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home consume_seeds failed (%s): %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as e:
            log.warning("analog_home consume_seeds error: %s", e)
            return False

    def set_trajectory(self, label_1: str, label_2: str, label_3: str) -> bool:
        """POST new trajectory labels to Analog Home API."""
        if not self._analog_home_url:
            return False
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "set-trajectory")
            resp = requests.post(url, json={
                "label_1": label_1,
                "label_2": label_2,
                "label_3": label_3,
            }, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home set_trajectory failed (%s): %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as e:
            log.warning("analog_home set_trajectory error: %s", e)
            return False

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
