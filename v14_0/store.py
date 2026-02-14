"""Thin persistence interface for state, artifacts, and controls.

Swap implementations (LocalFileStore → DuckDBStore → PostgresStore) to
change the backend without touching the agent loop or action logic.
"""

import json
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
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

    def set_trajectory(self, label_1: str, label_2: str, label_3: str, reason: str = "", default_temperature: float | None = None) -> bool:
        """Reset vote buttons with 3 new labels. Returns True on success."""
        return False


class LocalFileStore(Store):
    """File-backed state + HTTP artifact publishing to Analog_Home API."""

    def __init__(self, state_path: str, analog_home_url: str = ""):
        self._state_path = state_path
        self._analog_home_url = analog_home_url
        self._pending_path = state_path.replace("_memories.json", "_pending_artifacts.json")

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

    def set_trajectory(self, label_1: str, label_2: str, label_3: str, reason: str = "", default_temperature: float | None = None) -> bool:
        """POST new trajectory labels to Analog Home API."""
        if not self._analog_home_url:
            return False
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "set-trajectory")
            payload = {
                "label_1": label_1,
                "label_2": label_2,
                "label_3": label_3,
                "reason": reason,
            }
            if default_temperature is not None:
                payload["default_temperature"] = default_temperature
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home set_trajectory failed (%s): %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as e:
            log.warning("analog_home set_trajectory error: %s", e)
            return False

    # --- Pending artifact queue (retry on API failure) ---

    def _load_pending(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self._pending_path):
                with open(self._pending_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_pending(self, pending: List[Dict[str, Any]]) -> None:
        try:
            with open(self._pending_path, "w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False)
        except Exception as e:
            log.warning("Failed to save pending artifacts: %s", e)

    def _flush_pending(self) -> None:
        """Try to publish any queued artifacts from previous failures."""
        if not self._analog_home_url:
            return
        pending = self._load_pending()
        if not pending:
            return
        import requests
        url = urljoin(self._analog_home_url.rstrip("/") + "/", "publish")
        still_pending = []
        for payload in pending:
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code >= 400:
                    still_pending.append(payload)
                    log.warning("Retry publish failed (%s): %s", resp.status_code, resp.text[:200])
                else:
                    log.info("Retry publish succeeded for artifact %s", payload.get("id"))
            except Exception:
                still_pending.append(payload)
        self._save_pending(still_pending)

    def write_artifact(self, cycle: int, artifact: Dict[str, Any]) -> None:
        """Publish artifact to Analog_Home API. Queues on failure for retry."""
        if not self._analog_home_url:
            return

        # Flush any previously failed artifacts first
        self._flush_pending()

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
            "search_queries": artifact.get("search_queries", ""),
            "temperature": artifact.get("temperature"),
        }
        try:
            import requests
            url = urljoin(self._analog_home_url.rstrip("/") + "/", "publish")
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code >= 400:
                log.warning("analog_home publish failed (%s): %s — queuing for retry", resp.status_code, resp.text[:200])
                self._queue_artifact(payload)
        except Exception as e:
            log.warning("analog_home publish error: %s — queuing for retry", e)
            self._queue_artifact(payload)

    def _queue_artifact(self, payload: Dict[str, Any]) -> None:
        """Add a failed artifact to the pending queue (max 50)."""
        pending = self._load_pending()
        pending.append(payload)
        # Cap at 50 to avoid unbounded growth
        if len(pending) > 50:
            pending = pending[-50:]
        self._save_pending(pending)
