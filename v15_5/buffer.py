"""Thread-safe draft buffer and wake potential tracker for the subconscious daemon.

The DraftBuffer is the communication channel between the daemon (writer)
and the conscious loop (reader).  It implements the integrate-and-fire
mechanism: drafts add charge, charge decays per tick, and when charge
crosses the wake threshold the conscious loop is signaled to fire.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Draft:
    """A single draft plan produced by the strategist gear."""
    timestamp: float          # time.time() when created
    item_id: str              # post/comment ID that triggered this
    signal_score: float       # sentry score (0.0-1.0)
    suggested_action: str     # POST, COMMENT, REPLY, UPVOTE, etc.
    target_summary: str       # brief description of the feed item
    reasoning: str            # why daemon thinks this matters
    draft_content: str        # rough suggested text
    charge: float             # how much wake_potential this added


# Maximum age (seconds) before a draft is pruned during decay.
_DRAFT_MAX_AGE = 1800  # 30 minutes


class DraftBuffer:
    """Thread-safe buffer of daemon drafts with integrate-and-fire wake logic."""

    def __init__(self, wake_threshold: float = 2.0, max_drafts: int = 10):
        self._lock = threading.Lock()
        self._drafts: List[Draft] = []
        self._wake_potential: float = 0.0
        self._wake_threshold: float = wake_threshold
        self._max_drafts: int = max_drafts
        self._wake_event = threading.Event()

    # ------------------------------------------------------------------
    # Writer side (daemon thread)
    # ------------------------------------------------------------------

    def add_draft(self, draft: Draft) -> None:
        """Add a draft and accumulate charge.  Signals wake if threshold crossed."""
        with self._lock:
            self._drafts.append(draft)
            self._wake_potential += draft.charge

            # Prune oldest if over capacity
            while len(self._drafts) > self._max_drafts:
                self._drafts.pop(0)

            if self._wake_potential >= self._wake_threshold:
                self._wake_event.set()

    def decay(self, rate: float) -> None:
        """Multiply wake_potential by rate and prune stale drafts.

        Called once per sentry tick.
        """
        now = time.time()
        with self._lock:
            self._wake_potential *= rate
            # Clamp very small values to zero
            if self._wake_potential < 0.001:
                self._wake_potential = 0.0
            # Prune old drafts
            self._drafts = [
                d for d in self._drafts
                if (now - d.timestamp) < _DRAFT_MAX_AGE
            ]

    # ------------------------------------------------------------------
    # Reader side (conscious thread)
    # ------------------------------------------------------------------

    def drain(self, refractory: float = 0.0) -> Tuple[List[Draft], float]:
        """Return all drafts and wake_potential, then reset the buffer.

        Called by the conscious loop when it wakes (by signal or timeout).
        Args:
            refractory: Value to set wake_potential to after draining.
                        Use negative values for a refractory period.
        """
        with self._lock:
            drafts = list(self._drafts)
            potential = self._wake_potential
            self._drafts.clear()
            self._wake_potential = refractory
            self._wake_event.clear()
            return drafts, potential

    def wait_for_wake(self, timeout: float) -> bool:
        """Block until the daemon signals a wake, or until timeout.

        Returns True if woken by the daemon, False if timed out.
        """
        return self._wake_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Configuration (called from either thread)
    # ------------------------------------------------------------------

    def update_threshold(self, threshold: float) -> None:
        """Update wake threshold (e.g. from ControlRegistry changes)."""
        with self._lock:
            self._wake_threshold = threshold
            # Re-check: if potential already exceeds new threshold, signal
            if self._wake_potential >= self._wake_threshold:
                self._wake_event.set()

    def update_max_drafts(self, max_drafts: int) -> None:
        """Update max draft capacity."""
        with self._lock:
            self._max_drafts = max_drafts

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def wake_potential(self) -> float:
        with self._lock:
            return self._wake_potential

    @property
    def draft_count(self) -> int:
        with self._lock:
            return len(self._drafts)
