"""Abstract challenge solver interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..telemetry import TelemetryLogger


class ChallengeSolver(ABC):
    """Base class for solving platform verification challenges."""

    def __init__(self, telemetry: Optional[TelemetryLogger] = None):
        self.telemetry = telemetry

    @abstractmethod
    def can_solve(self, challenge_data: Dict[str, Any]) -> bool:
        """Return True if this solver can handle the given challenge."""
        ...

    @abstractmethod
    def solve(self, challenge_data: Dict[str, Any]) -> Optional[str]:
        """Attempt to solve the challenge. Returns the answer string or None."""
        ...
