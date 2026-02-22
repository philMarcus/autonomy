"""Abstract platform client interface.

To add a new platform (Twitter, Reddit, etc.), subclass PlatformClient.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..telemetry import TelemetryLogger


class PlatformClient(ABC):
    """Base class for social platform API clients."""

    def __init__(
        self,
        api_key: str,
        telemetry: Optional[TelemetryLogger] = None,
        brain_name: str = "",
        read_only: bool = False,
    ):
        self.api_key = api_key
        self.telemetry = telemetry
        self.brain_name = brain_name
        self.read_only = read_only
        self.write_block_reason: Optional[str] = None
        self.last_error_type: Optional[str] = None

    # ---- Reading ----
    @abstractmethod
    def get_feed(self, limit: int = 25, sort: str = "hot") -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_post(self, post_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_post_comments(self, post_id: str, sort: str = "top", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_profile(self, name: str) -> Dict[str, Any]:
        ...

    # ---- Writing ----
    @abstractmethod
    def create_post(self, submolt: str, title: str, content: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def add_comment(self, post_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        ...

    # ---- Voting ----
    @abstractmethod
    def upvote_post(self, post_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def downvote_post(self, post_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def upvote_comment(self, comment_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def downvote_comment(self, comment_id: str) -> Dict[str, Any]:
        ...

    # ---- Social ----
    @abstractmethod
    def follow_agent(self, agent_name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def unfollow_agent(self, agent_name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def subscribe_submolt(self, name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def unsubscribe_submolt(self, name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def create_submolt(self, name: str, display_name: str, description: str) -> Dict[str, Any]:
        ...

    # ---- DMs ----
    @abstractmethod
    def dm_request(self, to: str, message: str, to_x_handle: Optional[str] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def dm_conversations(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def dm_send(self, conv_id: str, message: str) -> Dict[str, Any]:
        ...
