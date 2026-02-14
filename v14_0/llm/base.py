"""Abstract LLM client interface.

To add a new provider (OpenAI, Anthropic, etc.), subclass LLMClient and ChatSession.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ChatSession(ABC):
    """A stateful multi-turn chat session with an LLM."""

    @abstractmethod
    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        """Send a message and return the model's text response."""
        ...

    # Metadata slots for telemetry (set by the runner)
    _cycle: Optional[int] = None
    _telemetry: Any = None
    _brain_name: str = ""
    _last_llm_exception: Optional[Dict[str, Any]] = None
    _last_grounding_metadata: Any = None
    model_name: str = ""


class LLMClient(ABC):
    """Factory for chat sessions and one-shot generation."""

    @abstractmethod
    def create_chat(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 900,
        model: str = "",
        tools: Optional[list] = None,
    ) -> ChatSession:
        """Create a new chat session with the given system instruction."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_output_tokens: int = 200,
        model: str = "",
    ) -> str:
        """One-shot generation (no chat state). Returns text."""
        ...
