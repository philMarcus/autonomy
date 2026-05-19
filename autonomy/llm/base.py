"""LLM abstractions for v15.0 multi-model support.

Two layers:
  - Legacy interface (LLMClient / ChatSession): kept for backward compat with
    planner.py, actions.py, challenges/, etc.  send_message() returns str.
  - New interface (ModelBackend / ModelRegistry / LLMResponse): multi-provider,
    cost-aware.  generate() returns LLMResponse.

GeminiBackend implements ModelBackend and also satisfies the LLMClient interface
via the CompatAdapter, so existing code works unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ============================================================
# New v15 types
# ============================================================

@dataclass
class ModelInfo:
    """Static metadata about a model."""
    model_id: str                      # e.g. "gemini-2.5-flash", "local:qwen-2.5-7b"
    provider: str                      # "gemini", "anthropic", "openai", "mistral", "local"
    display_name: str = ""
    is_local: bool = False
    input_cost_per_1k: float = 0.0     # USD per 1K input tokens (0 for local)
    output_cost_per_1k: float = 0.0    # USD per 1K output tokens (0 for local)
    max_context_tokens: int = 128_000
    supports_json_mode: bool = True
    supports_tools: bool = False


@dataclass
class LLMResponse:
    """Structured response from any model."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_id: str = ""
    latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Tool-calling types (v18 — used by send_message_with_tools)
# ============================================================

@dataclass
class ToolCall:
    """A tool/function call requested by the model."""
    id: str                    # provider-assigned call ID
    name: str                  # function name
    args: Dict[str, Any] = field(default_factory=dict)  # parsed arguments


@dataclass
class ToolResult:
    """Result of executing a tool call, sent back to the model."""
    call_id: str               # must match ToolCall.id
    name: str                  # function name (echoed back)
    content: str               # JSON-serialized result string


# ============================================================
# Legacy interface (v14 compat — planner/actions still use this)
# ============================================================

class ChatSession(ABC):
    """A stateful multi-turn chat session with an LLM."""

    @abstractmethod
    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        """Send a message and return the model's text response."""
        ...

    def send_message_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Dict],
        tool_executor: Callable[[List[ToolCall]], List[ToolResult]],
        max_rounds: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Send message with tool support.

        If the model requests tool calls, executes them via tool_executor
        and continues the conversation until the model produces a final text
        response (or max_rounds is exhausted).

        Default implementation: falls back to send_message (ignores tools).
        Backends that support native tool calling override this.
        """
        return self.send_message(prompt, json_mode=json_mode)

    # Metadata slots for telemetry (set by the runner)
    _cycle: Optional[int] = None
    _telemetry: Any = None
    _brain_name: str = ""
    _last_llm_exception: Optional[Dict[str, Any]] = None
    _last_grounding_metadata: Any = None
    model_name: str = ""


class LLMClient(ABC):
    """Factory for chat sessions and one-shot generation (v14 interface)."""

    @abstractmethod
    def create_chat(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 900,
        model: str = "",
        tools: Optional[list] = None,
        **kwargs,
    ) -> ChatSession:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_output_tokens: int = 200,
        model: str = "",
    ) -> str:
        """One-shot generation. Returns text (v14 compat)."""
        ...


# ============================================================
# New v15 backend interface
# ============================================================

class ModelBackend(ABC):
    """Multi-model provider backend.

    Each provider (Gemini, Anthropic, OpenAI, Mistral, Local) implements this
    once.  The model_id parameter selects among the provider's models.
    """

    @abstractmethod
    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs,
    ) -> ChatSession:
        ...

    @abstractmethod
    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        """One-shot generation. Returns structured LLMResponse."""
        ...

    @abstractmethod
    def available_models(self) -> List[ModelInfo]:
        """List models this backend can serve."""
        ...


# ============================================================
# Adapter: makes a ModelBackend + model_id look like an LLMClient
# so that existing v14 code (planner, actions, challenges) works.
# ============================================================

class CompatAdapter(LLMClient):
    """Wraps a ModelBackend + default model_id into the v14 LLMClient interface."""

    def __init__(self, backend: ModelBackend, default_model_id: str):
        self._backend = backend
        self._default_model_id = default_model_id

    def create_chat(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 900,
        model: str = "",
        tools: Optional[list] = None,
        **kwargs,
    ) -> ChatSession:
        model_id = model or self._default_model_id
        return self._backend.create_chat(
            model_id=model_id,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_output_tokens: int = 200,
        model: str = "",
    ) -> str:
        model_id = model or self._default_model_id
        resp = self._backend.generate(
            model_id=model_id,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return resp.text

    @property
    def backend(self) -> ModelBackend:
        return self._backend
