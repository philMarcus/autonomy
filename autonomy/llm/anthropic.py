"""Anthropic (Claude) LLM backend for v15.0.

Implements ModelBackend for Claude models via the Anthropic Python SDK.
Requires: pip install anthropic
"""

import time
from typing import Any, Dict, List, Optional

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo


# All Anthropic models this backend serves
ANTHROPIC_MODELS: List[ModelInfo] = [
    ModelInfo(
        model_id="claude-haiku-4-5",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
        max_context_tokens=200_000,
    ),
    ModelInfo(
        model_id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_context_tokens=200_000,
    ),
    ModelInfo(
        model_id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.025,
        max_context_tokens=200_000,
    ),
]


class AnthropicChatSession(ChatSession):
    """Multi-turn chat via Anthropic Messages API with manual history."""

    def __init__(self, client: Any, model_id: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int):
        self._client = client
        self.model_name = model_id
        self._system = system_instruction
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._history: List[Dict[str, str]] = []

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        self._history.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "messages": list(self._history),
        }
        if self._system:
            kwargs["system"] = self._system

        resp = self._client.messages.create(**kwargs)

        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text

        self._history.append({"role": "assistant", "content": text})
        return text


class AnthropicBackend(ModelBackend):
    """Anthropic provider backend — serves all Claude models."""

    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic backend requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs,
    ) -> AnthropicChatSession:
        return AnthropicChatSession(
            client=self._client,
            model_id=model_id,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        resp = self._client.messages.create(
            model=model_id,
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text

        input_tokens = getattr(resp.usage, "input_tokens", 0) or 0
        output_tokens = getattr(resp.usage, "output_tokens", 0) or 0

        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        return list(ANTHROPIC_MODELS)
