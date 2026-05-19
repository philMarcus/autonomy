"""Mistral AI LLM backend for v15.0.

Implements ModelBackend for Mistral models via the Mistral Python SDK.
Requires: pip install mistralai
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo, ToolCall, ToolResult


# All Mistral models this backend serves
MISTRAL_MODELS: List[ModelInfo] = [
    ModelInfo(
        model_id="mistral-small-latest",
        provider="mistral",
        display_name="Mistral Small",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0006,
        max_context_tokens=128_000,
    ),
    ModelInfo(
        model_id="mistral-large-latest",
        provider="mistral",
        display_name="Mistral Large",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
        max_context_tokens=128_000,
    ),
]


class MistralChatSession(ChatSession):
    """Multi-turn chat via Mistral Chat API with manual history."""

    def __init__(self, client: Any, model_id: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int):
        self._client = client
        self.model_name = model_id
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._messages: List[Dict[str, str]] = []
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        if system_instruction:
            self._messages.append({"role": "system", "content": system_instruction})

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        self._messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(self._messages),
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.complete(**kwargs)
        text = resp.choices[0].message.content or ""

        # Capture accurate token counts from the API response
        if resp.usage:
            self._last_input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            self._last_output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0

        self._messages.append({"role": "assistant", "content": text})
        return text

    def send_message_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Dict],
        tool_executor: Callable[[List[ToolCall]], List[ToolResult]],
        max_rounds: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Tool calling stub — falls back to text-only send_message.

        TODO: implement native Mistral function calling.
        """
        log = logging.getLogger("autonomy.llm.mistral")
        log.debug("send_message_with_tools: native tool calling not yet "
                   "implemented for Mistral — falling back to send_message")
        return self.send_message(prompt, json_mode=json_mode)


class MistralBackend(ModelBackend):
    """Mistral provider backend — serves Mistral Small and Large."""

    def __init__(self, api_key: str):
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "Mistral backend requires the 'mistralai' package. "
                "Install it with: pip install mistralai"
            )
        self._client = Mistral(api_key=api_key)

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs,
    ) -> MistralChatSession:
        return MistralChatSession(
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
        resp = self._client.chat.complete(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)

        text = resp.choices[0].message.content or ""
        input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        return list(MISTRAL_MODELS)
