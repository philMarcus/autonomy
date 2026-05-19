"""OpenAI LLM backend for v15.0.

Implements ModelBackend for GPT models via the OpenAI Python SDK.
Requires: pip install openai
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo, ToolCall, ToolResult


# All OpenAI models this backend serves
# Models that don't support the temperature parameter (reasoning models)
_NO_TEMPERATURE = {"gpt-5-nano", "gpt-5-mini"}

OPENAI_MODELS: List[ModelInfo] = [
    # GPT 5.4 family (latest, March 2026)
    # Pricing per 1K tokens (from per-million: nano $0.20/$1.25, mini $0.75/$4.50, 5.4 $2.50/$15)
    ModelInfo(
        model_id="gpt-5.4-nano",
        provider="openai",
        display_name="GPT-5.4 Nano",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.00125,
        max_context_tokens=128_000,
        supports_json_mode=True,
    ),
    ModelInfo(
        model_id="gpt-5.4-mini",
        provider="openai",
        display_name="GPT-5.4 Mini",
        input_cost_per_1k=0.00075,
        output_cost_per_1k=0.0045,
        max_context_tokens=128_000,
        supports_json_mode=True,
    ),
    ModelInfo(
        model_id="gpt-5.4",
        provider="openai",
        display_name="GPT-5.4",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.015,
        max_context_tokens=128_000,
        supports_json_mode=True,
    ),
]


class OpenAIChatSession(ChatSession):
    """Multi-turn chat via OpenAI Chat Completions with manual history."""

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
            "max_completion_tokens": self._max_output_tokens,
        }
        if self.model_name not in _NO_TEMPERATURE:
            kwargs["temperature"] = self._temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
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

        TODO: implement native OpenAI function calling.
        """
        log = logging.getLogger("autonomy.llm.openai")
        log.debug("send_message_with_tools: native tool calling not yet "
                   "implemented for OpenAI — falling back to send_message")
        return self.send_message(prompt, json_mode=json_mode)


class OpenAIBackend(ModelBackend):
    """OpenAI provider backend — serves GPT models."""

    def __init__(self, api_key: str):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI backend requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        self._client = openai.OpenAI(api_key=api_key)

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs,
    ) -> OpenAIChatSession:
        return OpenAIChatSession(
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
        gen_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_output_tokens,
        }
        if model_id not in _NO_TEMPERATURE:
            gen_kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**gen_kwargs)
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
        return list(OPENAI_MODELS)
