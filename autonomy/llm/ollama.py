"""Ollama backend for local model inference via REST API.

Ollama runs models natively with GGUF quantization and keeps them warm
in memory. Much faster than HuggingFace/PyTorch (1-5s vs 30-250s).

Requires: Ollama running at http://localhost:11434 (default)
Install: https://ollama.ai
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo, ToolCall, ToolResult

# ----------------------------------------------------------------------
# Context window sizing
#
# Ollama's default num_ctx is small (4096 in 0.35). Anything longer is
# silently truncated from the FRONT — the model loses the system prompt and
# the beginning of the user turn and answers from the tail. Our prompts run
# 10-50K chars, so every call sizes num_ctx to what it actually sends:
# generous token estimate + room for the answer, rounded up, floored at 8K,
# capped by the model's own window (from /api/show) and a sanity ceiling so
# a runaway prompt can't demand a KV cache the GPU can't hold.
# ----------------------------------------------------------------------
_CHARS_PER_TOKEN = 3.0        # conservative for English + markdown + JSON
_NUM_CTX_FLOOR = 8192
_NUM_CTX_CEILING = 65536
_NUM_CTX_MARGIN = 512
_model_ctx_cache: Dict[str, int] = {}


def _model_context_length(base_url: str, ollama_model: str) -> int:
    """The model's trained context window per /api/show (cached). 32768 if unknown."""
    if ollama_model in _model_ctx_cache:
        return _model_ctx_cache[ollama_model]
    ctx = 32768
    try:
        resp = requests.post(f"{base_url}/api/show", json={"model": ollama_model}, timeout=10)
        if resp.ok:
            info = resp.json().get("model_info") or {}
            for k, v in info.items():
                if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                    ctx = v
                    break
    except Exception:
        pass
    _model_ctx_cache[ollama_model] = ctx
    return ctx


def _num_ctx_for(chars: int, max_output: int, model_ctx: int) -> int:
    """Context window to request for a call that sends `chars` characters."""
    need = int(chars / _CHARS_PER_TOKEN) + max(0, int(max_output)) + _NUM_CTX_MARGIN
    need = ((need + 1023) // 1024) * 1024
    need = max(_NUM_CTX_FLOOR, need)
    return max(1024, min(need, _NUM_CTX_CEILING, model_ctx))


def _messages_chars(messages: List[Dict[str, Any]], extra: Any = None) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        total += len(c) if isinstance(c, str) else len(json.dumps(c, default=str))
        if m.get("tool_calls"):
            total += len(json.dumps(m["tool_calls"], default=str))
    if extra is not None:
        total += len(json.dumps(extra, default=str))
    return total


class OllamaChatSession(ChatSession):
    """Stateful multi-turn chat via Ollama's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        disable_thinking: bool = False,
    ):
        self.model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._disable_thinking = disable_thinking
        self._history: List[Dict[str, Any]] = []
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_thinking = ""
        self._model_id = model_name

        if system_instruction:
            self._history.append({"role": "system", "content": system_instruction})

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        """Send a message and return the model's text response."""
        self._history.append({"role": "user", "content": prompt})

        # Strip the "ollama:" prefix to get the actual Ollama model name
        ollama_model = self.model_name
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]

        payload: Dict[str, Any] = {
            "model": ollama_model,
            "messages": self._history,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_output_tokens,
                "num_ctx": _num_ctx_for(_messages_chars(self._history), self._max_output_tokens,
                                        _model_context_length(self._base_url, ollama_model)),
            },
        }
        # Disable thinking when requested (e.g. sentry scoring — thinking blocks confuse parsers)
        if self._disable_thinking:
            payload["think"] = False
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {}) or {}
        text = (msg.get("content") or "").strip()
        # Ollama returns the thought trace separately from content for thinking models
        # (qwen3, deepseek-r1). Callers can read `session._last_thinking` to surface
        # reasoning without risking JSON parse failures on the primary response.
        self._last_thinking = (msg.get("thinking") or "").strip()

        # Track token usage
        self._last_input_tokens = data.get("prompt_eval_count", 0) or 0
        self._last_output_tokens = data.get("eval_count", 0) or 0

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": text})

        return text

    @staticmethod
    def _schemas_to_ollama_tools(tool_schemas: List[Dict]) -> List[Dict[str, Any]]:
        """Convert our JSON-Schema tool defs to Ollama's OpenAI-style tool format."""
        tools = []
        for s in tool_schemas:
            tools.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("parameters") or {"type": "object", "properties": {}},
                },
            })
        return tools

    def send_message_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Dict],
        tool_executor: Callable[[List[ToolCall]], List[ToolResult]],
        max_rounds: int = 12,
        json_mode: bool = False,
    ) -> str:
        """Send a message with Ollama-native tool calling (/api/chat ``tools``).

        Flow mirrors the Gemini implementation:
        1. Send the user prompt with the tool list attached.
        2. If the reply carries ``message.tool_calls``, execute them through
           ``tool_executor`` and append one ``role: tool`` message per result.
        3. Repeat until the model answers in text or ``max_rounds`` is spent,
           in which case it is asked once more, without tools, for a final answer.

        Token counts are summed across rounds into ``_last_*_tokens``.
        ``json_mode`` is ignored while tools are attached (Ollama's ``format``
        and tool calls don't mix well); callers parse the JSON themselves.
        """
        log = logging.getLogger("autonomy.llm.ollama")
        ollama_model = self.model_name
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]
        tools = self._schemas_to_ollama_tools(tool_schemas)

        self._history.append({"role": "user", "content": prompt})
        total_in = 0
        total_out = 0

        def _chat(with_tools: bool) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model": ollama_model,
                "messages": self._history,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_output_tokens,
                    "num_ctx": _num_ctx_for(
                        _messages_chars(self._history, tools if with_tools else None),
                        self._max_output_tokens,
                        _model_context_length(self._base_url, ollama_model)),
                },
            }
            if with_tools and tools:
                payload["tools"] = tools
            if self._disable_thinking:
                payload["think"] = False
            resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()

        for round_idx in range(max_rounds + 1):
            final_round = round_idx == max_rounds
            if final_round:
                self._history.append({
                    "role": "user",
                    "content": "Tool budget exhausted. Respond now with your final answer "
                               "using what you already know.",
                })
            data = _chat(with_tools=not final_round)
            msg = data.get("message", {}) or {}
            total_in += data.get("prompt_eval_count", 0) or 0
            total_out += data.get("eval_count", 0) or 0
            self._last_thinking = (msg.get("thinking") or "").strip()
            text = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            # Keep the assistant turn (with its tool_calls) so the model sees its own requests.
            assistant_turn: Dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            self._history.append(assistant_turn)

            if not tool_calls or final_round:
                if tool_calls:
                    log.warning("send_message_with_tools: %s still requested tools on the final round",
                                ollama_model)
                break

            calls: List[ToolCall] = []
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                name = fn.get("name", "")
                calls.append(ToolCall(id=f"call_{name}_{round_idx}_{i}", name=name, args=args))
            log.info("Ollama tool round %d: %s", round_idx + 1, [c.name for c in calls])
            for result in tool_executor(calls):
                self._history.append({"role": "tool", "tool_name": result.name,
                                      "content": result.content})

        self._last_input_tokens = total_in
        self._last_output_tokens = total_out
        return text


class OllamaBackend(ModelBackend):
    """Ollama provider backend — serves models running in Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._models_cache: Optional[List[ModelInfo]] = None

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _discover_models(self) -> List[ModelInfo]:
        """Query Ollama for available models."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                if not name:
                    continue
                size_bytes = m.get("size", 0)
                size_gb = size_bytes / 1e9 if size_bytes else 0
                # Estimate context from model details (default 128K)
                details = m.get("details", {})
                param_size = details.get("parameter_size", "")

                models.append(ModelInfo(
                    model_id=f"ollama:{name}",
                    provider="ollama",
                    display_name=f"{name} ({size_gb:.1f}GB, Ollama)",
                    is_local=True,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    max_context_tokens=128_000,
                    supports_json_mode=True,
                ))
            return models
        except Exception:
            return []

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        disable_thinking: bool = False,
        **kwargs,
    ) -> OllamaChatSession:
        return OllamaChatSession(
            base_url=self._base_url,
            model_name=model_id,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            disable_thinking=disable_thinking,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """One-shot generation via /api/generate."""
        ollama_model = model_id
        if ollama_model.startswith("ollama:"):
            ollama_model = ollama_model[7:]

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
                "num_ctx": _num_ctx_for(len(prompt), max_output_tokens,
                                        _model_context_length(self._base_url, ollama_model)),
            },
        }
        # Disable thinking for tasks that need clean output (verification, scoring)
        if kwargs.get("disable_thinking", False):
            payload["think"] = False

        t0 = time.time()
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.time() - t0) * 1000)

        text = data.get("response", "").strip()
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        if self._models_cache is None:
            self._models_cache = self._discover_models()
        return self._models_cache
