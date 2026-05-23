"""Google Gemini LLM backend for v15.0.

Implements ModelBackend with all Gemini models.  Retains the stateless
generate_content approach with manual history tracking (works around the
Gemini 2.5 Flash first-message empty-response bug).
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo, ToolCall, ToolResult
from ..config import (
    LLM_TPM_SOFT_CAP, LLM_TPM_CHAR_TO_TOKEN, LLM_TPM_WINDOW_SECONDS,
    LLM_BACKOFF_INITIAL_SECONDS, LLM_BACKOFF_MAX_SECONDS,
)


# ============================================================
# TPM budgeting (approximate) + 429 backoff
# ============================================================
class GeminiBudget:
    def __init__(self):
        self._events: List[Tuple[float, int]] = []
        self._backoff_s = 0.0
        self._blocked_until = 0.0

    def _prune(self) -> None:
        now = time.time()
        self._events = [(t, tok) for (t, tok) in self._events if now - t < LLM_TPM_WINDOW_SECONDS]

    def est_tokens(self, prompt_chars: int) -> int:
        try:
            return int(max(0.0, float(prompt_chars)) / float(LLM_TPM_CHAR_TO_TOKEN))
        except Exception:
            return int(prompt_chars / 4)

    def should_throttle(self, est_tokens: int) -> Tuple[bool, int]:
        self._prune()
        used = sum(tok for _, tok in self._events)
        return (used + est_tokens) > LLM_TPM_SOFT_CAP, used

    def record(self, est_tokens: int) -> None:
        self._prune()
        self._events.append((time.time(), int(est_tokens)))

    def blocked_remaining(self) -> float:
        try:
            return max(0.0, float(self._blocked_until) - time.time())
        except Exception:
            return 0.0

    def note_429(self) -> float:
        if self._backoff_s <= 0:
            self._backoff_s = LLM_BACKOFF_INITIAL_SECONDS
        else:
            self._backoff_s = min(LLM_BACKOFF_MAX_SECONDS, self._backoff_s * 2)
        self._blocked_until = time.time() + float(self._backoff_s)
        return self._backoff_s

    def reset_backoff(self) -> None:
        self._backoff_s = 0.0
        self._blocked_until = 0.0


# Module-level singleton budget tracker
BUDGET = GeminiBudget()


# ============================================================
# Gemini ChatSession — stateless generate_content with history
# ============================================================
class GeminiChatSession(ChatSession):
    def __init__(self, client: genai.Client, model_name: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int,
                 tools: Optional[list] = None):
        self._client = client
        self.model_name = model_name
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._tools = tools
        self._history: List[types.Content] = []

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        prompt_chars = len(prompt or "")
        est_tokens = BUDGET.est_tokens(prompt_chars)

        # Global cooldown after 429s
        rem = BUDGET.blocked_remaining()
        if rem > 0:
            time.sleep(float(rem))

        # TPM guardrail
        should_throttle, used = BUDGET.should_throttle(est_tokens)
        if should_throttle:
            sleep_s = max(1.0, LLM_TPM_WINDOW_SECONDS - 1.0)
            time.sleep(sleep_s)

        # Build contents: history + new user message
        contents = list(self._history) + [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        config_kwargs: Dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if self._system_instruction:
            config_kwargs["system_instruction"] = self._system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if self._tools:
            config_kwargs["tools"] = self._tools

        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except TypeError:
            config_kwargs.pop("response_mime_type", None)
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            # Trigger exponential backoff on 429 / rate-limit errors
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                wait = BUDGET.note_429()
                # Re-raise so caller sees the error
            raise

        # Capture grounding metadata if available
        self._last_grounding_metadata = None
        try:
            self._last_grounding_metadata = resp.candidates[0].grounding_metadata
        except (IndexError, AttributeError):
            pass

        # Capture token usage for cost tracking
        # Gemini 2.5+ Pro models have THINKING tokens billed at output rate but
        # reported in a separate field. Must add them to output tokens or we
        # underestimate cost by 2-10x for reasoning models.
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        try:
            usage = resp.usage_metadata
            self._last_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            visible_out = getattr(usage, "candidates_token_count", 0) or 0
            thinking = getattr(usage, "thoughts_token_count", 0) or 0
            self._last_output_tokens = visible_out + thinking
        except (AttributeError, TypeError):
            pass

        raw = self._extract_text(resp)

        # Append exchange to history for multi-turn support
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        if raw:
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=raw)])
            )

        BUDGET.record(est_tokens)
        BUDGET.reset_backoff()
        return raw

    @staticmethod
    def _extract_text(resp) -> str:
        raw = getattr(resp, "text", "") or ""
        if not raw:
            try:
                parts = resp.candidates[0].content.parts
                raw = "\n".join(p.text for p in parts if getattr(p, "text", None))
            except Exception:
                raw = ""
        return raw

    # --------------------------------------------------------
    # Tool-calling support (v18)
    # --------------------------------------------------------

    @staticmethod
    def _schemas_to_declarations(
        tool_schemas: List[Dict],
    ) -> List[types.FunctionDeclaration]:
        """Convert JSON-schema tool definitions to Gemini FunctionDeclaration objects.

        Each schema dict must have at minimum:
            {"name": str, "description": str, "parameters": {...}}
        where ``parameters`` follows JSON Schema (type/properties/required).
        """
        decls: List[types.FunctionDeclaration] = []
        for schema in tool_schemas:
            params = schema.get("parameters")
            decls.append(types.FunctionDeclaration(
                name=schema["name"],
                description=schema.get("description", ""),
                parameters=params,
            ))
        return decls

    @staticmethod
    def _extract_function_calls(resp) -> List[ToolCall]:
        """Pull ToolCall objects out of a Gemini response, if any."""
        calls: List[ToolCall] = []
        try:
            parts = resp.candidates[0].content.parts
        except (IndexError, AttributeError):
            return calls
        if not parts:
            return calls
        for idx, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Build a stable call ID from the function name + index
                call_id = f"call_{fc.name}_{idx}"
                # fc.args is a proto MapComposite — convert to plain dict
                args = dict(fc.args) if fc.args else {}
                calls.append(ToolCall(id=call_id, name=fc.name, args=args))
        return calls

    def send_message_with_tools(
        self,
        prompt: str,
        tool_schemas: List[Dict],
        tool_executor: Callable[[List[ToolCall]], List[ToolResult]],
        max_rounds: int = 12,
        json_mode: bool = False,
    ) -> str:
        """Send a message with Gemini-native function calling.

        Flow:
        1. Send the user prompt with function declarations attached.
        2. If the model responds with function_call parts, execute them
           via ``tool_executor`` and feed function_response parts back.
        3. Repeat until the model returns a text response or max_rounds
           is exhausted.

        Budget tracking, 429 backoff, history management, and token
        counting are all preserved — every generate_content call goes
        through the same guardrails as send_message.
        """
        log = logging.getLogger("autonomy.llm.gemini")

        # Convert tool schemas to Gemini FunctionDeclarations
        declarations = self._schemas_to_declarations(tool_schemas)
        fn_tools = [types.Tool(function_declarations=declarations)]

        # Gemini does NOT allow mixing built-in tools (google_search) with
        # custom function declarations in the same API call — returns 400
        # INVALID_ARGUMENT. When custom tools are present, drop built-in
        # tools. Search grounding is still available via the seeker daemon's
        # pre-loaded findings in the prompt context.
        combined_tools = fn_tools  # custom only, no built-in google_search

        # --- Initial user turn ---------------------------------------------------
        prompt_chars = len(prompt or "")
        est_tokens = BUDGET.est_tokens(prompt_chars)

        # Global cooldown after 429s
        rem = BUDGET.blocked_remaining()
        if rem > 0:
            time.sleep(float(rem))

        # TPM guardrail
        should_throttle, used = BUDGET.should_throttle(est_tokens)
        if should_throttle:
            sleep_s = max(1.0, LLM_TPM_WINDOW_SECONDS - 1.0)
            time.sleep(sleep_s)

        # Build contents: history + new user message
        base_contents = list(self._history) + [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        config_kwargs: Dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "tools": combined_tools,
        }
        if self._system_instruction:
            config_kwargs["system_instruction"] = self._system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        # Context caching is LAZY: only created if round 1 returns tool calls
        # and we need round 2+. Single-turn cycles (majority) pay zero cache overhead.
        _cache_name = None
        _base_config_kwargs = dict(config_kwargs)  # save for cache creation later
        contents = base_contents

        # --- Multi-round tool loop -----------------------------------------------
        total_input_tokens = 0
        total_output_tokens = 0
        accumulated_text = []  # capture visible text from ALL rounds (not just final)

        for round_idx in range(max_rounds + 1):  # +1 so we can do max_rounds tool exchanges
            try:
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except TypeError:
                # Some older SDK versions don't support response_mime_type — retry without
                config_kwargs.pop("response_mime_type", None)
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    BUDGET.note_429()
                raise

            # Accumulate token usage across rounds
            try:
                usage = resp.usage_metadata
                total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                visible_out = getattr(usage, "candidates_token_count", 0) or 0
                thinking = getattr(usage, "thoughts_token_count", 0) or 0
                total_output_tokens += visible_out + thinking
            except (AttributeError, TypeError):
                pass

            BUDGET.record(est_tokens)
            # After the first round, est_tokens for subsequent rounds is small
            # (just the function response), so reset for TPM accounting.
            est_tokens = 0

            # Capture any visible text from this round (model may produce text
            # alongside tool calls — e.g. [INTERNAL MONOLOGUE] before calling tools)
            _round_text = self._extract_text(resp)
            if _round_text and _round_text.strip():
                accumulated_text.append(_round_text)

            # Check for function calls in the response
            tool_calls = self._extract_function_calls(resp)

            if not tool_calls:
                # No tool calls — model produced a final text response
                break

            # --- Execute tool calls and build response parts ----------------------
            log.debug("Tool round %d: %d call(s) — %s",
                      round_idx, len(tool_calls),
                      ", ".join(tc.name for tc in tool_calls))

            # Lazy context caching: cache the base prompt so round 2+ pays 90% less.
            # The cache holds the base (system instruction + initial prompt + tool decls).
            # contents is reset to [] so round 2 only sends the tool call/response
            # turns as delta — these get appended below.
            if not _cache_name and round_idx == 0:
                try:
                    cache_config = types.CreateCachedContentConfig(
                        display_name=f"planner-{id(self)}",
                        system_instruction=self._system_instruction or "",
                        contents=base_contents,
                        tools=combined_tools,
                        ttl="120s",
                    )
                    _cache = self._client.caches.create(
                        model=self.model_name,
                        config=cache_config,
                    )
                    _cache_name = _cache.name
                    config_kwargs.pop("system_instruction", None)
                    config_kwargs.pop("tools", None)
                    config_kwargs["cached_content"] = _cache_name
                    # Start fresh — tool call/response turns appended below
                    contents = []
                    log.info("Context cache created (lazy): %s", _cache_name)
                except Exception as _cache_err:
                    log.debug("Context caching unavailable (%s), continuing with full context", _cache_err)

            results = tool_executor(tool_calls)

            # Build a map for quick lookup
            result_map: Dict[str, ToolResult] = {r.call_id: r for r in results}

            # Append model's function_call turn to contents
            try:
                model_content = resp.candidates[0].content
                contents.append(model_content)
            except (IndexError, AttributeError):
                # Shouldn't happen if we got tool_calls, but guard anyway
                break

            # Build function_response parts (one per call, matched by name)
            fn_response_parts: List[types.Part] = []
            for tc in tool_calls:
                tr = result_map.get(tc.id)
                if tr is None:
                    # Executor didn't return a result for this call — send empty
                    result_payload = {"error": "no result returned"}
                else:
                    # Parse the content string back to dict for Gemini
                    try:
                        result_payload = json.loads(tr.content)
                    except (json.JSONDecodeError, TypeError):
                        result_payload = {"result": tr.content}

                fn_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=tc.name,
                        response=result_payload,
                    )
                ))

            contents.append(
                types.Content(role="user", parts=fn_response_parts)
            )
        else:
            # Exhausted max_rounds — send one final turn WITHOUT tools to force
            # the model to produce a text response instead of another tool call.
            log.warning("send_message_with_tools: exhausted %d rounds, forcing final response", max_rounds)
            try:
                _final_config = {
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_output_tokens,
                }
                if self._system_instruction:
                    _final_config["system_instruction"] = self._system_instruction
                contents.append(types.Content(role="user", parts=[
                    types.Part(text="You have used all available tool rounds. Produce your final JSON action now based on what you've gathered so far.")
                ]))
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**_final_config),
                )
            except Exception:
                pass  # fall through to text extraction with whatever we have

        # --- Post-loop: capture metadata and update history -----------------------

        # Grounding metadata from final response
        self._last_grounding_metadata = None
        try:
            self._last_grounding_metadata = resp.candidates[0].grounding_metadata
        except (IndexError, AttributeError):
            pass

        # Store cumulative token counts for cost tracking
        self._last_input_tokens = total_input_tokens
        self._last_output_tokens = total_output_tokens

        # Combine text from all rounds — earlier rounds may contain [INTERNAL MONOLOGUE]
        # or other visible reasoning that preceded tool calls. The final round's text
        # (typically just JSON) is already in accumulated_text from the loop above.
        raw = "\n".join(accumulated_text) if accumulated_text else ""

        # Append the original user prompt and final model text to session
        # history so subsequent send_message calls see the full conversation.
        # (Intermediate tool-call/response turns are intentionally omitted
        # from session history to keep it clean for follow-up turns.)
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        if raw:
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=raw)])
            )

        BUDGET.reset_backoff()

        # Clean up the context cache (if created)
        if _cache_name:
            try:
                self._client.caches.delete(name=_cache_name)
                log.debug("Context cache deleted: %s", _cache_name)
            except Exception:
                pass  # best-effort cleanup; cache TTL will expire anyway

        return raw


# ============================================================
# Gemini ModelBackend
# ============================================================

# All Gemini models this backend serves
GEMINI_MODELS: List[ModelInfo] = [
    # --- 2.5 family ---
    ModelInfo(
        model_id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0025,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.5-flash-lite",
        provider="gemini",
        display_name="Gemini 2.5 Flash-Lite",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.01,
        supports_tools=True,
    ),
    # --- 3.x family ---
    ModelInfo(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash Preview",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.003,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro Preview",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.012,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        display_name="Gemini 3.1 Pro Preview",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.012,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-3.1-flash-lite-preview",
        provider="gemini",
        display_name="Gemini 3.1 Flash-Lite Preview",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.0015,
        supports_tools=True,
    ),
    # --- 2.0 family (deprecated June 2026) ---
    ModelInfo(
        model_id="gemini-2.0-flash",
        provider="gemini",
        display_name="Gemini 2.0 Flash (deprecated)",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
        supports_tools=True,
    ),
    ModelInfo(
        model_id="gemini-2.0-flash-lite",
        provider="gemini",
        display_name="Gemini 2.0 Flash-Lite (deprecated)",
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
    ),
]


class GeminiBackend(ModelBackend):
    """Gemini provider backend — serves all Gemini model variants."""

    def __init__(self, api_key: str, timeout: int = 120):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout * 1000),  # ms
        )

    @property
    def raw_client(self) -> genai.Client:
        """Access the underlying genai.Client (e.g. for challenge solvers)."""
        return self._client

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs,
    ) -> GeminiChatSession:
        return GeminiChatSession(
            client=self._client,
            model_name=model_id,
            system_instruction=system_instruction or "",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.time()
        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        latency_ms = int((time.time() - t0) * 1000)
        text = (response.text or "").strip()

        # Try to extract token counts from usage metadata
        # Include thoughts_token_count (billed at output rate for thinking models)
        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            visible_out = getattr(usage, "candidates_token_count", 0) or 0
            thinking = getattr(usage, "thoughts_token_count", 0) or 0
            output_tokens = visible_out + thinking
        except (AttributeError, TypeError):
            pass

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    # Imagen model tiers: fast ($0.02), standard ($0.04), ultra ($0.06)
    IMAGEN_MODELS = {
        "imagen-fast": {"model_id": "imagen-4.0-fast-generate-001", "cost": 0.02},
        "imagen-standard": {"model_id": "imagen-4.0-generate-001", "cost": 0.04},
        "imagen-ultra": {"model_id": "imagen-4.0-ultra-generate-001", "cost": 0.06},
    }

    def generate_image(
        self,
        prompt: str,
        tier: str = "imagen-ultra",
        aspect_ratio: str = "1:1",
    ) -> tuple:
        """Generate an image from a text prompt using Imagen.

        Args:
            tier: "imagen-fast" ($0.02), "imagen-standard" ($0.04), or "imagen-ultra" ($0.06)

        Returns (raw_png_bytes, model_id, cost_usd).
        """
        info = self.IMAGEN_MODELS.get(tier, self.IMAGEN_MODELS["imagen-ultra"])
        model_id = info["model_id"]
        cost = info["cost"]

        response = self._client.models.generate_images(
            model=model_id,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )
        if not response.generated_images:
            raise RuntimeError("Imagen returned no images")
        return response.generated_images[0].image.image_bytes, model_id, cost

    def available_models(self) -> List[ModelInfo]:
        return list(GEMINI_MODELS)
