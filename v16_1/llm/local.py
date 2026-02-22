"""Local model backend for v15.0 (HuggingFace transformers).

Implements ModelBackend for locally-hosted models via HuggingFace transformers.
Models are loaded lazily on first use to avoid VRAM allocation at startup.
Uses 4-bit quantization (bitsandbytes) on CUDA to fit 7B+ models in <=10GB VRAM.

Requires: pip install torch transformers accelerate bitsandbytes
"""

import time
from typing import Any, Dict, List, Optional

from .base import ChatSession, LLMResponse, ModelBackend, ModelInfo


# Map our model IDs to HuggingFace hub identifiers
HF_MODEL_MAP: Dict[str, str] = {
    # Small models (fit float16 on <=10GB VRAM, no quantization needed)
    "local:qwen2.5-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "local:llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    # Full models (need 4-bit quantization on <=10GB VRAM)
    "local:qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "local:mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    "local:llama-3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
}

LOCAL_MODELS: List[ModelInfo] = [
    ModelInfo(
        model_id="local:qwen2.5-1.5b",
        provider="local",
        display_name="Qwen2.5 1.5B Instruct (local)",
        is_local=True,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_context_tokens=32_768,
    ),
    ModelInfo(
        model_id="local:llama-3.2-3b",
        provider="local",
        display_name="Llama 3.2 3B (local)",
        is_local=True,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_context_tokens=128_000,
    ),
    ModelInfo(
        model_id="local:qwen2.5-7b",
        provider="local",
        display_name="Qwen2.5 7B Instruct (local)",
        is_local=True,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_context_tokens=128_000,
    ),
    ModelInfo(
        model_id="local:mistral-7b",
        provider="local",
        display_name="Mistral 7B (local)",
        is_local=True,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_context_tokens=32_768,
    ),
    ModelInfo(
        model_id="local:llama-3.1-8b",
        provider="local",
        display_name="Llama 3.1 8B (local)",
        is_local=True,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_context_tokens=128_000,
    ),
]


def _detect_device() -> str:
    """Detect best available device: cuda > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class _LoadedModel:
    """Container for a loaded model + tokenizer pair."""
    __slots__ = ("model", "tokenizer", "device")

    def __init__(self, model: Any, tokenizer: Any, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


class LocalChatSession(ChatSession):
    """Multi-turn chat via local HuggingFace model with chat template."""

    def __init__(self, loaded: _LoadedModel, model_id: str,
                 system_instruction: str, temperature: float,
                 max_output_tokens: int):
        self._loaded = loaded
        self.model_name = model_id
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._messages: List[Dict[str, str]] = []
        if system_instruction:
            self._messages.append({"role": "system", "content": system_instruction})

    def send_message(self, prompt: str, json_mode: bool = False) -> str:
        import torch

        self._messages.append({"role": "user", "content": prompt})

        tokenizer = self._loaded.tokenizer
        model = self._loaded.model

        input_text = tokenizer.apply_chat_template(
            self._messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(self._loaded.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._max_output_tokens,
            "do_sample": self._temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self._temperature > 0:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        self._messages.append({"role": "assistant", "content": text})
        return text


class LocalBackend(ModelBackend):
    """Local model provider — serves HuggingFace models on GPU/CPU."""

    def __init__(self):
        self._device = _detect_device()
        self._loaded: Dict[str, _LoadedModel] = {}  # model_id -> loaded

    def _ensure_loaded(self, model_id: str) -> _LoadedModel:
        """Lazy-load model + tokenizer on first use."""
        if model_id in self._loaded:
            return self._loaded[model_id]

        hf_name = HF_MODEL_MAP.get(model_id)
        if not hf_name:
            raise ValueError(
                f"Unknown local model: {model_id!r}. "
                f"Available: {sorted(HF_MODEL_MAP.keys())}"
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Local backend requires 'transformers' and 'torch'. "
                "Install with: pip install torch transformers accelerate bitsandbytes"
            )

        print(f"[LOCAL] Loading {hf_name} on {self._device}...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Models <=3B fit in float16 on 10GB VRAM; larger need 4-bit quantization
        SMALL_MODELS = {"local:qwen2.5-1.5b", "local:llama-3.2-3b"}
        needs_quant = model_id not in SMALL_MODELS

        load_kwargs: Dict[str, Any] = {}
        if self._device == "cuda":
            import torch
            if needs_quant:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    print("[LOCAL] bitsandbytes not installed, falling back to float16")
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = {"": "cuda:0"}
        else:
            load_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)

        # Report actual device placement
        try:
            actual = next(model.parameters()).device
        except StopIteration:
            actual = self._device
        elapsed = time.time() - t0
        print(f"[LOCAL] {hf_name} loaded in {elapsed:.1f}s (device={actual})")

        loaded = _LoadedModel(model=model, tokenizer=tokenizer, device=self._device)
        self._loaded[model_id] = loaded
        return loaded

    def create_chat(
        self,
        model_id: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        tools: Optional[list] = None,
    ) -> LocalChatSession:
        loaded = self._ensure_loaded(model_id)
        return LocalChatSession(
            loaded=loaded,
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
        import torch

        loaded = self._ensure_loaded(model_id)
        tokenizer = loaded.tokenizer
        model = loaded.model

        t0 = time.time()

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(loaded.device)
        input_token_count = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][input_token_count:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        output_token_count = len(new_ids)

        latency_ms = int((time.time() - t0) * 1000)

        return LLMResponse(
            text=text.strip(),
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            cost_usd=0.0,
            model_id=model_id,
            latency_ms=latency_ms,
        )

    def available_models(self) -> List[ModelInfo]:
        return list(LOCAL_MODELS)
