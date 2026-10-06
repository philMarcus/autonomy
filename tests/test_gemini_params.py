"""Gemini parameter policy: thinking_level in, sampling params only where honoured."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autonomy.llm import gemini as G  # noqa: E402


def _level(cfg):
    """The SDK stores thinking_level as an enum; compare its lower-cased value."""
    v = cfg.thinking_level
    return str(getattr(v, "value", v)).lower()


def test_version_parse_and_sampling_policy():
    assert G._gemini_version("gemini-3.8-flash") == (3, 8)
    assert G._gemini_version("gemini-2.5-pro") == (2, 5)
    assert G._gemini_version("gemini-3-pro-image") == (3, 0)
    assert G._gemini_version("gemini-pro-latest") is None
    # Sampling honoured below 3.6, dropped from 3.6 on and for "latest" aliases.
    assert G.model_supports_sampling("gemini-2.5-pro")
    assert G.model_supports_sampling("gemini-3.1-pro-preview")
    assert G.model_supports_sampling("gemini-3.5-flash-lite")
    assert not G.model_supports_sampling("gemini-3.6-flash")
    assert not G.model_supports_sampling("gemini-3.8-flash")
    assert not G.model_supports_sampling("gemini-flash-latest")
    # thinking_level only for 3.x
    assert G.model_supports_thinking_level("gemini-3.8-flash")
    assert G.model_supports_thinking_level("gemini-3.1-pro-preview")
    assert not G.model_supports_thinking_level("gemini-2.5-flash")


def test_normalize_thinking_level():
    assert G.normalize_thinking_level("HIGH") == "high"
    assert G.normalize_thinking_level(" low ") == "low"
    for bad in ("default", "", None, "max", 3):
        assert G.normalize_thinking_level(bad) is None


def test_config_kwargs_by_model():
    kw = G.sampling_and_thinking_kwargs("gemini-3.8-flash", 1.2, "high")
    assert "temperature" not in kw and _level(kw["thinking_config"]) == "high"
    kw = G.sampling_and_thinking_kwargs("gemini-2.5-pro", 1.2, "high")
    assert kw == {"temperature": 1.2}
    kw = G.sampling_and_thinking_kwargs("gemini-3.5-flash-lite", 0.3, "default")
    assert kw == {"temperature": 0.3}
    assert G.sampling_and_thinking_kwargs("gemini-3.8-flash", None, None) == {}


def test_session_uses_policy():
    s = G.GeminiChatSession(client=None, model_name="gemini-3.8-flash", system_instruction="",
                            temperature=0.9, max_output_tokens=10, thinking_level="medium")
    kw = s._sampling_kwargs()
    assert "temperature" not in kw and _level(kw["thinking_config"]) == "medium"
    s = G.GeminiChatSession(client=None, model_name="gemini-2.5-flash", system_instruction="",
                            temperature=0.9, max_output_tokens=10, thinking_level="medium")
    assert s._sampling_kwargs() == {"temperature": 0.9}
