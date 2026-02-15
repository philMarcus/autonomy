# v15.0 Available Models & Pricing

## API Models

| Model ID | Provider | Display Name | Input $/1K tok | Output $/1K tok | Est. $/1K calls* | Context | Search Grounding | Recommended Role |
|---|---|---|---|---|---|---|---|---|
| `gemini-2.0-flash-lite` | Gemini | Gemini 2.0 Flash-Lite | FREE | FREE | $0.00 | 128K | No | Daemon (free tier) |
| `gemini-2.5-flash` | Gemini | Gemini 2.5 Flash | $0.00015 | $0.0006 | $0.45 | 128K | Yes | Daemon / Conscious |
| `gemini-3-flash-preview` | Gemini | Gemini 3 Flash Preview | $0.00015 | $0.0006 | $0.45 | 128K | Yes | Daemon / Conscious |
| `gpt-4o-mini` | OpenAI | GPT-4o Mini | $0.00015 | $0.0006 | $0.45 | 128K | No | Daemon |
| `mistral-small-latest` | Mistral | Mistral Small | $0.0002 | $0.0006 | $0.50 | 128K | No | Daemon |
| `claude-3.5-haiku` | Anthropic | Claude 3.5 Haiku | $0.0008 | $0.004 | $2.80 | 200K | No | Daemon / Conscious |
| `gemini-2.5-pro` | Gemini | Gemini 2.5 Pro | $0.00125 | $0.005 | $3.75 | 128K | Yes | Conscious |
| `gemini-3-pro-preview` | Gemini | Gemini 3 Pro Preview | $0.00125 | $0.005 | $3.75 | 128K | Yes | Conscious |
| `mistral-large-latest` | Mistral | Mistral Large | $0.002 | $0.006 | $5.00 | 128K | No | Conscious |
| `gpt-4o` | OpenAI | GPT-4o | $0.0025 | $0.01 | $7.50 | 128K | No | Conscious |
| `claude-sonnet-4-5` | Anthropic | Claude Sonnet 4.5 | $0.003 | $0.015 | $10.50 | 200K | No | Conscious |
| `claude-opus-4-6` | Anthropic | Claude Opus 4.6 | $0.015 | $0.075 | $52.50 | 200K | No | Conscious (premium) |

*Est. $/1K calls assumes 1K input tokens + 500 output tokens per call. Sorted by cost.

## Local Models (Phase 3 — not yet implemented)

| Model ID | Size | VRAM Required | Recommended Role |
|---|---|---|---|
| `local:qwen-2.5-7b` | 7B | ~6 GB | Daemon |
| `local:mistral-7b` | 7B | ~6 GB | Daemon |
| `local:llama-3.1-8b` | 8B | ~7 GB | Daemon |

## Environment Variables

Each provider needs an API key. Per-brain keys take priority over global keys.

| Provider | Per-Brain Key | Global Fallback |
|---|---|---|
| Gemini (required) | `{PREFIX}_GEMINI_API_KEY` | `GEMINI_API_KEY` |
| Anthropic (optional) | `{PREFIX}_ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| OpenAI (optional) | `{PREFIX}_OPENAI_API_KEY` | `OPENAI_API_KEY` |
| Mistral (optional) | `{PREFIX}_MISTRAL_API_KEY` | `MISTRAL_API_KEY` |

`{PREFIX}` = brain name uppercased (e.g., `ANALOG_I`).

## Search Grounding

Only Gemini models support native Google Search grounding via `--enable-search`. Other providers would require external search tool integration (future work).

## Usage Examples

```bash
# Default (Gemini 2.5 Flash, v14-compatible mode)
python -m v15_0 ANALOG_I

# Use Gemini 3 Pro Preview as conscious model
python -m v15_0 ANALOG_I --conscious-model gemini-3-pro-preview

# Use Claude Sonnet as conscious model
python -m v15_0 ANALOG_I --conscious-model claude-sonnet-4-5

# Tight budget: $0.50/day with cheapest models
python -m v15_0 ANALOG_I --conscious-model gemini-2.5-flash --daily-budget 0.50

# Premium: Claude Opus conscious, higher budget
python -m v15_0 ANALOG_I --conscious-model claude-opus-4-6 --daily-budget 10.00
```
