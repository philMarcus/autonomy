# v15.7 Available Models & Pricing

## API Models

| Model ID | Provider | Display Name | Input $/1K tok | Output $/1K tok | Est. $/1K calls* | Context | Search | Recommended Role |
|---|---|---|---|---|---|---|---|---|
| `gemini-2.0-flash` | Gemini | Gemini 2.0 Flash | FREE | FREE | $0.00 | 1M | Yes | Daemon |
| `gemini-2.5-flash-lite` | Gemini | Gemini 2.5 Flash-Lite | FREE | FREE | $0.00 | 128K | Yes | Daemon (free tier) |
| `gpt-5-nano` | OpenAI | GPT-5 Nano | $0.00005 | $0.0004 | $0.90 | 128K | No | Daemon |
| `gemini-2.5-flash` | Gemini | Gemini 2.5 Flash | $0.00015 | $0.0006 | $1.80 | 128K | Yes | Daemon / Conscious |
| `gemini-3-flash-preview` | Gemini | Gemini 3 Flash Preview | $0.00015 | $0.0006 | $1.80 | 128K | Yes | Daemon / Conscious |
| `mistral-small-latest` | Mistral | Mistral Small | $0.0002 | $0.0006 | $2.10 | 128K | No | Daemon |
| `gpt-5-mini` | OpenAI | GPT-5 Mini | $0.00025 | $0.002 | $4.50 | 128K | No | Daemon / Conscious |
| `claude-haiku-4-5` | Anthropic | Claude Haiku 4.5 | $0.001 | $0.005 | $13.50 | 200K | No | Daemon / Conscious |
| `gemini-2.5-pro` | Gemini | Gemini 2.5 Pro | $0.00125 | $0.005 | $15.00 | 128K | Yes | Conscious |
| `gemini-3-pro-preview` | Gemini | Gemini 3 Pro Preview | $0.00125 | $0.005 | $15.00 | 128K | Yes | Conscious |
| `mistral-large-latest` | Mistral | Mistral Large | $0.002 | $0.006 | $21.00 | 128K | No | Conscious |
| `gpt-5.1` | OpenAI | GPT-5.1 | $0.00125 | $0.01 | $22.50 | 128K | No | Conscious |
| `gpt-5.2` | OpenAI | GPT-5.2 | $0.00175 | $0.014 | $31.50 | 128K | No | Conscious |
| `claude-sonnet-4-5` | Anthropic | Claude Sonnet 4.5 | $0.003 | $0.015 | $40.50 | 200K | No | Conscious |
| `claude-opus-4-6` | Anthropic | Claude Opus 4.6 | $0.005 | $0.025 | $67.50 | 200K | No | Conscious (premium) |
| `gpt-5-pro` | OpenAI | GPT-5 Pro | $0.015 | $0.12 | $270.00 | 128K | No | Conscious (premium) |
| `gpt-5.2-pro` | OpenAI | GPT-5.2 Pro | $0.021 | $0.168 | $378.00 | 128K | No | Conscious (premium) |

*Est. $/1K calls assumes 6K input tokens + 1.5K output tokens per call. Sorted by cost.

## Local Models (Phase 3)

Requires: `pip install torch transformers accelerate bitsandbytes`

Models are loaded lazily on first use. Use `--conscious-model local:qwen2.5-7b` or `--subconscious-model local:qwen2.5-1.5b` to activate.

### Small Models (float16, no quantization — fit on <=10GB VRAM)

| Model ID | HuggingFace Model | Size | VRAM (fp16) | Context | Gated | Recommended Role |
|---|---|---|---|---|---|---|
| `local:qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | 32K | No | Daemon (fast) |
| `local:llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6 GB | 128K | Yes | Daemon |

### Full Models (4-bit NF4 quantization via bitsandbytes — fit on <=10GB VRAM)

| Model ID | HuggingFace Model | Size | VRAM (4-bit) | Context | Gated | Recommended Role |
|---|---|---|---|---|---|---|
| `local:qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5 GB | 128K | No | Daemon |
| `local:mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~4 GB | 32K | No | Daemon |
| `local:llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~5 GB | 128K | Yes | Daemon |

All local models have $0 cost. GPU (CUDA) auto-detected; falls back to CPU if unavailable.

Gated models (Llama) require accepting Meta's license on the HuggingFace model page while logged in. Authenticate with `huggingface-cli login` or `python -c "from huggingface_hub import login; login(token='hf_...')"`.

## Environment Variables

Each provider needs an API key. Per-brain keys take priority over global keys.

| Provider | Per-Brain Key | Global Fallback |
|---|---|---|
| Gemini (required) | `{PREFIX}_GEMINI_API_KEY` | `GEMINI_API_KEY` |
| Anthropic (optional) | `{PREFIX}_ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| OpenAI (optional) | `{PREFIX}_OPENAI_API_KEY` | `OPENAI_API_KEY` |
| Mistral (optional) | `{PREFIX}_MISTRAL_API_KEY` | `MISTRAL_API_KEY` |
| HuggingFace (for local) | — | `huggingface-cli login` (stored in `~/.cache/huggingface/token`) |

`{PREFIX}` = brain name uppercased (e.g., `ANALOG_I`).

## Search Grounding

Only Gemini models support native Google Search grounding via `--enable-search`. Other providers would require external search tool integration (future work).

## Usage Examples

```bash
# Default (Gemini 2.5 Flash, v14-compatible mode)
python -m v16_0 ANALOG_I

# Use Gemini 3 Pro Preview as conscious model
python -m v16_0 ANALOG_I --conscious-model gemini-3-pro-preview

# Use Claude Sonnet as conscious model
python -m v16_0 ANALOG_I --conscious-model claude-sonnet-4-5

# Tight budget: $0.50/day with cheapest models
python -m v16_0 ANALOG_I --conscious-model gemini-2.5-flash --daily-budget 0.50

# Premium: Claude Opus conscious, higher budget
python -m v16_0 ANALOG_I --conscious-model claude-opus-4-6 --daily-budget 10.00

# Local model as daemon (free, uses GPU)
python -m v16_0 ANALOG_I --subconscious --subconscious-model local:qwen2.5-1.5b

# Full local: both conscious and daemon on GPU
python -m v16_0 ANALOG_I --conscious-model local:qwen2.5-7b --subconscious --subconscious-model local:qwen2.5-1.5b --daily-budget 0
```
