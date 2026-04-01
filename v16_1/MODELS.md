# v16.2 Available Models

## Pricing

**Source of truth:** `v16_1/llm/pricing.json` (USD per 1K tokens, updated 2026-03-31).

The benchmark script warns if pricing.json is older than 30 days. Update it by editing the JSON file directly — reference links to each provider's pricing page are included in the file.

## API Providers

| Provider | Models Available | Search Grounding | Key Env Var |
|---|---|---|---|
| Gemini | 2.5 Flash, 2.5 Flash-Lite, 2.5 Pro, 3 Flash Preview, 3 Pro Preview, 3.1 Pro Preview, 3.1 Flash-Lite Preview, 2.0 Flash (deprecated), 2.0 Flash-Lite (deprecated) | Yes | `GEMINI_API_KEY` |
| Anthropic | Claude Haiku 4.5, Sonnet 4.6, Opus 4.6 | No | `ANTHROPIC_API_KEY` |
| OpenAI | GPT-5 Nano/Mini, GPT-5.1/5.2, GPT-5 Pro/5.2 Pro, GPT-5.4 Nano/Mini | No | `OPENAI_API_KEY` |
| Mistral | Mistral Small, Mistral Large | No | `MISTRAL_API_KEY` |

Per-brain keys take priority: `{PREFIX}_GEMINI_API_KEY` > `GEMINI_API_KEY`. `{PREFIX}` = brain name uppercased (e.g., `ANALOG_I`).

### Deprecation Notices

- **Gemini 2.0 Flash / Flash-Lite**: Shut down June 1, 2026. Migrate to 2.5 Flash-Lite (cheapest) or 2.5 Flash.

## Local Models

Requires: `pip install torch transformers accelerate bitsandbytes`

Models are loaded lazily on first use. Use `--conscious-model local:qwen2.5-7b` or `--subconscious-model local:qwen2.5-1.5b` to activate.

### Small Models (float16, no quantization — fit on <=10GB VRAM)

| Model ID | HuggingFace Model | Size | VRAM (fp16) | Context | Gated |
|---|---|---|---|---|---|
| `local:qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | 32K | No |
| `local:llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6 GB | 128K | Yes |

### Full Models (4-bit NF4 quantization via bitsandbytes — fit on <=10GB VRAM)

| Model ID | HuggingFace Model | Size | VRAM (4-bit) | Context | Gated |
|---|---|---|---|---|---|
| `local:qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5 GB | 128K | No |
| `local:mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~4 GB | 32K | No |
| `local:llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~5 GB | 128K | Yes |

All local models have $0 cost. GPU (CUDA) auto-detected; falls back to CPU if unavailable.

Gated models (Llama) require accepting Meta's license on the HuggingFace model page while logged in. Authenticate with `huggingface-cli login`.

## Search Grounding

Only Gemini models support native Google Search grounding via `--enable-search`.

## Benchmarking

Run `python -c "from v16_1.benchmark import main; main()"` to test models on sentry/strategist/planner tasks. Results saved to `benchmark_results.json`.

## Usage Examples

```bash
# Default (Gemini 2.5 Flash conscious, single-loop mode)
python -m v16_1 ANALOG_I

# Budget daemon: 2.5 Flash-Lite is cheapest API option
python -m v16_1 ANALOG_I --subconscious --subconscious-model gemini-2.5-flash-lite

# Premium conscious with search
python -m v16_1 ANALOG_I --conscious-model gemini-3.1-pro-preview --enable-search

# Local daemon (free, uses GPU)
python -m v16_1 ANALOG_I --subconscious --subconscious-model local:qwen2.5-7b

# Full local: both conscious and daemon on GPU
python -m v16_1 ANALOG_I --conscious-model local:qwen2.5-7b --subconscious --subconscious-model local:qwen2.5-1.5b --daily-budget 0
```
