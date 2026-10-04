# v19.0 Available Models

## Pricing

**Source of truth:** `autonomy/llm/pricing.json` (USD per 1K tokens, updated 2026-10-04).

The benchmark script warns if pricing.json is older than 30 days. Update it by editing the JSON file directly — reference links to each provider's pricing page are included in the file.

## API Providers

| Provider | Models Available | Search Grounding | Key Env Var |
|---|---|---|---|
| Gemini | 3.8 Flash, 3.5 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 3.1 Pro Preview, 2.5 Pro / Flash / Flash-Lite (legacy) | Yes | `GEMINI_API_KEY` |
| Anthropic | Claude Haiku 4.5, Sonnet 4.6, Opus 4.6 | No | `ANTHROPIC_API_KEY` |
| OpenAI | GPT-5 Nano/Mini, GPT-5.1/5.2, GPT-5 Pro/5.2 Pro, GPT-5.4 Nano/Mini | No | `OPENAI_API_KEY` |
| Mistral | Mistral Small, Mistral Large | No | `MISTRAL_API_KEY` |

Per-brain keys take priority: `{PREFIX}_GEMINI_API_KEY` > `GEMINI_API_KEY`. `{PREFIX}` = brain name uppercased (e.g., `ANALOG_I`).

### Deprecation Notices

- **Imagen 4 (all tiers)**: shut down Sep 2026. Images now come from the Gemini-native image models via `GeminiBackend.generate_image()`: `image-lite` = `gemini-3.1-flash-lite-image` (~$0.034/img, 1K only), `image-standard` = `gemini-3.1-flash-image` (~$0.067/img, default), `image-pro` = `gemini-3-pro-image` (~$0.134/img). Old `imagen-*` tier names in controls.json are aliased.
- **Gemini 3 Flash Preview / 3 Pro Preview / 3.1 Flash-Lite Preview**: removed from the API; dropped from the registry Oct 2026.
- **Gemini 2.5 family**: still served but Google restricts it for new projects. Kept as legacy entries; defaults moved to 3.8 Flash (conscious) and 3.5 Flash-Lite (seeker/verification).
- **Gemini 3.6–3.8 Flash promo pricing** ($0.75/$3.75 per M) ends 2026-12-31 — doubles after. Update pricing.json in January.
- **Gemini 2.0 Flash / Flash-Lite**: shut down June 1, 2026.

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

Run `python benchmark_models.py --role all` to test models on sentry/strategist/verification/compressor tasks. Results saved to `benchmark_results.json`.

## Usage Examples

```bash
# Default (conscious pool from controls: gemini-3.8-flash=1, gemini-3.1-pro-preview=0.15)
python -m autonomy ANALOG_I

# Premium conscious
python -m autonomy ANALOG_I --conscious-model gemini-3.1-pro-preview

# Local daemon (free, uses GPU)
python -m autonomy ANALOG_I --subconscious-model local:qwen2.5-7b

# Full local: both conscious and daemon on GPU
python -m autonomy ANALOG_I --conscious-model local:qwen2.5-7b --subconscious-model local:qwen2.5-1.5b --daily-budget 0

# Single-loop mode (no daemon)
python -m autonomy ANALOG_I --no-subconscious
```
