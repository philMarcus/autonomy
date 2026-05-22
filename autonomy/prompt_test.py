"""Test a prompt_bench output against any Ollama model.

Usage:
    python -m autonomy.prompt_test gemma4:e4b strategist
    python -m autonomy.prompt_test gemma4:e4b conscious --temp 0.9
    python -m autonomy.prompt_test gemma4:e4b verifier --prompts-dir /tmp/prompts
    python -m autonomy.prompt_test gemma4:e4b strategist --save  # saves response to response.txt
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


def main():
    # Load .env so OLLAMA_URL is available
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

    parser = argparse.ArgumentParser(description="Test a prompt_bench gear against an Ollama model")
    parser.add_argument("model", help="Ollama model name (e.g. gemma4:e4b, gemma3:12b)")
    parser.add_argument("gear", help="Gear name (e.g. strategist, conscious, sentry_batch)")
    parser.add_argument("--prompts-dir", default="prompts", help="Prompts directory (default: prompts/)")
    parser.add_argument("--temp", type=float, help="Override temperature (default: from metadata.json)")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens")
    parser.add_argument("--ollama-url",
                        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        help="Ollama API URL (default: $OLLAMA_URL or localhost:11434)")
    parser.add_argument("--save", action="store_true", help="Save response to response.txt in gear dir")
    parser.add_argument("--no-stream", action="store_true", help="Wait for full response instead of streaming")
    args = parser.parse_args()

    gear_dir = os.path.join(args.prompts_dir, args.gear)
    if not os.path.isdir(gear_dir):
        print(f"Error: {gear_dir} not found. Run prompt_bench first:")
        print(f"  python -m autonomy.prompt_bench ANALOG_I")
        sys.exit(1)

    # Load prompt files
    system_path = os.path.join(gear_dir, "system.txt")
    user_path = os.path.join(gear_dir, "user.txt")
    meta_path = os.path.join(gear_dir, "metadata.json")

    with open(system_path, "r", encoding="utf-8") as f:
        system_text = f.read()
    with open(user_path, "r", encoding="utf-8") as f:
        user_text = f.read()

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    temp = args.temp if args.temp is not None else metadata.get("temperature", 0.7)
    max_tokens = args.max_tokens or metadata.get("max_output_tokens", 4096)

    # Check if system is a "none" placeholder
    has_system = system_text.strip() and not system_text.startswith("(none")

    print(f"{'=' * 60}")
    print(f"Model:       {args.model}")
    print(f"Gear:        {args.gear}")
    print(f"Temperature: {temp}")
    print(f"Max tokens:  {max_tokens}")
    print(f"System:      {len(system_text):,} chars" if has_system else "System:      (none)")
    print(f"User:        {len(user_text):,} chars")
    print(f"{'=' * 60}")
    print()

    # Build messages
    messages = []
    if has_system:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": args.model,
        "messages": messages,
        "stream": not args.no_stream,
        "options": {
            "temperature": temp,
            "num_predict": max_tokens,
        },
    }

    url = f"{args.ollama_url}/api/chat"
    start = time.time()

    try:
        if args.no_stream:
            print("Waiting for response...")
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.time() - start
            print(content)
            print(f"\n{'=' * 60}")
            print(f"Time: {elapsed:.1f}s  |  Tokens: ~{len(content) // 4}")
        else:
            # Stream
            resp = requests.post(url, json=payload, stream=True, timeout=300)
            resp.raise_for_status()
            full_response = []
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_response.append(token)
                    if chunk.get("done"):
                        break
            content = "".join(full_response)
            elapsed = time.time() - start
            print(f"\n\n{'=' * 60}")
            eval_count = 0
            try:
                # Last chunk has eval metrics
                eval_count = chunk.get("eval_count", 0)
                prompt_eval = chunk.get("prompt_eval_count", 0)
            except Exception:
                pass
            stats = f"Time: {elapsed:.1f}s"
            if eval_count:
                stats += f"  |  Output tokens: {eval_count}  |  {eval_count / elapsed:.1f} tok/s"
            if prompt_eval:
                stats += f"  |  Prompt tokens: {prompt_eval}"
            print(stats)

    except requests.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {args.ollama_url}")
        print("Is Ollama running? Try: ollama serve")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"Error: {e}")
        if "not found" in str(e).lower() or resp.status_code == 404:
            print(f"Model '{args.model}' not found. Try: ollama pull {args.model}")
        sys.exit(1)

    # Save response if requested
    if args.save and content:
        out_path = os.path.join(gear_dir, "response.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
