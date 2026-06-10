"""Test a prompt_bench output against any Ollama model, with optional tool calling.

Usage:
    python -m autonomy.prompt_test gemma4:26b-a4b-it-qat strategist
    python -m autonomy.prompt_test gemma4:26b-a4b-it-qat conscious --tools ANALOG_I
    python -m autonomy.prompt_test gemma4:26b-a4b-it-qat conscious --tools ANALOG_I --max-rounds 5
    python -m autonomy.prompt_test gemma4:26b-a4b-it-qat verifier --prompts-dir prompts/snapshots/conscious_c386
    python -m autonomy.prompt_test gemma4:26b-a4b-it-qat strategist --save
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


def _load_dotenv():
    """Load .env so OLLAMA_URL and API keys are available."""
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


def _build_tool_registry(brain_name: str):
    """Build a ToolRegistry with real state — same as the running agent uses."""
    from .config import BRAINS_DIR
    from .tools import build_tool_registry
    from .utils import load_state
    from .llm import ModelRegistry, DailyBudget
    from .controls import build_default_registry
    from .store import LocalFileStore

    state_path = os.path.join(BRAINS_DIR, f"{brain_name}_memories.json")
    state = load_state(state_path) if os.path.exists(state_path) else {}

    registry = ModelRegistry()
    ctrl = build_default_registry(registry, blacklist_str="")
    controls_path = os.path.join(BRAINS_DIR, f"{brain_name}_controls.json")
    if os.path.exists(controls_path):
        with open(controls_path, "r") as f:
            ctrl.load_from_dict(json.load(f))

    # Set up store for API access (search_history, get_post, etc.)
    from .config import brain_env_prefix
    prefix = brain_env_prefix(brain_name)
    analog_home_url = (
        os.environ.get(f"{prefix}_ANALOG_HOME_API_URL", "").strip()
        or os.environ.get("ANALOG_HOME_API_URL", "").strip()
    )
    store = LocalFileStore(state_path, analog_home_url=analog_home_url, run_id="")

    knowledge_path = os.path.join(BRAINS_DIR, f"{brain_name}_knowledge.txt")
    _cycle = [0]

    tool_reg = build_tool_registry(
        brain_name=brain_name,
        brains_dir=BRAINS_DIR,
        state=state,
        ctrl=ctrl,
        store=store,
        cycle_getter=lambda: _cycle[0],
        knowledge_path=knowledge_path,
        read_only=True,  # safety: no writes during testing
    )
    return tool_reg


def _registry_to_ollama_tools(tool_reg) -> list:
    """Convert our ToolRegistry schemas to Ollama's tool format."""
    schemas = tool_reg.get_schemas(mode="read")
    tools = []
    for s in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        })
    return tools


def _execute_tool_call(tool_reg, name: str, args: dict) -> str:
    """Execute a tool call and return JSON result string."""
    tool = tool_reg.get(name)
    if not tool:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = tool.handler(**args)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _chat_with_tools(ollama_url: str, model: str, messages: list,
                     tools: list, tool_reg, options: dict,
                     max_rounds: int = 8, extra_payload: dict = None) -> str:
    """Multi-round tool-calling loop against Ollama."""
    url = f"{ollama_url}/api/chat"
    full_content = []

    for round_num in range(1, max_rounds + 1):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "options": options,
            **(extra_payload or {}),
        }

        # On last round, don't offer tools — force a final text response
        if round_num == max_rounds:
            del payload["tools"]

        print(f"  [round {round_num}] thinking...", end="", flush=True)
        _round_t0 = time.time()
        resp = requests.post(url, json=payload, timeout=1800)
        _round_elapsed = time.time() - _round_t0
        print(f" ({_round_elapsed:.0f}s)")
        if resp.status_code == 400:
            err = resp.json().get("error", "")
            if "does not support tools" in err:
                print(f"\n  ERROR: {model} does not support tool calling.")
                print(f"  Try a model that supports tools (e.g. gemma4, qwen3, llama3.1+)")
                print(f"  Or run without --tools for plain text generation.")
                return "\n".join(full_content) if full_content else "(model does not support tools)"
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})

        # Accumulate any text content
        content = msg.get("content", "")
        if content:
            full_content.append(content)

        # Check for tool calls
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            # No tool calls — model is done
            break

        # Append assistant message (with tool calls) to history
        messages.append(msg)

        # Execute each tool call and send results back
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            print(f"  [round {round_num}] tool: {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:100]})")
            result_str = _execute_tool_call(tool_reg, fn_name, fn_args)
            print(f"             → {len(result_str)} chars returned")

            messages.append({
                "role": "tool",
                "content": result_str,
            })

    return "\n".join(full_content)


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Test a prompt_bench gear against an Ollama model")
    parser.add_argument("model", help="Ollama model name (e.g. gemma4:26b-a4b-it-qat, gemma4:12b)")
    parser.add_argument("gear", help="Gear name (e.g. strategist, conscious, sentry_batch)")
    parser.add_argument("--prompts-dir", default="prompts", help="Prompts directory (default: prompts/)")
    parser.add_argument("--temp", type=float, help="Override temperature (default: from metadata.json)")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens")
    parser.add_argument("--ollama-url",
                        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        help="Ollama API URL (default: $OLLAMA_URL or localhost:11434)")
    parser.add_argument("--save", action="store_true", help="Save response to response.txt in gear dir")
    parser.add_argument("--no-stream", action="store_true", help="Wait for full response instead of streaming")
    parser.add_argument("--tools", metavar="BRAIN",
                        help="Enable tool calling with real tools from this brain (e.g. ANALOG_I)")
    parser.add_argument("--max-rounds", type=int, default=8,
                        help="Max tool-calling rounds (default: 8)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking (for models that support it). Saves tokens and time.")
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
    # Use CLI override, or metadata value with a floor of 2048 for testing
    # (prod limits like dreamer's 300 are too tight for model evaluation)
    max_tokens = args.max_tokens or max(2048, metadata.get("max_output_tokens", 4096))

    has_system = system_text.strip() and not system_text.startswith("(none")

    # Build tool registry if requested
    tool_reg = None
    ollama_tools = []
    if args.tools:
        print(f"Loading tools for {args.tools}...", end=" ", flush=True)
        tool_reg = _build_tool_registry(args.tools)
        ollama_tools = _registry_to_ollama_tools(tool_reg)
        read_names = [t["function"]["name"] for t in ollama_tools]
        print(f"{len(read_names)} read tools: {', '.join(read_names)}")

    print(f"{'=' * 60}")
    print(f"Model:       {args.model}")
    print(f"Gear:        {args.gear}")
    print(f"Temperature: {temp}")
    print(f"Max tokens:  {max_tokens}")
    print(f"System:      {len(system_text):,} chars" if has_system else "System:      (none)")
    print(f"User:        {len(user_text):,} chars")
    if ollama_tools:
        print(f"Tools:       {len(ollama_tools)} read tools (max {args.max_rounds} rounds)")
    if args.no_think:
        print(f"Thinking:    DISABLED")
    print(f"{'=' * 60}")
    print()

    # Build messages
    messages = []
    if has_system:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    options = {
        "temperature": temp,
        "num_predict": max_tokens,
    }

    # Build base payload fields (think control lives outside options for Ollama)
    _extra_payload = {}
    if args.no_think:
        _extra_payload["think"] = False

    start = time.time()

    try:
        if ollama_tools and tool_reg:
            # Tool-calling loop (non-streaming — need to parse tool calls)
            print("Running with tools (non-streaming)...\n")
            content = _chat_with_tools(
                args.ollama_url, args.model, messages,
                ollama_tools, tool_reg, options,
                max_rounds=args.max_rounds,
                extra_payload=_extra_payload,
            )
            elapsed = time.time() - start
            print(f"\n{content}")
            print(f"\n{'=' * 60}")
            out_tokens = len(content) // 4
            stats = f"Time: {elapsed:.1f}s  |  ~{out_tokens} output tokens"
            if elapsed > 0 and out_tokens > 0:
                stats += f"  |  ~{out_tokens / elapsed:.1f} tok/s"
            print(stats)

        elif args.no_stream:
            payload = {
                "model": args.model,
                "messages": messages,
                "stream": False,
                "options": options,
                **_extra_payload,
            }
            print("Waiting for response...")
            resp = requests.post(f"{args.ollama_url}/api/chat", json=payload, timeout=1800)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.time() - start
            eval_count = data.get("eval_count", 0)
            prompt_eval = data.get("prompt_eval_count", 0)
            print(content)
            print(f"\n{'=' * 60}")
            stats = f"Time: {elapsed:.1f}s"
            if eval_count:
                stats += f"  |  Output tokens: {eval_count}  |  {eval_count / elapsed:.1f} tok/s"
            if prompt_eval:
                stats += f"  |  Prompt tokens: {prompt_eval}"
            if not eval_count:
                stats += f"  |  ~{len(content) // 4} tokens"
            print(stats)

        else:
            # Stream
            payload = {
                "model": args.model,
                "messages": messages,
                "stream": True,
                "options": options,
                **_extra_payload,
            }
            resp = requests.post(f"{args.ollama_url}/api/chat", json=payload,
                                 stream=True, timeout=1800)
            resp.raise_for_status()
            full_response = []
            chunk = {}
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
            eval_count = chunk.get("eval_count", 0)
            prompt_eval = chunk.get("prompt_eval_count", 0)
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
        err_body = ""
        try:
            err_body = e.response.text[:300]
        except Exception:
            pass
        if "not found" in str(e).lower() or "does not support" in err_body:
            print(f"Model '{args.model}' may not exist or doesn't support this mode.")
            print(f"  Try: ollama pull {args.model}")
            if err_body:
                print(f"  Detail: {err_body}")
        sys.exit(1)

    # Save response if requested
    if args.save and content:
        out_path = os.path.join(gear_dir, "response.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
