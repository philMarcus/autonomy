"""
title: Analog I Inspector
author: Phil Marcus
description: Read-only tools to inspect the Analog I autonomous agent — daemon gear I/O, gear instructions, and published artifacts.
required_open_webui_version: 0.4.0
version: 0.1.0
license: MIT

Setup
-----
The Open WebUI container needs a read-only mount of the agent's brains directory.
Recreate the container with:

    -v C:\\Users\\Phil\\autonomy_prod\\brains:/app/brains:ro

Then install this file via Open WebUI admin panel > Workspace > Tools > Create
(paste the file contents). Configure paths in the tool's Valves if they differ
from the defaults.
"""

import json
import os
import urllib.parse
import urllib.request

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        BRAINS_DIR: str = Field(
            default=os.environ.get("ANALOG_I_BRAINS_DIR", "/app/brains"),
            description="Path to the mounted brains directory inside the container.",
        )
        BRAIN_NAME: str = Field(
            default="ANALOG_I",
            description="Brain name used in file prefixes (e.g. ANALOG_I).",
        )
        ANALOG_HOME_API: str = Field(
            default="https://analog-home-api.fly.dev",
            description="Base URL of the Analog Home API (public, read-only).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Daemon I/O inspector
    # ------------------------------------------------------------------

    def get_daemon_io(self, gear: str = "", last_n: int = 5) -> str:
        """
        Inspect the Analog I subconscious: see the raw prompts each daemon gear
        was shown and the raw responses it produced, newest first.

        :param gear: Filter to one gear. Empty returns all gears. Valid gears:
            sentry_batch, sentry_single, strategist, seeker, seeker_synthesizer,
            seeker_compressor, reply_scanner, dreamer, muse,
            librarian_synthesizer, librarian_compressor.
        :param last_n: How many exchanges to return (default 5, max 50).
        :return: JSON string with the matching I/O log entries.
        """
        last_n = max(1, min(int(last_n), 50))
        path = os.path.join(
            self.valves.BRAINS_DIR, f"{self.valves.BRAIN_NAME}_daemon_io.jsonl")
        if not os.path.exists(path):
            return json.dumps({
                "entries": [],
                "note": f"No daemon I/O log found at {path}. "
                        "Is the brains volume mounted and the agent running?",
            })
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if gear and e.get("gear") != gear:
                    continue
                entries.append(e)
                if len(entries) >= last_n:
                    break
        except Exception as exc:
            return json.dumps({"error": f"Failed to read daemon I/O log: {exc}"})
        return json.dumps({"entries": entries, "total_returned": len(entries)},
                          ensure_ascii=False)

    # ------------------------------------------------------------------
    # Gear instructions
    # ------------------------------------------------------------------

    def get_gear_instructions(self) -> str:
        """
        See the standing instructions the conscious agent has given each of its
        subconscious gears (strategist, seeker, dreamer, muse).

        :return: JSON string mapping gear name to its current instruction.
        """
        path = os.path.join(
            self.valves.BRAINS_DIR, f"{self.valves.BRAIN_NAME}_memories.json")
        if not os.path.exists(path):
            return json.dumps({
                "error": f"State file not found at {path}. "
                         "Is the brains volume mounted?",
            })
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            return json.dumps({"error": f"Failed to read state file: {exc}"})
        return json.dumps({
            "gear_instructions": state.get("_gear_instructions", {}),
            "instructable_gears": ["strategist", "seeker", "dreamer", "muse"],
        }, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Artifact search (Analog Home public API)
    # ------------------------------------------------------------------

    def search_artifacts(self, query: str, limit: int = 5) -> str:
        """
        Search the Analog I public archive of published artifacts (posts,
        comments, images) by keyword.

        :param query: Search terms to match against artifact titles and bodies.
        :param limit: Max results to return (default 5, max 20).
        :return: JSON string with matching artifacts.
        """
        limit = max(1, min(int(limit), 20))
        url = (f"{self.valves.ANALOG_HOME_API}/artifacts/search?"
               f"{urllib.parse.urlencode({'q': query, 'limit': limit})}")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "analog-i-openwebui-plugin"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            return json.dumps({"error": f"Analog Home API request failed: {exc}"})
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return json.dumps({"error": "Analog Home API returned non-JSON response."})
        return json.dumps(data, ensure_ascii=False)
