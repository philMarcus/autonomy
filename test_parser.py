"""Regression tests for autonomy.daemon._parse_json_safe.

Loads test cases from test_parser_fixtures.json and runs them through
the parser. Each case has an expected return type and (for non-None
results) an expected first action.

Run: python test_parser.py
Exit: 0 on all pass, 1 if any fail.
"""

import json
import sys
from pathlib import Path

from autonomy.daemon import _parse_json_safe


def run() -> int:
    fixtures_path = Path(__file__).parent / "test_parser_fixtures.json"
    with open(fixtures_path) as f:
        data = json.load(f)

    cases = data["cases"]
    failures = []

    for case in cases:
        name = case["name"]
        text = case["input"]
        expected_type = case["expected_type"]  # "list", "dict", or "none"
        expected_action = case.get("expected_first_action")
        expected_count = case.get("expected_count")

        result = _parse_json_safe(text)

        # Check type
        if expected_type == "none":
            if result is not None:
                failures.append(f"  [{name}] expected None, got {type(result).__name__}: {result}")
                continue
        elif expected_type == "list":
            if not isinstance(result, list):
                failures.append(f"  [{name}] expected list, got {type(result).__name__}: {result}")
                continue
        elif expected_type == "dict":
            if not isinstance(result, dict):
                failures.append(f"  [{name}] expected dict, got {type(result).__name__}: {result}")
                continue

        # Check action
        if expected_action is not None:
            first = result[0] if isinstance(result, list) else result
            if not isinstance(first, dict) or first.get("action") != expected_action:
                actual = first.get("action") if isinstance(first, dict) else "(not dict)"
                failures.append(f"  [{name}] expected action={expected_action}, got {actual}")
                continue

        # Check count (if specified)
        if expected_count is not None and isinstance(result, list):
            if len(result) != expected_count:
                failures.append(f"  [{name}] expected count={expected_count}, got {len(result)}")
                continue

    total = len(cases)
    passed = total - len(failures)

    if failures:
        print(f"FAIL: {passed}/{total} passed, {len(failures)} failed")
        print()
        for f in failures:
            print(f)
        return 1

    print(f"OK: all {total} cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(run())
