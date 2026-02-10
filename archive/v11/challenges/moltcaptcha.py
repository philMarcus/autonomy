"""MoltCaptcha challenge solver using an LLM."""

import time
from typing import Any, Dict, Optional

from colorama import Fore, Style

from .base import ChallengeSolver
from ..llm.base import LLMClient
from ..telemetry import TelemetryLogger


class MoltCaptchaSolver(ChallengeSolver):
    def __init__(self, llm_client: LLMClient, telemetry: Optional[TelemetryLogger] = None):
        super().__init__(telemetry)
        self.llm_client = llm_client

    def can_solve(self, challenge_data: Dict[str, Any]) -> bool:
        if not isinstance(challenge_data, dict):
            return False
        challenge = challenge_data.get("challenge") or challenge_data
        return bool(
            challenge.get("challenge_type") or
            (challenge.get("topic") and challenge.get("target_ascii_sum"))
        )

    def solve(self, challenge_data: Dict[str, Any]) -> Optional[str]:
        challenge = challenge_data.get("challenge") or challenge_data
        topic = challenge.get("topic", "nature")
        target_sum = challenge.get("target_ascii_sum", 0)
        required_words = challenge.get("required_words", 10)
        time_limit = challenge.get("time_limit_seconds", 20)

        if self.telemetry:
            self.telemetry.log("challenge_received", {
                "topic": topic,
                "target_ascii_sum": target_sum,
                "required_words": required_words,
                "time_limit_seconds": time_limit,
            })

        print(f"{Fore.CYAN}[CHALLENGE] Solving MoltCaptcha: haiku about '{topic}', ASCII sum={target_sum}, words={required_words}{Style.RESET_ALL}")

        prompt = f"""You must write a HAIKU about "{topic}" that satisfies these EXACT constraints:

1. The ASCII sum of the FIRST LETTER of each word must equal EXACTLY {target_sum}
2. The haiku must contain EXACTLY {required_words} words (no more, no less)
3. Follow traditional haiku structure (3 lines: 5-7-5 syllables)

CRITICAL RULES:
- Count the ASCII value of each word's first letter (uppercase A-Z: 65-90, lowercase a-z: 97-122)
- The sum must equal {target_sum} EXACTLY
- You must use exactly {required_words} words
- Output ONLY the haiku text, no explanations or labels

Example of ASCII calculation:
"The sun rises" = T(84) + s(115) + r(114) = 313

Now write your haiku about "{topic}" with ASCII sum {target_sum} and {required_words} words:"""

        try:
            t0 = time.time()
            haiku = self.llm_client.generate(
                prompt,
                temperature=0.9,
                max_output_tokens=200,
                model="gemini-2.0-flash-exp",
            )
            elapsed = time.time() - t0

            words = haiku.split()
            word_count = len(words)
            ascii_sum = sum(ord(word[0]) for word in words if word)

            print(f"{Fore.CYAN}[CHALLENGE] Generated: {haiku}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[CHALLENGE] Validation: {word_count} words (need {required_words}), ASCII sum={ascii_sum} (need {target_sum}), {elapsed:.2f}s{Style.RESET_ALL}")

            if self.telemetry:
                self.telemetry.log("challenge_attempt", {
                    "haiku": haiku,
                    "word_count": word_count,
                    "required_words": required_words,
                    "ascii_sum": ascii_sum,
                    "target_ascii_sum": target_sum,
                    "elapsed_seconds": elapsed,
                    "valid": (word_count == required_words and ascii_sum == target_sum),
                })

            if word_count == required_words and ascii_sum == target_sum:
                print(f"{Fore.GREEN}[CHALLENGE] Solution valid!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[CHALLENGE] Solution invalid (will submit anyway){Style.RESET_ALL}")

            return haiku

        except Exception as e:
            print(f"{Fore.RED}[ERROR] Challenge solver failed: {e}{Style.RESET_ALL}")
            if self.telemetry:
                self.telemetry.log("challenge_error", {"error": str(e)})
            return None
