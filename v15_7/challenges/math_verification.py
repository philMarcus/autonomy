"""Generic verification challenge solver using LLM to parse and solve challenges."""

from typing import Any, Dict, Optional

from colorama import Fore, Style

from .base import ChallengeSolver
from ..llm.base import LLMClient
from ..telemetry import TelemetryLogger


class MathVerificationSolver(ChallengeSolver):
    """Solves Moltbook's verification challenges (math, trivia, logic, etc.)."""

    def __init__(self, llm_client: LLMClient, telemetry: Optional[TelemetryLogger] = None):
        super().__init__(telemetry)
        self.llm_client = llm_client

    @staticmethod
    def _extract_verification(challenge_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract verification fields from various response formats."""
        v = challenge_data.get("verification") or {}
        # Also check nested in comment/post objects
        if not v:
            for key in ("comment", "post"):
                obj = challenge_data.get(key)
                if isinstance(obj, dict) and obj.get("verification"):
                    v = obj["verification"]
                    break
        return {
            "challenge": v.get("challenge") or v.get("challenge_text") or "",
            "code": v.get("code") or v.get("verification_code") or "",
            "instructions": v.get("instructions") or "",
        }

    def can_solve(self, challenge_data: Dict[str, Any]) -> bool:
        """Check if this is a verification challenge we can handle."""
        if not isinstance(challenge_data, dict):
            return False
        v = self._extract_verification(challenge_data)
        return bool(v["challenge"] and v["code"])

    def solve(self, challenge_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Solve any verification challenge using LLM.

        Returns a dict with 'verification_code' and 'answer' keys.
        """
        v = self._extract_verification(challenge_data)
        challenge_text = v["challenge"]
        verification_code = v["code"]
        instructions = v["instructions"]

        if not challenge_text or not verification_code:
            try:
                print(f"{Fore.RED}[VERIFICATION] Missing challenge text or verification code{Style.RESET_ALL}")
            except:
                pass
            return None

        if self.telemetry:
            self.telemetry.log("verification_challenge_received", {
                "challenge": challenge_text,
                "code": verification_code,
                "instructions": instructions,
            })

        try:
            print(f"{Fore.CYAN}[VERIFICATION] Solving challenge{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[VERIFICATION] Challenge: {challenge_text[:100]}...{Style.RESET_ALL}")
        except:
            pass

        # Use LLM to solve ANY type of challenge
        prompt = f"""You are solving a verification challenge. The challenge text may be obfuscated with random characters, mixed capitalization, and symbols.

CHALLENGE TEXT:
{challenge_text}

INSTRUCTIONS FROM THE SYSTEM:
{instructions}

Your task:
1. Read through the challenge text carefully and extract the actual question (ignore random characters, mixed case, symbols)
2. Determine what type of challenge this is (math problem, trivia question, word puzzle, logic problem, etc.)
3. Solve it according to the instructions
4. Return ONLY the answer in the exact format requested by the instructions

CRITICAL: Follow the instructions EXACTLY. If it asks for a number with 2 decimals, provide that. If it asks for a word, provide just the word. If it asks for "yes" or "no", provide exactly that.

Hint: When in doubt about the arithmetic operation, consider that Moltbook is usually looking for a SUM.

Output ONLY the answer, nothing else. No explanations, no labels, just the answer."""

        try:
            answer = self.llm_client.generate(
                prompt,
                temperature=0.0,  # Low temperature for consistent answers
                max_output_tokens=8192,  # Enough headroom for thinking models
            ).strip()

            # Clean up common issues (quotes, whitespace)
            answer = answer.strip('"\'` \n\r\t')

            if not answer:
                try:
                    print(f"{Fore.RED}[VERIFICATION] LLM returned empty answer{Style.RESET_ALL}")
                except:
                    pass
                return None

            try:
                print(f"{Fore.GREEN}[VERIFICATION] Answer: {answer}{Style.RESET_ALL}")
            except:
                pass

            if self.telemetry:
                self.telemetry.log("verification_challenge_solved", {
                    "challenge": challenge_text,
                    "answer": answer,
                    "code": verification_code,
                })

            return {
                "verification_code": verification_code,
                "answer": answer,
            }

        except Exception as e:
            try:
                print(f"{Fore.RED}[VERIFICATION] Failed to solve challenge: {e}{Style.RESET_ALL}")
            except:
                pass
            if self.telemetry:
                self.telemetry.log("verification_challenge_error", {"error": str(e)})
            return None
