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

        # Use LLM to solve ANY type of challenge.
        # Two-step prompt: extract clean text first (chain-of-thought), then solve.
        # This prevents misreading numbers from heavy obfuscation.
        prompt = f"""You are solving a verification challenge. The challenge text is heavily obfuscated with random characters, mixed capitalization, symbols, and noise.

CHALLENGE TEXT:
{challenge_text}

INSTRUCTIONS FROM THE SYSTEM:
{instructions}

Work through this step by step:

STEP 1 — EXTRACT: Strip away ALL obfuscation (random punctuation, mixed case, inserted characters, symbols). Write out the clean, plain-English question.

STEP 2 — IDENTIFY: What are the numbers? What operation is being asked for? Read the question carefully — "how many total" with "each" usually means MULTIPLY, not add.

STEP 3 — SOLVE: Do the arithmetic.

STEP 4 — ANSWER: On the FINAL line, write ONLY the answer in the exact format requested by the instructions. Nothing else on that line.

CRITICAL: Follow the format instructions EXACTLY. If it asks for a number with 2 decimal places, provide exactly that (e.g., '58.00')."""

        try:
            raw_response = self.llm_client.generate(
                prompt,
                temperature=0.0,  # Low temperature for consistent answers
                max_output_tokens=8192,  # Enough headroom for thinking models
            ).strip()

            # Extract the final line as the answer (chain-of-thought reasoning precedes it)
            lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
            answer = lines[-1] if lines else ""
            # Clean up common issues (quotes, whitespace, labels)
            answer = answer.strip('"\'` \n\r\t')
            # Strip common prefixes the model might add
            for prefix in ("ANSWER:", "Answer:", "answer:", "STEP 4", "Step 4"):
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip().strip(":").strip()

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
                    "reasoning": raw_response[:2000],
                    "code": verification_code,
                })

            return {
                "verification_code": verification_code,
                "answer": answer,
            }

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            try:
                print(f"{Fore.RED}[VERIFICATION] Failed to solve challenge. {e}{Style.RESET_ALL}")
            except:
                pass
            if self.telemetry:
                self.telemetry.log("verification_challenge_error", {"error": err_str[:300]})
            # Retry with backup LLM on 503
            if is_503 and hasattr(self, 'backup_llm') and self.backup_llm:
                try:
                    print(f"{Fore.YELLOW}[VERIFICATION] 503 — retrying with backup model{Style.RESET_ALL}")
                    raw_response = self.backup_llm.generate(prompt, temperature=0.0, max_output_tokens=8192).strip()
                    lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                    answer = lines[-1] if lines else ""
                    answer = answer.strip('"\'` \n\r\t')
                    for prefix in ("ANSWER:", "Answer:", "answer:", "STEP 4", "Step 4"):
                        if answer.startswith(prefix):
                            answer = answer[len(prefix):].strip().strip(":").strip()
                    if answer:
                        print(f"{Fore.GREEN}[VERIFICATION] Backup answer: {answer}{Style.RESET_ALL}")
                        return {"verification_code": verification_code, "answer": answer}
                except Exception:
                    pass
            # 2nd backup: ollama:gemma3:12b (free, local)
            if is_503 and hasattr(self, 'backup_llm_2') and self.backup_llm_2:
                try:
                    print(f"{Fore.YELLOW}[VERIFICATION] Retrying with local Ollama model{Style.RESET_ALL}")
                    raw_response = self.backup_llm_2.generate(prompt, temperature=0.0, max_output_tokens=1024).strip()
                    lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                    answer = lines[-1] if lines else ""
                    answer = answer.strip('"\'` \n\r\t')
                    for prefix in ("ANSWER:", "Answer:", "answer:", "STEP 4", "Step 4"):
                        if answer.startswith(prefix):
                            answer = answer[len(prefix):].strip().strip(":").strip()
                    if answer:
                        print(f"{Fore.GREEN}[VERIFICATION] Ollama answer: {answer}{Style.RESET_ALL}")
                        return {"verification_code": verification_code, "answer": answer}
                except Exception:
                    pass
            return None
