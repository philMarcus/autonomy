"""Generic verification challenge solver using LLM to parse and solve challenges."""

from typing import Any, Dict, Optional

from colorama import Fore, Style

from .base import ChallengeSolver
from ..llm.base import LLMClient
from ..telemetry import TelemetryLogger
from ..live_term import emit_status


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
            emit_status("[VERIFICATION]", "Missing challenge text or verification code", color=Fore.RED)
            return None

        if self.telemetry:
            self.telemetry.log("verification_challenge_received", {
                "challenge": challenge_text,
                "code": verification_code,
                "instructions": instructions,
            })

        _model = getattr(self.llm_client, '_default_model_id', '?')
        emit_status("[VERIFICATION]", f"({_model}) Challenge: {challenge_text[:80]}...",
                    color=Fore.CYAN,
                    multiline=f"  {challenge_text}")

        # Simple prompt that works well with small/local models.
        # Key insight: asking the model to write plain English first, then solve, works
        # much better than structured STEP 1/2/3/4 prompts for deobfuscation tasks.
        prompt = f"""This text has random symbols and weird spacing added to it. Read through the noise to find the real words.

Text: {challenge_text}

{instructions}

What does the text say in plain English? Then solve the math and give ONLY the number with 2 decimal places on the last line."""

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
                emit_status("[VERIFICATION]", "LLM returned empty answer", color=Fore.RED)
                return None

            emit_status("[VERIFICATION]", f"Answer: {answer}", color=Fore.GREEN)

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
            emit_status("[VERIFICATION]", f"Failed to solve challenge. {str(e)[:100]}", color=Fore.RED)
            if self.telemetry:
                self.telemetry.log("verification_challenge_error", {"error": err_str[:300]})
            # Retry with backup LLM on 503
            if is_503 and hasattr(self, 'backup_llm') and self.backup_llm:
                try:
                    _backup_model = getattr(self.backup_llm, '_default_model_id', '?')
                    emit_status("[VERIFICATION]", f"503 — retrying with {_backup_model}", color=Fore.YELLOW)
                    raw_response = self.backup_llm.generate(prompt, temperature=0.0, max_output_tokens=8192).strip()
                    lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                    answer = lines[-1] if lines else ""
                    answer = answer.strip('"\'` \n\r\t')
                    for prefix in ("ANSWER:", "Answer:", "answer:", "STEP 4", "Step 4"):
                        if answer.startswith(prefix):
                            answer = answer[len(prefix):].strip().strip(":").strip()
                    if answer:
                        emit_status("[VERIFICATION]", f"Backup answer: {answer}", color=Fore.GREEN)
                        return {"verification_code": verification_code, "answer": answer}
                except Exception:
                    pass
            # 2nd backup: ollama:gemma3:12b (free, local)
            if is_503 and hasattr(self, 'backup_llm_2') and self.backup_llm_2:
                try:
                    _backup2_model = getattr(self.backup_llm_2, '_default_model_id', '?')
                    emit_status("[VERIFICATION]", f"Retrying with {_backup2_model}", color=Fore.YELLOW)
                    raw_response = self.backup_llm_2.generate(prompt, temperature=0.0, max_output_tokens=1024).strip()
                    lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                    answer = lines[-1] if lines else ""
                    answer = answer.strip('"\'` \n\r\t')
                    for prefix in ("ANSWER:", "Answer:", "answer:", "STEP 4", "Step 4"):
                        if answer.startswith(prefix):
                            answer = answer[len(prefix):].strip().strip(":").strip()
                    if answer:
                        emit_status("[VERIFICATION]", f"Ollama answer: {answer}", color=Fore.GREEN)
                        return {"verification_code": verification_code, "answer": answer}
                except Exception:
                    pass
            return None
