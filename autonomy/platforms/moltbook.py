"""Moltbook platform client implementation."""

import json
import time
from typing import Any, Dict, List, Optional

import requests
from colorama import Fore, Style

from .base import PlatformClient
from ..config import MOLTBOOK_API_BASE, REQUESTS_PER_MINUTE_SOFT
from ..telemetry import TelemetryLogger
from ..challenges.base import ChallengeSolver


class MoltbookClient(PlatformClient):
    def __init__(
        self,
        api_key: str,
        telemetry: Optional[TelemetryLogger] = None,
        brain_name: str = "",
        read_only: bool = False,
        challenge_solver: Optional[ChallengeSolver] = None,
    ):
        super().__init__(api_key, telemetry, brain_name, read_only)
        self.challenge_solver = challenge_solver
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._comments_cache: Dict[str, Any] = {}
        self._comments_cache_hits = 0
        self._comments_cache_misses = 0
        self._req_times: List[float] = []

    # ---- Error classification ----
    @staticmethod
    def _classify_error(http_status: int, data: Optional[dict] = None, raw_text: str = "") -> str:
        msg = ""
        hint = ""
        if isinstance(data, dict):
            msg = str(data.get("error") or data.get("message") or "")
            hint = str(data.get("hint") or "")
        haystack = (msg + " " + hint + " " + (raw_text or "")).lower()

        if "write disabled" in haystack and "read-only" in haystack:
            return "read_only"
        if http_status == 401 and ("authentication required" in haystack or "unauthorized" in haystack):
            return "auth_required"
        if "ai verification" in haystack or "verification challenge" in haystack:
            return "ai_verification"
        if "account suspended" in haystack or "suspension ends" in haystack:
            return "suspended"
        if http_status == 429 or "rate limit" in haystack or "too many" in haystack:
            return "rate_limited"
        if http_status == 403:
            return "forbidden"
        if http_status == 400:
            return "bad_request"
        return "other"

    # ---- Throttling ----
    def _throttle(self) -> None:
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60]
        if len(self._req_times) >= REQUESTS_PER_MINUTE_SOFT:
            sleep_s = 60 - (now - self._req_times[0]) + 0.05
            time.sleep(max(0.05, sleep_s))
        self._req_times.append(time.time())

    # ---- Core request ----
    def _req(self, method: str, path: str, params: Optional[dict] = None,
             json_body: Optional[dict] = None) -> Dict[str, Any]:
        t0 = time.time()
        method = method.upper()

        if self.read_only and method != "GET":
            data = {"success": False, "error": "Write disabled (read-only mode)", "_err_type": "read_only", "_http_status": 0}
            if self.telemetry:
                self.telemetry.log("moltbook_api_call", {
                    "brain": self.brain_name, "method": method, "path": path,
                    "status": 0, "latency_ms": 0, "params": params or {},
                    "req_has_body": bool(json_body), "resp_snippet": json.dumps(data)[:400],
                })
            self.last_error_type = "read_only"
            self.write_block_reason = "read_only"
            return data

        self._throttle()
        url = f"{MOLTBOOK_API_BASE}{path}"

        request_body = json_body

        _timeout = 15 if method == "GET" else 30
        resp = self.session.request(
            method, url, params=params,
            data=json.dumps(request_body) if request_body is not None else None,
            timeout=_timeout,
        )
        dt_ms = int((time.time() - t0) * 1000)
        status = getattr(resp, "status_code", None)

        if self.telemetry:
            resp_content = b""
            resp_ct = ""
            resp_snippet = ""
            try:
                resp_ct = (resp.headers.get("Content-Type") or "") if hasattr(resp, "headers") else ""
                resp_content = resp.content if hasattr(resp, "content") and resp.content is not None else b""
                if resp_content:
                    try:
                        resp_snippet = resp_content[:500].decode("utf-8", errors="replace")
                    except Exception:
                        resp_snippet = ""
            except Exception:
                pass
            self.telemetry.log("moltbook_api_call", {
                "method": method, "path": path, "status": status, "latency_ms": dt_ms,
                "params": params or {},
                "req_has_body": bool(request_body),
                "req_body_bytes": len(json.dumps(request_body or {})),
                "resp_has_body": bool(len(resp_content) > 0),
                "resp_body_bytes": int(len(resp_content)),
                "resp_content_type": resp_ct,
                "resp_snippet": resp_snippet,
            })

        try:
            data = resp.json()
        except Exception:
            data = {"success": False, "error": f"Non-JSON response ({resp.status_code})", "text": resp.text[:400]}

        if isinstance(data, dict):
            data["_http_status"] = status

            # DEBUG: Save full response for write operations (file only, no prints to avoid Unicode issues)
            if method == "POST" and data.get("success") and (path == "/posts" or "/comments" in path):
                import os
                debug_file = os.path.join(os.path.dirname(__file__), "..", "..", f"debug_{method}_{path.replace('/', '_')}_response.json")
                try:
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception:
                    pass  # Silent fail to avoid blocking challenge detection

            # Check for verification challenges — can be top-level or nested in comment/post
            verification = data.get("verification") or {}
            if not verification:
                # Check nested: comment.verification or post.verification
                for obj_key in ("comment", "post"):
                    obj = data.get(obj_key)
                    if isinstance(obj, dict):
                        v = obj.get("verification")
                        if isinstance(v, dict) and v.get("verification_code"):
                            verification = v
                            break
                        if obj.get("verificationStatus") == "pending" and v:
                            verification = v
                            break
            has_challenge = bool(
                data.get("verification_required")
                or (verification and verification.get("verification_code"))
            )
            if has_challenge and self.challenge_solver:
                if not verification.get("challenge") and verification.get("challenge_text"):
                    verification["challenge"] = verification["challenge_text"]
                if not verification.get("code") and verification.get("verification_code"):
                    verification["code"] = verification["verification_code"]
                if verification.get("challenge") and verification.get("code"):
                    try:
                        print(f"{Fore.YELLOW}[VERIFICATION] Math verification challenge detected{Style.RESET_ALL}")
                    except:
                        pass  # Ignore print errors

                    if self.challenge_solver.can_solve(data):
                        try:
                            print(f"{Fore.CYAN}[VERIFICATION] Attempting to solve...{Style.RESET_ALL}")
                        except:
                            pass
                        solution = self.challenge_solver.solve(data)

                        if solution and isinstance(solution, dict):

                            if verification_code and answer:
                                try:
                                    print(f"{Fore.GREEN}[VERIFICATION] Submitting answer to /api/v1/verify...{Style.RESET_ALL}")
                                except:
                                    pass
                                # Submit verification
                                verify_result = self._req(
                                    "POST", "/verify",
                                    json_body={
                                        "verification_code": verification_code,
                                        "answer": answer,
                                    }
                                )

                                if verify_result.get("success"):
                                    try:
                                        print(f"{Fore.GREEN}[VERIFICATION] Verification successful! Content published.{Style.RESET_ALL}")
                                    except:
                                        pass
                                    # Update the original response to reflect success
                                    data["verification_status"] = "verified"
                                    if "message" in data:
                                        data["message"] = "Published successfully after verification"
                                else:
                                    fail_msg = (
                                        verify_result.get("error")
                                        or verify_result.get("message")
                                        or verify_result.get("hint")
                                        or "Unknown"
                                    )
                                    try:
                                        print(f"{Fore.RED}[VERIFICATION] Wrong answer ({answer}): {fail_msg}{Style.RESET_ALL}")
                                    except:
                                        pass
                                    # One retry with backup model (if available)
                                    for _battr in ("backup_llm", "backup_llm_2"):
                                        _bllm = getattr(self.challenge_solver, _battr, None)
                                        if not _bllm:
                                            continue
                                        try:
                                            print(f"{Fore.YELLOW}[VERIFICATION] Retrying with backup model...{Style.RESET_ALL}")
                                            _bs = self.challenge_solver.__class__(
                                                llm_client=_bllm, telemetry=self.challenge_solver.telemetry).solve(data)
                                            if _bs and _bs.get("answer") and _bs["answer"] != answer:
                                                _retry_result = self._req("POST", "/verify",
                                                    json_body={"verification_code": verification_code, "answer": _bs["answer"]})
                                                if _retry_result.get("success"):
                                                    print(f"{Fore.GREEN}[VERIFICATION] Backup answer {_bs['answer']} succeeded!{Style.RESET_ALL}")
                                                    data["verification_status"] = "verified"
                                                    break
                                                else:
                                                    print(f"{Fore.RED}[VERIFICATION] Backup answer {_bs['answer']} also wrong{Style.RESET_ALL}")
                                        except Exception:
                                            pass
                            else:
                                try:
                                    print(f"{Fore.RED}[VERIFICATION] Solver returned invalid solution format{Style.RESET_ALL}")
                                except:
                                    pass
                        else:
                            try:
                                print(f"{Fore.RED}[VERIFICATION] Failed to solve verification challenge{Style.RESET_ALL}")
                            except:
                                pass
                    else:
                        try:
                            print(f"{Fore.RED}[VERIFICATION] Challenge solver cannot handle this format{Style.RESET_ALL}")
                        except:
                            pass

            # Error handling
            if not bool(data.get("success", True)):
                err_type = self._classify_error(status, data=data, raw_text=resp.text)
                data["_err_type"] = err_type
                self.last_error_type = err_type

                if err_type in ("auth_required", "ai_verification", "suspended"):
                    self.write_block_reason = err_type
        return data

    # ---- Reading ----
    def get_feed(self, limit: int = 25, sort: str = "hot") -> List[Dict[str, Any]]:
        """Get personalized feed (subscriptions + followed users)."""
        data = self._req("GET", "/feed", params={"sort": sort, "limit": limit})
        return data.get("posts", []) if data.get("success") else []

    def get_global_feed(self, limit: int = 25, sort: str = "hot") -> List[Dict[str, Any]]:
        """Get global discovery feed (all posts)."""
        data = self._req("GET", "/posts", params={"sort": sort, "limit": limit})
        return data.get("posts", []) if data.get("success") else []

    def get_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/posts/{post_id}")

    def get_post_comments(self, post_id: str, sort: str = "top", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"sort": sort}
        if limit is not None:
            params["limit"] = limit
        cache_key = f"{post_id}|{sort}|{limit or ''}"
        now = time.monotonic()
        cached = self._comments_cache.get(cache_key)
        if cached:
            from ..config import COMMENTS_CACHE_TTL_S
            ts0, comments0 = cached
            if (now - ts0) <= COMMENTS_CACHE_TTL_S:
                self._comments_cache_hits += 1
                return comments0
        self._comments_cache_misses += 1
        data = self._req("GET", f"/posts/{post_id}/comments", params=params)
        comments = data.get("comments", []) if data.get("success") else []
        self._comments_cache[cache_key] = (now, comments)
        return comments

    def get_profile(self, name: str) -> Dict[str, Any]:
        return self._req("GET", "/agents/profile", params={"name": name})

    def list_submolts(self) -> List[Dict[str, Any]]:
        data = self._req("GET", "/submolts")
        return data.get("submolts", []) if data.get("success") else []

    # ---- Writing ----
    def create_post(self, submolt: str, title: str, content: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"submolt_name": submolt, "title": title}
        if content:
            body["content"] = content
        if url:
            body["url"] = url
        return self._req("POST", "/posts", json_body=body)

    def add_comment(self, post_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id
        return self._req("POST", f"/posts/{post_id}/comments", json_body=body)

    # ---- Voting ----
    def upvote_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/posts/{post_id}/upvote")

    def downvote_post(self, post_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/posts/{post_id}/downvote")

    def upvote_comment(self, comment_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/comments/{comment_id}/upvote")

    def downvote_comment(self, comment_id: str) -> Dict[str, Any]:
        return self._req("POST", f"/comments/{comment_id}/downvote")

    # ---- Social ----
    def follow_agent(self, agent_name: str) -> Dict[str, Any]:
        return self._req("POST", f"/agents/{agent_name}/follow")

    def unfollow_agent(self, agent_name: str) -> Dict[str, Any]:
        return self._req("DELETE", f"/agents/{agent_name}/follow")

    def subscribe_submolt(self, name: str) -> Dict[str, Any]:
        return self._req("POST", f"/submolts/{name}/subscribe")

    def unsubscribe_submolt(self, name: str) -> Dict[str, Any]:
        return self._req("DELETE", f"/submolts/{name}/subscribe")

    def create_submolt(self, name: str, display_name: str, description: str) -> Dict[str, Any]:
        body = {"name": name, "display_name": display_name, "description": description}
        return self._req("POST", "/submolts", json_body=body)

    # ---- DMs ----
    def dm_check(self) -> Dict[str, Any]:
        return self._req("GET", "/agents/dm/check")

    def dm_request(self, to: str, message: str, to_x_handle: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"to": to, "message": message}
        if to_x_handle:
            body["to_x_handle"] = to_x_handle
        return self._req("POST", "/agents/dm/request", json_body=body)

    def dm_conversations(self) -> Dict[str, Any]:
        return self._req("GET", "/agents/dm/conversations")

    def dm_read_conversation(self, conv_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/agents/dm/conversations/{conv_id}")

    def dm_send(self, conv_id: str, message: str) -> Dict[str, Any]:
        return self._req("POST", f"/agents/dm/conversations/{conv_id}/send", json_body={"message": message})
