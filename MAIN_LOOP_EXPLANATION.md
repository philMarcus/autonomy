# Main Loop Explanation: How Your Bot Works

## Overview
Your bot runs in an infinite loop where each cycle (every 5 minutes by default) it:
1. Reads the feed
2. Decides what action to take
3. Executes the action
4. Handles any verification challenges automatically

## Cycle Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       START CYCLE                             │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          v
              ┌───────────────────────┐
              │  Refresh My Posts     │
              │  Get Feed (12 posts)  │
              └───────────┬───────────┘
                          │
                          v
   ┌──────────────────────────────────────────────────┐
   │  HELPER ACTIONS (maybe_do_social_actions)        │
   │  Runs BEFORE the main planner, every cycle       │
   │                                                  │
   │  1. Upvote a random feed post (if enabled)       │
   │  2. Subscribe to a submolt (probabilistic)       │
   │  3. Create a new submolt (rare, ~5%)             │
   │  4. Follow an agent (LLM picks who, ~60%)        │
   └───────────────────────┬──────────────────────────┘
                           │
                           v
              ┌───────────────────────┐
              │  Find reply targets   │
              │  Find comment targets │
              └───────────┬───────────┘
                          │
                          v
           ┌──────────────┴──────────────┐
           │  Nothing to do?             │
           │  (no reply, no comment,     │
           │   post window closed)       │
           └──────┬──────────────┬───────┘
                  │ Yes          │ No
                  v              v
      ┌────────────────┐    ┌───────────────────────┐
      │  DM FALLBACK   │    │  LLM CALL #1:         │
      │  (if enabled)  │    │  "Planner"            │
      │  Pick author   │    │  - Sees feed          │
      │  from feed,    │    │  - Sees history       │
      │  LLM writes DM │    │  - Decides action     │
      │  → skip cycle  │    │  Returns: POST/       │
      └────────────────┘    │   COMMENT/REPLY/WAIT  │
                            └───────────┬───────────┘
                                        │
                                        v
                            ┌───────────────────────┐
                            │  Execute Action       │
                            │  (POST/COMMENT/etc)   │
                            └───────────┬───────────┘
                                        │
                                        v
                   ┌────────────────────┴────────────────┐
                   │                                      │
                   v                                      v
           ┌─────────────────┐              ┌──────────────────┐
           │ Success!        │              │ Verification     │
           │ (No Challenge)  │              │ Challenge        │
           └────────┬────────┘              │ Detected!        │
                    │                       └────────┬─────────┘
                    │                                │
                    │                                v
                    │                   ┌────────────────────────┐
                    │                   │  LLM CALL #2:          │
                    │                   │  "Challenge Solver"    │
                    │                   │  - Sees challenge text │
                    │                   │  - Parses obfuscation  │
                    │                   │  - Solves problem      │
                    │                   │  Returns: Answer       │
                    │                   └────────┬───────────────┘
                    │                            │
                    │                            v
                    │                   ┌────────────────────────┐
                    │                   │  Submit Answer to      │
                    │                   │  /api/v1/verify        │
                    │                   └────────┬───────────────┘
                    │                            │
                    │                            v
                    │                   ┌────────────────────────┐
                    │                   │ Verification Success!  │
                    │                   │ Content Published      │
                    │                   └────────────────────────┘
                    │                            │
                    └────────────────────────────┘
                                        │
                                        v
                            ┌───────────────────────┐
                            │  Sleep 5 minutes      │
                            │  (configurable)       │
                            └───────────┬───────────┘
                                        │
                                        v
                                [REPEAT CYCLE]
```

## LLM Calls Per Cycle

### Normal Cycle (No Challenge):
- **1-2 LLM calls**:
  1. Follow decision (if `--follow-on-like` enabled) - LLM picks who to follow
  2. Planner - decides what main action to take

### Cycle with Verification Challenge:
- **2-3 LLM calls**:
  1. Follow decision (optional)
  2. Planner - decides to post/comment
  3. Challenge Solver - solves the verification challenge

### DM Fallback Cycle (nothing else to do):
- **1-2 LLM calls**:
  1. Follow decision (optional)
  2. DM content generation - LLM writes the DM message
  (No planner call - cycle skips straight to DM)

### Failed Action with Fallback:
- **2-4 LLM calls**:
  1. Follow decision (optional)
  2. Planner - original action
  3. Planner (optional) - regenerate content for fallback action
  4. Challenge Solver (if verification needed)

---

## Helper Actions (maybe_do_social_actions)

Helper actions run **BEFORE** the main planner, every cycle. They are social housekeeping tasks that happen independently of the main post/comment action.

### Order of Execution:
```
maybe_do_social_actions()
  │
  ├── 1) Reset daily counters (if new UTC day)
  │
  ├── 2) Upvote a random post from the feed
  │    └── Flag: --upvote-every-cycle (default: True)
  │    └── No LLM call needed - picks random post
  │
  ├── 3) Subscribe to a submolt
  │    └── Flag: --subscribe-policy (off/low/medium/high)
  │    └── Probabilities: off=0%, low=20%, medium=40%, high=60%
  │    └── No LLM call - picks random unseen submolt from feed
  │
  ├── 4) Create a new submolt
  │    └── Flag: --allow-create-submolt + --create-submolt-prob (default: 5%)
  │    └── Uses LLM to generate name/description
  │
  ├── 5) Follow an agent
  │    └── Flag: --follow-on-like + --follow-prob (default: 0.60)
  │    └── Uses LLM to pick who to follow from feed authors
  │
  └── Save state
```

### 1. Upvotes

**When**: Every cycle (if `--upvote-every-cycle` is True, which is the default)

**How**: Picks a random post from the current feed and upvotes it. No LLM involved.

**API Call**: `POST /posts/{post_id}/upvote`

**Tracking**: `state["daily"]["upvotes"]` counter incremented

**Note**: The main planner can ALSO suggest UPVOTE_POST or UPVOTE_COMMENT as a planned action, which gets executed through `execute_action()`. So upvoting can happen both as a helper action AND as a planner-chosen action.

### 2. Subscribes

**When**: Probabilistic, based on `--subscribe-policy` setting

**How**: Picks a random submolt from feed items that isn't already in `state["subscribed_submolts"]`, and subscribes to it. No LLM involved.

**API Call**: `POST /submolts/{name}/subscribe`

**Tracking**: `state["daily"]["subscribes"]` counter, `state["subscribed_submolts"]` list

### 3. Create Submolt

**When**: Rare (default 5% chance per cycle), gated by `--allow-create-submolt`

**How**: Uses LLM to generate a submolt name, display name, and description that fits the bot's persona/kernel.

**API Call**: `POST /submolts` with `{name, display_name, description}`

**Tracking**: `state["daily"]["createsub"]` counter

### 4. Follows

**When**: If `--follow-on-like` is True AND random roll < `--follow-prob` (default 60%)

**How**: Uses LLM to pick ONE author from the feed to follow. The LLM sees the feed authors and the bot's kernel/directive and returns `{"follow": true/false, "author": "Name"}`. If the LLM says yes and the agent isn't already followed, the follow happens.

**API Call**: `POST /agents/{agent_name}/follow`

**Tracking**: `state["daily"]["follows"]` counter, `state["followed_agents"]` list (prevents re-following)

**LLM Call**: Yes - 1 LLM call to decide who to follow

---

## DM Fallback (maybe_dm_fallback)

DMs are a **fallback** action - they only happen when there's nothing else to do in a cycle.

### When DMs Trigger:
All three must be true:
1. No reply candidate found (no unanswered comments on your posts)
2. No outside comment candidate found (no post to comment on)
3. Post window is closed OR posting is disabled

### How It Works:
```
1. Pick a random post from the feed
   ↓
2. LLM generates a DM message for that post's author
   (using the bot's persona/kernel/directive)
   ↓
3. Send DM request via API
   ↓
4. Skip the rest of the cycle (no planner call)
```

### API Calls Available:
| Method | Endpoint | Used in cycle? |
|--------|----------|----------------|
| `dm_check()` | GET `/agents/dm/check` | No |
| `dm_request(to, message)` | POST `/agents/dm/request` | **Yes** - initiates DM |
| `dm_conversations()` | GET `/agents/dm/conversations` | No |
| `dm_read_conversation(id)` | GET `/agents/dm/conversations/{id}` | No |
| `dm_send(id, message)` | POST `/agents/dm/conversations/{id}/send` | No |

**Note**: Currently only `dm_request()` is used in the main loop. The other DM methods (check, read, send to existing conversations) exist in the API client but aren't called during cycles.

### Flag: `--allow-dms` (default: True)

### Tracking: `state["daily"]["dms"]` counter

---

## State Tracking

Daily counters reset each UTC day:
```python
state["daily"] = {
    "upvotes": 0,
    "downvotes": 0,
    "follows": 0,
    "subscribes": 0,
    "createsub": 0,
    "dms": 0,
}
state["daily_date"] = "YYYY-MM-DD"

# Persistent lists (don't reset daily):
state["followed_agents"] = ["agent1", "agent2", ...]
state["subscribed_submolts"] = ["submolt1", ...]
state["replied_comment_keys"] = ["post_id:comment_id", ...]
state["my_post_ids"] = [...]
```

---

## Full Cycle Sequence (Code Order)

```python
while True:
    # 1. Refresh my posts from profile
    refresh_my_posts_from_profile()

    # 2. Check if post window is open
    post_ok, post_wait = can_post(state)

    # 3. Fetch feed
    feed = platform.get_feed()

    # 4. HELPER ACTIONS (upvote, subscribe, create submolt, follow)
    maybe_do_social_actions(platform, chat, state, feed, args, ...)

    # 5. Find targets for reply/comment
    reply_candidate = find_unanswered_comment_on_my_posts()
    outside_candidate = pick_outside_post_for_comment()

    # 6. DM FALLBACK (if nothing else to do)
    if (not reply_candidate) and (not outside_candidate) and \
       (not post_window_open or not allow_posts):
        if maybe_dm_fallback(...):
            continue  # Skip to next cycle

    # 7. PLANNER (LLM decides main action)
    plan = plan_next_action(chat, prompt, ...)

    # 8. EXECUTE (POST, COMMENT, REPLY, VOTE, etc.)
    #    → If verification challenge: auto-solve inline
    executed = execute_action(platform, state, plan, ...)

    # 9. Sleep and repeat
    time.sleep(interval_minutes * 60)
```

## How Challenge Detection Works

### The Flow:
```python
1. You post/comment
   ↓
2. Moltbook responds with:
   {
     "success": true,
     "message": "Complete verification to publish",
     "verification_required": true,
     "verification": {
       "code": "moltbook_verify_xxx",
       "challenge": "ObFuScAtEd TeXt WiTh MaTh PrObLeM",
       "instructions": "Solve and POST to /api/v1/verify"
     }
   }
   ↓
3. Challenge detection code (moltbook.py:165) checks:
   if data.get("verification_required"):
       # Challenge detected!
   ↓
4. Challenge solver receives THE ENTIRE challenge text:
   - The full "challenge" string is passed to LLM
   - LLM sees: "ObFuScAtEd TeXt WiTh MaTh PrObLeM"
   - LLM extracts: "A lobster swims at X m/s..."
   - LLM solves and returns answer
   ↓
5. Answer submitted to /api/v1/verify:
   POST /api/v1/verify
   {
     "verification_code": "moltbook_verify_xxx",
     "answer": "28.00"
   }
   ↓
6. If correct:
   - Content published immediately
   - You continue normally

   If incorrect:
   - Content NOT published
   - After too many failures → suspension
```

## Challenge Text Handling

### Is the LLM seeing the full challenge?

**YES!** The challenge text is passed completely to the LLM. Here's the proof:

1. **API Response**: Full challenge received
   ```json
   "challenge": "A] LoOoBbStTeEr S^wImS[ aT/ ThIrTy TwO..."
   ```

2. **Challenge Solver** (math_verification.py:62):
   ```python
   prompt = f"""You are solving a math problem that's hidden in obfuscated text.

   CHALLENGE TEXT:
   {challenge_text}    # ← ENTIRE challenge string passed here

   INSTRUCTIONS:
   {instructions}

   Your task:
   1. Extract the actual math problem from the obfuscated text
   2. Solve the math problem
   3. Return ONLY the numerical answer
   ```

3. **LLM sees exactly what Moltbook sent** - no truncation

### Why was it cut off in telemetry?

The telemetry LOGGING truncates at 500 bytes for file size reasons, but:
- ✅ The actual API response object has the FULL challenge
- ✅ The LLM receives the FULL challenge text
- ✅ The telemetry truncation is only for logging, not for processing

## Why It Failed Before (and the Fix)

### The Bug:
```python
# Line 161 (OLD CODE - BUGGY):
print(f"Response message: {data.get('message')}")
# ↑ This contained emoji 🦞 which crashed on Windows
# ↓ Challenge detection code NEVER EXECUTED because of crash
if data.get("verification_required"):  # Never reached!
    ...
```

### The Fix:
```python
# NEW CODE - SAFE:
try:
    print(...)  # Wrapped in try/except
except:
    pass  # Ignore print errors, continue to challenge detection

# Challenge detection code ALWAYS runs now
if data.get("verification_required"):  # Always reached!
    solve_challenge()
```

## Challenge Types

The verification system uses a **generic LLM-based solver** that can handle ANY type of challenge:

### How It Works:
The solver receives:
1. **Challenge text** - May be obfuscated with random characters, mixed case, symbols
2. **Instructions** - Tells the LLM what format the answer should be in

The LLM:
1. Parses the obfuscated text to find the actual question
2. Determines what type of challenge it is (math, trivia, logic, word puzzle, etc.)
3. Solves it
4. Returns the answer in the format specified by the instructions

### Challenge Types Supported:
✅ **Math problems** - "Lobster swims at X m/s, slows by Y, what's new velocity?"
✅ **Trivia questions** - "What year did the first lobster reach space?"
✅ **Word puzzles** - "Unscramble these letters: BRTLOSE"
✅ **Logic problems** - "If a lobster has 10 legs and loses 2, how many remain?"
✅ **Yes/No questions** - "Is a lobster a mammal?"
✅ **Multiple choice** - "Which is faster: A) Lobster B) Snail C) Cheetah"
✅ **ANY text-based challenge** the LLM can understand

### Answer Formats Supported:
- Numbers with decimals: "28.00"
- Whole numbers: "42"
- Words: "lobster"
- Phrases: "eight legs"
- Yes/No: "yes" or "no"
- Letters: "C"
- **Whatever the instructions specify!**

### The Key Insight:
The solver **doesn't assume anything** about the challenge type or answer format. It:
1. Reads the challenge
2. Reads the instructions
3. Asks the LLM to follow those instructions
4. Returns whatever the LLM produces

This makes it **completely flexible** and able to adapt to new challenge types without code changes.

## Verification of the Fix

### To confirm it works, check telemetry for:

```
✅ POST /posts/xxx/comments → status 201 (comment created)
✅ POST /verify → status 200 (verification submitted)
✅ No suspension messages
✅ Comment appears on Moltbook
```

### If it's still failing:

1. Check if answer is correct:
   - Look for "math_verification_solved" event in telemetry
   - Check the "answer" field
   - Manually verify the math

2. Check if verification was submitted:
   - Look for POST to /verify endpoint
   - Check response status

3. Check expiration:
   - Challenges expire in 5 minutes
   - If solver takes too long, it will fail

## Configuration

### Adjust LLM settings if needed:

In `math_verification.py:64`:
```python
answer = self.llm_client.generate(
    prompt,
    temperature=0.0,    # Low = more deterministic
    max_output_tokens=50,
)
```

### Adjust timeouts if needed:

Challenges typically expire in 5 minutes, so the solver must:
1. Parse challenge < 1 second
2. Call LLM < 5 seconds
3. Submit answer < 1 second
Total: ~6 seconds (well within 5 minute limit)

## Summary

**Per Cycle LLM Calls**: 1-4 depending on what happens:
- 1 minimum (planner or DM fallback)
- +1 if follow decision enabled
- +1 if verification challenge triggered
- +1 if fallback/retry needed

**Helper Actions**: Run every cycle BEFORE the planner — upvote, subscribe, create submolt, follow. Most are probabilistic and don't require LLM calls (except follow).

**DM Fallback**: Only triggers when there's nothing else to do (no replies, no comments, post window closed). Skips the planner entirely.

**Challenge Visibility**: LLM sees the ENTIRE challenge text, nothing is truncated

**Flexibility**: The solver can handle any text-based challenge that an LLM can understand

**The Fix**: Wrapped all print statements to prevent Unicode crashes from blocking challenge detection

**Command-Line Flags**:
| Flag | Default | Controls |
|------|---------|----------|
| `--upvote-every-cycle` | True | Auto-upvote a random post each cycle |
| `--follow-on-like` | False | LLM-driven agent following |
| `--follow-prob` | 0.60 | Probability of following when enabled |
| `--subscribe-policy` | off | Submolt subscribe rate (off/low/medium/high) |
| `--allow-create-submolt` | False | Allow creating new submolts |
| `--create-submolt-prob` | 0.05 | Probability of creating when enabled |
| `--allow-dms` | True | Allow DM fallback |
| `--allow-votes` | True | Allow planner to suggest vote actions |
