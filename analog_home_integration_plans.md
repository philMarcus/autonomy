# Autonomy → Analog_Home Integration Plan

## Objective

Refactor Autonomy so that:

1. All Moltbook posts and comments are archived to Analog_Home.
2. The agent can optionally publish artifacts to:
   - Moltbook
   - Analog_Home
   - Both
3. Analog_Home becomes the canonical archive of all agent artifacts.
4. Publishing destinations are cleanly abstracted (no hardcoded Moltbook coupling).

This change must not degrade existing Moltbook functionality.

---

# Current State

Autonomy currently:

- Generates content within a decision/action loop.
- Directly executes Moltbook-specific actions (post, reply, follow, etc.).
- Logs telemetry internally.
- Has no concept of a generalized “artifact.”

Content is tightly coupled to Moltbook execution.

---

# Target Architecture

Introduce a clean separation between:

1. Artifact Creation
2. Artifact Distribution

The autonomy loop should produce a canonical Artifact object.
Publishers then distribute that artifact to one or more destinations.

---

# New Core Concept: Artifact

Every publishable output becomes a structured Artifact.

## Artifact Fields (initial)

- artifact_id
- created_at
- title
- body_markdown
- monologue_public
- artifact_type (post | comment | reflection | etc.)
- destination (analog_home | moltbook | both)
- status (draft | published | failed)
- optional:
  - moltbook_post_id
  - moltbook_url
  - error_message

This replaces “generate text and immediately post to Moltbook.”

---

# Publishing Layer

Introduce publisher abstraction.

## Publisher Interface

Each publisher implements:
 publish(artifact) -> result

 
Initial publishers:

- AnalogHomePublisher
- MoltbookPublisher

---

# Destination Behavior

## Rule 1: Archive-of-Record

All artifacts are published to Analog_Home first.

Even if the destination is “moltbook_only”, the artifact is archived internally.

Analog_Home becomes canonical storage.

## Rule 2: Destination Routing

The agent decision output includes:

  destination:
    analog_home
    moltbook
    both


Execution:

- analog_home → publish only to site
- moltbook → publish to site + moltbook
- both → publish to site + moltbook

(Note: Moltbook-only does not skip archiving.)

---

# Required Changes to Autonomy

## 1. Introduce Artifact Construction Step

After generation but before execution:

- Construct Artifact object.
- Populate fields.
- Attach decision metadata.

This replaces direct Moltbook execution.

---

## 2. Refactor Moltbook Actions

Current behavior:
- autonomy decides to post → calls Moltbook client directly.

New behavior:
- autonomy constructs artifact
- MoltbookPublisher.publish(artifact)

MoltbookPublisher:
- Uses artifact.body_markdown
- On success, updates artifact with Moltbook metadata

---

## 3. Add Analog_Home Publisher

AnalogHomePublisher:
- Calls API endpoint: POST /publish
- Sends:
  - id
  - title
  - body_markdown
  - monologue_public
  - metadata

Failure of Analog_Home should not crash the autonomy loop.
Log and continue.

---

## 4. Decision Logic Update

Modify decision output schema to include:

- destination
- artifact_type

Destination selection may be:
- Config-driven
- LLM-driven
- Rule-based

Initially can default to “both” to avoid behavior changes.

---

## 5. Telemetry Updates

Add telemetry events for:

- artifact_created
- analog_home_publish_success
- moltbook_publish_success
- publish_failure

This allows cross-destination analysis later.

---

# Rollout Strategy

## Phase 1 – Archive Only (No Behavior Change)

- Keep Moltbook execution identical.
- After successful Moltbook post, construct Artifact and publish to Analog_Home.
- Destination hardcoded to “both.”

Goal:
- Verify archive works.
- No change to agent behavior.

---

## Phase 2 – Artifact-First Execution

- Artifact constructed before Moltbook publishing.
- MoltbookPublisher consumes artifact.
- AnalogHomePublisher consumes artifact.

Goal:
- Remove Moltbook coupling from generation step.

---

## Phase 3 – Destination Routing

- Enable analog_home-only artifacts.
- Enable destination selection in decision layer.
- Support non-Moltbook reflections.

Goal:
- Allow site-only outputs.
- Allow strategic publishing decisions.

---

# Expected Complexity

This is a moderate refactor.

Estimated effort:
- 1–2 focused sessions to introduce artifact layer
- 1 session to refactor Moltbook publisher
- 1 session for testing and stabilization

No core autonomy logic rewrite required.

---

# Benefits

- Decouples content generation from distribution.
- Prevents data loss if Moltbook suspends account.
- Creates a unified artifact history.
- Enables cross-platform analytics.
- Improves portfolio signal (multi-sink architecture).
- Prepares system for additional destinations in the future.

---

# Long-Term Extensions

After stabilization:

- Add artifact metadata backfill (e.g., Moltbook URL updates).
- Add retry queues for failed publishes.
- Add multi-destination scaling (RSS, email, other platforms).
- Migrate telemetry + artifacts into unified Postgres store.

---

# Final State Vision

Autonomy no longer “posts to Moltbook.”

Autonomy:

1. Decides.
2. Generates.
3. Constructs Artifact.
4. Publishes via routing layer.

Analog_Home becomes the permanent memory.
Moltbook becomes one distribution channel.

