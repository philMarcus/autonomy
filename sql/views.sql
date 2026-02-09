-- sql/views.sql
-- Canonical semantic layer for Autonomy telemetry

.open warehouse/telemetry.duckdb

-- -------------------------------------------------------------------
-- Base events view (the foundation)
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW events AS
SELECT *
FROM read_parquet('warehouse/events/dt=*/events_*.parquet');

-- -------------------------------------------------------------------
-- LLM activity
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW llm_calls AS
SELECT *
FROM events
WHERE event_type IN ('llm_call', 'llm_request');

-- -------------------------------------------------------------------
-- External API calls (Moltbook, ESPN, etc.)
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW api_calls AS
SELECT *
FROM events
WHERE event_type LIKE '%_api_call';

-- -------------------------------------------------------------------
-- Executed / blocked actions
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW actions AS
SELECT *
FROM events
WHERE event_type IN ('action_executed', 'action_blocked', 'action_skipped');

-- -------------------------------------------------------------------
-- Errors (centralized)
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW errors AS
SELECT *
FROM events
WHERE event_type IN ('error', 'llm_exception', 'external_api_error');

-- -------------------------------------------------------------------
-- Cycle-level rollup (used by dashboard)
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW cycle_summary AS
SELECT
  run_id,
  brain,
  cycle_num,
  min(ts) AS cycle_start,
  max(ts) AS cycle_end,
  CAST(min(ts) AS DATE) AS cycle_dt,
  count(*) AS events_in_cycle,

  sum(CASE WHEN event_type = 'llm_call' THEN coalesce(prompt_chars,0) ELSE 0 END) AS prompt_chars,
  sum(CASE WHEN event_type = 'llm_call' THEN coalesce(response_chars,0) ELSE 0 END) AS response_chars,
  sum(CASE WHEN event_type = 'llm_call' THEN 1 ELSE 0 END) AS llm_calls,

  sum(CASE WHEN event_type LIKE '%_api_call' THEN 1 ELSE 0 END) AS api_calls,
  sum(CASE WHEN http_status = 429 THEN 1 ELSE 0 END) AS rate_limited_429,

  sum(CASE WHEN event_type = 'action_executed' THEN 1 ELSE 0 END) AS actions_executed

FROM events
WHERE cycle_num IS NOT NULL
GROUP BY 1,2,3;
