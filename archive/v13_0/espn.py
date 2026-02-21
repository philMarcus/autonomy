"""ESPN external data injection for planner context."""

import json
import time
import datetime
from typing import Any, Dict, List, Optional

import requests

from . import VERSION
from .config import ESPN_DEFAULT_LEAGUE, ESPN_LEAGUE_MAP
from .telemetry import TelemetryLogger


def espn_scoreboard_url(league: str) -> str:
    league = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    if "/" in league:
        sport, lg = league.split("/", 1)
    elif league in ESPN_LEAGUE_MAP:
        sport, lg = ESPN_LEAGUE_MAP[league]
    else:
        sport, lg = ESPN_LEAGUE_MAP["nfl"]
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"


def espn_summary_url(league: str) -> str:
    league = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    if "/" in league:
        sport, lg = league.split("/", 1)
    elif league in ESPN_LEAGUE_MAP:
        sport, lg = ESPN_LEAGUE_MAP[league]
    else:
        sport, lg = ESPN_LEAGUE_MAP["nfl"]
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/summary"


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None,
                   timeout_s: int = 8, telemetry: Optional[TelemetryLogger] = None) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout_s,
                         headers={"User-Agent": f"autonomy/{VERSION}"})
        if r.status_code != 200:
            if telemetry:
                telemetry.log("external_api_error", {"provider": "http", "url": url, "status": r.status_code})
            return None
        return r.json()
    except Exception as e:
        if telemetry:
            telemetry.log("external_api_error", {"provider": "http", "url": url, "error": str(e)})
        return None


def _espn_compact_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        comps = ev.get("competitions") or []
        comp = comps[0] if isinstance(comps, list) and comps else {}
        status = (comp or ev).get("status") or {}
        st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
        state_str = (st_type.get("state") or "").lower()
        detail = st_type.get("detail") or ""
        competitors = []
        if isinstance(comp, dict):
            for team in (comp.get("competitors") or []):
                if not isinstance(team, dict):
                    continue
                t = team.get("team") or {}
                competitors.append({
                    "abbr": t.get("abbreviation"),
                    "team": t.get("shortDisplayName") or t.get("displayName") or t.get("name"),
                    "homeAway": team.get("homeAway"),
                    "score": team.get("score"),
                })
        out.append({
            "id": ev.get("id") or (comp.get("id") if isinstance(comp, dict) else None),
            "name": ev.get("shortName") or ev.get("name"),
            "date": ev.get("date"),
            "state": state_str,
            "detail": detail,
            "competitors": competitors,
        })
    return out


def _safe_get(obj: Any, path: List[Any]) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if isinstance(cur, list) and 0 <= key < len(cur):
                cur = cur[key]
            else:
                return None
        else:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
    return cur


def _espn_extract_best_play_summaries(summary_json: Dict[str, Any], max_plays: int = 20) -> Dict[str, Any]:
    plays = _safe_get(summary_json, ["plays"])
    if not isinstance(plays, list):
        plays = _safe_get(summary_json, ["drives", "current", "plays"])
    if not isinstance(plays, list):
        plays = _safe_get(summary_json, ["drives", "previous", 0, "plays"])
    compact_plays: List[Dict[str, Any]] = []
    if isinstance(plays, list):
        for p in plays[-max_plays:]:
            if not isinstance(p, dict):
                continue
            compact_plays.append({
                "clock": p.get("clock", {}).get("displayValue") if isinstance(p.get("clock"), dict) else p.get("clock"),
                "period": _safe_get(p, ["period", "number"]) or _safe_get(p, ["period"]),
                "text": p.get("text") or p.get("shortText") or p.get("headline"),
                "type": _safe_get(p, ["type", "text"]) or _safe_get(p, ["type", "abbreviation"]),
                "scoringPlay": p.get("scoringPlay"),
                "homeScore": p.get("homeScore"),
                "awayScore": p.get("awayScore"),
            })

    situation = _safe_get(summary_json, ["situation"])
    if not isinstance(situation, dict):
        situation = _safe_get(summary_json, ["header", "competitions", 0, "situation"])
    if not isinstance(situation, dict):
        situation = {}

    sit_out = {}
    if isinstance(situation, dict):
        sit_out = {
            "downDistanceText": situation.get("downDistanceText"),
            "possession": situation.get("possession"),
            "yardLine": situation.get("yardLine"),
            "shortDownDistanceText": situation.get("shortDownDistanceText"),
            "lastPlay": situation.get("lastPlay", {}).get("text") if isinstance(situation.get("lastPlay"), dict) else None,
        }

    return {
        "situation": sit_out,
        "recentPlays": compact_plays,
    }


def _espn_pick_event(events: List[Dict[str, Any]], keywords: str = "") -> Optional[Dict[str, Any]]:
    if not events:
        return None
    kw = (keywords or "").lower().strip()
    if kw:
        kw_matches = [ev for ev in events if kw in (ev.get("name") or ev.get("shortName") or "").lower()]
        if len(kw_matches) == 1:
            return kw_matches[0]
        if len(kw_matches) > 1:
            events = kw_matches

    def ev_state(ev: Dict[str, Any]) -> str:
        comp = None
        comps = ev.get("competitions") or []
        if comps and isinstance(comps, list):
            comp = comps[0]
        status = (comp or ev).get("status") or {}
        st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
        return (st_type.get("state") or "").lower()

    def ev_time(ev: Dict[str, Any]) -> float:
        ds = ev.get("date") or ""
        try:
            ds2 = ds.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(ds2).timestamp()
        except Exception:
            return 0.0

    live = [ev for ev in events if ev_state(ev) == "in"]
    if live:
        return sorted(live, key=ev_time, reverse=True)[0]
    upcoming = [ev for ev in events if ev_state(ev) == "pre"]
    if upcoming:
        return sorted(upcoming, key=ev_time)[0]
    finished = [ev for ev in events if ev_state(ev) == "post"]
    if finished:
        return sorted(finished, key=ev_time, reverse=True)[0]
    return events[0]


def get_espn_context(
    state: Dict[str, Any],
    league: str,
    date_yyyymmdd: str = "",
    cache_seconds: int = 60,
    keywords: str = "",
    include_summary: bool = True,
    telemetry: Optional[TelemetryLogger] = None,
) -> str:
    now = time.time()
    league_key = (league or ESPN_DEFAULT_LEAGUE).strip().lower()
    date_key = (date_yyyymmdd or "").strip()
    if not date_key:
        date_key = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    cache = state.get("espn_cache") if isinstance(state.get("espn_cache"), dict) else {}
    cache_id = f"{league_key}:{date_key}:{(keywords or '').strip().lower()}:{'1' if include_summary else '0'}"
    ts = float(cache.get("ts", 0) or 0)
    if cache.get("id") == cache_id and (now - ts) < max(5, cache_seconds) and isinstance(cache.get("text"), str):
        return cache.get("text") or ""

    sb_url = espn_scoreboard_url(league_key)
    data = _http_get_json(sb_url, params={"dates": date_key}, telemetry=telemetry)
    if not isinstance(data, dict):
        return ""

    events_raw = [ev for ev in (data.get("events") or []) if isinstance(ev, dict)]
    compact_events = _espn_compact_events(events_raw)
    target = _espn_pick_event(events_raw, keywords=keywords)
    if not isinstance(target, dict):
        return ""

    comp = None
    comps = target.get("competitions") or []
    if comps and isinstance(comps, list):
        comp = comps[0]

    status = (comp or target).get("status") or {}
    st_type = (status.get("type") or {}) if isinstance(status, dict) else {}
    state_str = st_type.get("state") or st_type.get("description") or ""
    detail = (st_type.get("detail") or "") if isinstance(st_type, dict) else ""

    competitors = []
    if isinstance(comp, dict):
        for team in (comp.get("competitors") or []):
            if not isinstance(team, dict):
                continue
            t = team.get("team") or {}
            competitors.append({
                "team": t.get("displayName") or t.get("shortDisplayName") or t.get("name"),
                "abbr": t.get("abbreviation"),
                "score": team.get("score"),
                "winner": team.get("winner"),
                "homeAway": team.get("homeAway"),
            })

    event_id = target.get("id") or (comp.get("id") if isinstance(comp, dict) else None)

    selected_summary: Dict[str, Any] = {}
    if include_summary and event_id:
        sum_url = espn_summary_url(league_key)
        sum_json = _http_get_json(sum_url, params={"event": event_id}, telemetry=telemetry)
        if isinstance(sum_json, dict):
            selected_summary = _espn_extract_best_play_summaries(sum_json, max_plays=25)
            if telemetry:
                telemetry.log("external_api_call", {"provider": "espn", "url": sum_url, "event": str(event_id), "cached": False})

    summary = {
        "provider": "espn",
        "league": league_key,
        "date": date_key,
        "events": compact_events,
        "selected_event_id": event_id,
        "selected_event": {
            "event": target.get("name") or target.get("shortName"),
            "event_date": target.get("date"),
            "status": state_str,
            "detail": detail,
            "competitors": competitors,
        },
        "selected_event_summary": selected_summary,
    }

    text_out = json.dumps(summary, ensure_ascii=False)
    state["espn_cache"] = {"id": cache_id, "ts": now, "text": text_out}
    if telemetry:
        telemetry.log("external_api_call", {"provider": "espn", "url": sb_url, "date": date_key, "cached": False})

    return text_out
