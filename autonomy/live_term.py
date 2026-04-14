"""Shared helpers for emitting status lines to the real terminal AND the
Analog Home live daemon feed at the same time.

The daemon subsystem (`autonomy/daemon.py`) has its own `_emit()` that does
both already. This module exists so the *conscious loop* and anything deeper
in the call stack (verifier, post-memory compressor) can follow the same
pattern without each caller threading `store`/`daemon`/`cycle` through every
function signature.

`_set_live_context()` is called once on startup from `__main__.py`, and any
module that imports `emit_status` inherits the context.
"""

from typing import Optional

from colorama import Fore, Style


# Module-level live context. Set once on startup by __main__.py.
_LIVE_CTX = {"store": None, "daemon": None}


def _set_live_context(store, daemon) -> None:
    """Called by __main__ on startup so deep-stack callers can emit."""
    _LIVE_CTX["store"] = store
    _LIVE_CTX["daemon"] = daemon


def _push_to_live(store, lines, daemon=None, cycle: int = 0) -> None:
    """Forward a list of lines to the Analog Home live daemon feed.

    Uses cycle + 10000 as the tick number so conscious events don't collide
    with daemon tick IDs.
    """
    if not store or not hasattr(store, 'push_daemon_tick'):
        return
    tick = cycle + 10000
    interval = 300
    if daemon and hasattr(daemon, '_ctrl'):
        interval = int(daemon._ctrl.get("sentry_interval_seconds") or 300)
    store.push_daemon_tick(tick, lines, interval, complete=False)


_LIVE_LINE_MAX = 140


def _safe_print(line: str) -> None:
    """Print with UnicodeEncodeError fallback for Windows consoles."""
    try:
        print(line)
    except UnicodeEncodeError:
        try:
            import sys
            sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        except Exception:
            print(line.encode("ascii", errors="replace").decode("ascii"))


def emit_status(tag: str, line: str, *, live_line: Optional[str] = None,
                color=None, store=None, daemon=None, cycle: int = 0,
                multiline: Optional[str] = None) -> None:
    """Emit a status line to the terminal AND the Analog Home live feed.

    Args:
      tag: bracket prefix like "[ACCOUNTANT]" or "[VERIFICATION]". Not duplicated
        if `line` already starts with the tag.
      line: the primary one-line message used for both surfaces.
      multiline: optional extra content printed to terminal only (e.g. the full
        verifier challenge body). Not mirrored to live.
      live_line: override for the live surface. Defaults to the combined tag+line
        truncated to 140 chars.
      color: colorama Fore.X for terminal coloring.
      store/daemon/cycle: if omitted, uses the module-level context set on startup.
    """
    if tag and not line.startswith(tag):
        term_line = f"{tag} {line}"
    else:
        term_line = line
    reset = Style.RESET_ALL if color else ""
    _safe_print(f"{color}{term_line}{reset}" if color else term_line)
    if multiline:
        _safe_print(f"{color}{multiline}{reset}" if color else multiline)

    if live_line is None:
        live_line = term_line
    if len(live_line) > _LIVE_LINE_MAX:
        live_line = live_line[:_LIVE_LINE_MAX - 1] + "…"

    _store = store if store is not None else _LIVE_CTX.get("store")
    _daemon = daemon if daemon is not None else _LIVE_CTX.get("daemon")
    if _store is not None:
        _push_to_live(_store, [live_line], _daemon, cycle=cycle)
