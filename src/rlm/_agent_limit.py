"""Cross-process cap on the number of live background sub-agents.

A rollout's process tree (root process + every kernel) shares one marker
directory (``RLM_LIVE_AGENTS_DIR`` — derived from the root session and
propagated to each kernel). Each live agent owns one marker file named
``<pid>-<uuid>.marker``; the live count is the number of markers whose PID is
still alive. An ``flock`` serializes the sweep-count-create across processes,
and a PID-liveness sweep reclaims slots leaked by hard-killed processes — which
a plain integer counter could not.

The cap is active only when ``RLM_MAX_LIVE_AGENTS`` is a positive int and a
markers dir + POSIX ``fcntl`` are available; otherwise every acquire is granted.
Assumes one rollout per process tree (the harness runs rlm as a fresh process
per rollout), so live PIDs map cleanly to live agents.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX: cap disabled
    fcntl = None


class AgentLimitReached(RuntimeError):
    """Raised by ``rlm.send`` when the ``RLM_MAX_LIVE_AGENTS`` cap is hit."""


def _limit() -> int | None:
    value = int(os.environ.get("RLM_MAX_LIVE_AGENTS", "0"))
    return value if value > 0 else None


def _markers_dir() -> Path | None:
    raw = os.environ.get("RLM_LIVE_AGENTS_DIR")
    return Path(raw) if raw else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _live_markers(markers_dir: Path) -> list[Path]:
    """Return live marker paths, unlinking any owned by a dead/garbage PID."""
    live: list[Path] = []
    for marker in markers_dir.glob("*.marker"):
        token = marker.name.split("-", 1)[0]
        try:
            pid = int(token)
        except ValueError:
            marker.unlink(missing_ok=True)
            continue
        if _pid_alive(pid):
            live.append(marker)
        else:
            marker.unlink(missing_ok=True)
    return live


def acquire_slot() -> tuple[bool, Path | None]:
    """Try to reserve a live-agent slot.

    Returns ``(granted, marker)``. ``granted=False`` means the cap is reached.
    ``marker`` is the file to hand to :func:`release_slot` (``None`` when the cap
    is disabled, so callers can release unconditionally).
    """
    limit = _limit()
    markers_dir = _markers_dir()
    if limit is None or markers_dir is None or fcntl is None:
        return True, None
    markers_dir.mkdir(parents=True, exist_ok=True)
    with open(markers_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if len(_live_markers(markers_dir)) >= limit:
                return False, None
            marker = markers_dir / f"{os.getpid()}-{uuid.uuid4().hex}.marker"
            marker.write_text(str(os.getpid()))
            return True, marker
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def release_slot(marker: Path | None) -> None:
    """Free a slot previously reserved via :func:`acquire_slot`."""
    if marker is not None:
        Path(marker).unlink(missing_ok=True)
