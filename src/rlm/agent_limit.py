"""Cross-process caps on background sub-agents (two independent pools).

A rollout's process tree (root process + every kernel) shares one base marker
directory (``RLM_LIVE_AGENTS_DIR`` — derived from the root session and propagated
to each kernel). Two pools live in subdirectories under it:

- ``total`` (``RLM_MAX_LIVE_AGENTS``) — resident agents, a slot held from creation
  to teardown; bounds how many sub-agents (and their kernels) exist at once.
- ``running`` (``RLM_MAX_RUNNING_AGENTS``) — agents actively executing a turn, a
  slot held only around ``advance()``; bounds parallelism. Idle agents hold none.

Each reserved slot owns one marker file named ``<pid>-<uuid>.marker``; a pool's
live count is the number of its markers whose PID is still alive. An ``flock``
serializes the sweep-count-create across processes, and a PID-liveness sweep
reclaims slots leaked by hard-killed processes.

A pool's cap is active only when its env var is a positive int and a markers dir
+ POSIX ``fcntl`` are available; otherwise every acquire is granted. Assumes one
rollout per process tree (the harness runs rlm as a fresh process per rollout),
so live PIDs map cleanly to live agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from rlm.config import get_config

try:
    import fcntl
except ImportError:  # non-POSIX: cap disabled
    fcntl = None


class AgentLimitReached(RuntimeError):
    """Raised by ``rlm.send`` when the total-agent cap (RLM_MAX_LIVE_AGENTS) is hit."""


# Pool -> (limit env var, markers subdir under RLM_LIVE_AGENTS_DIR).
TOTAL = "total"
RUNNING = "running"
# Pool -> (Config attribute holding its limit, markers subdir name).
POOLS = {
    TOTAL: ("max_live_agents", "total"),
    RUNNING: ("max_running_agents", "running"),
}


def pool_limit(pool: str) -> int | None:
    return getattr(get_config(), POOLS[pool][0])


def pool_markers_dir(pool: str) -> Path | None:
    raw = get_config().live_agents_dir
    return Path(raw) / POOLS[pool][1] if raw else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def live_markers(markers_dir: Path) -> list[Path]:
    """Return live marker paths, unlinking any owned by a dead/garbage PID."""
    live: list[Path] = []
    for marker in markers_dir.glob("*.marker"):
        token = marker.name.split("-", 1)[0]
        try:
            pid = int(token)
        except ValueError:
            marker.unlink(missing_ok=True)
            continue
        if pid_alive(pid):
            live.append(marker)
        else:
            marker.unlink(missing_ok=True)
    return live


def make_marker(markers_dir: Path) -> Path:
    """Create and return one ``<pid>-<uuid>.marker`` file in ``markers_dir``."""
    marker = markers_dir / f"{os.getpid()}-{uuid.uuid4().hex}.marker"
    marker.write_text(str(os.getpid()))
    return marker


def acquire_slot(pool: str) -> tuple[bool, Path | None]:
    """Try to reserve a slot in ``pool`` (``TOTAL`` or ``RUNNING``).

    Returns ``(granted, marker)``. ``granted=False`` means the pool's cap is
    reached. ``marker`` is the file to hand to :func:`release_slot` (``None`` when
    the cap is disabled, so callers can release unconditionally).
    """
    limit = pool_limit(pool)
    markers_dir = pool_markers_dir(pool)
    if limit is None or markers_dir is None or fcntl is None:
        return True, None
    markers_dir.mkdir(parents=True, exist_ok=True)
    with open(markers_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if len(live_markers(markers_dir)) >= limit:
                return False, None
            return True, make_marker(markers_dir)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def release_slot(marker: Path | None) -> None:
    """Free a slot previously reserved via :func:`acquire_slot`."""
    if marker is not None:
        Path(marker).unlink(missing_ok=True)


WAIT_POLL_INTERVAL = 0.25


async def acquire_slot_blocking(pool: str) -> Path | None:
    """Reserve a slot in ``pool``, waiting until one frees (vs. the immediate no).

    Used by the per-turn running slot and the one-off ``rlm(...)`` path, which the
    model awaits anyway. Safety valve: after ``RLM_AGENT_WAIT_TIMEOUT`` seconds
    (default 300; ``0`` = wait forever) it force-reserves a slot over the cap and
    proceeds, so a rollout can't deadlock when slot-holders are themselves waiting
    for slots (e.g. a parent turn awaiting a nested child).
    """
    granted, marker = acquire_slot(pool)
    if granted:
        return marker
    timeout = get_config().agent_wait_timeout
    start = time.monotonic()
    while True:
        await asyncio.sleep(WAIT_POLL_INTERVAL)
        granted, marker = acquire_slot(pool)
        if granted:
            return marker
        if timeout is not None and time.monotonic() - start >= timeout:
            logging.getLogger(__name__).warning(
                "%s-agent cap wait exceeded %.0fs; proceeding over the cap",
                pool,
                timeout,
            )
            markers_dir = pool_markers_dir(pool)
            if markers_dir is None or fcntl is None:
                return None
            markers_dir.mkdir(parents=True, exist_ok=True)
            return make_marker(markers_dir)
