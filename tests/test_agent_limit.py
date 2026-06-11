"""Tests for the cross-process live-agent cap (rlm._agent_limit)."""

import asyncio
import subprocess
import sys

from rlm import _agent_limit as lim


def test_cap_grants_up_to_limit_then_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "2")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    g1, m1 = lim.acquire_slot(lim.TOTAL)
    g2, m2 = lim.acquire_slot(lim.TOTAL)
    assert g1 and g2 and m1 is not None and m2 is not None

    g3, m3 = lim.acquire_slot(lim.TOTAL)
    assert not g3 and m3 is None  # at capacity

    lim.release_slot(m1)
    g4, m4 = lim.acquire_slot(lim.TOTAL)
    assert g4  # a slot freed up

    lim.release_slot(m2)
    lim.release_slot(m4)


def test_pools_are_independent(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")  # total pool
    monkeypatch.setenv("RLM_MAX_RUNNING_AGENTS", "1")  # running pool
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    gt, mt = lim.acquire_slot(lim.TOTAL)
    assert gt
    gt2, _ = lim.acquire_slot(lim.TOTAL)
    assert not gt2  # total pool is full

    gr, mr = lim.acquire_slot(lim.RUNNING)
    assert gr  # running is a separate pool, unaffected by the full total pool

    lim.release_slot(mt)
    lim.release_slot(mr)


def test_cap_disabled_when_limit_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("RLM_MAX_LIVE_AGENTS", raising=False)
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    for _ in range(20):
        granted, marker = lim.acquire_slot(lim.TOTAL)
        assert granted and marker is None  # disabled -> always granted, no marker


def test_sweep_reclaims_dead_pid_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "2")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    g1, m1 = lim.acquire_slot(lim.TOTAL)  # one live marker (this process)
    assert g1

    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()  # reaped -> pid is dead
    stale = lim._markers_dir(lim.TOTAL) / f"{proc.pid}-deadbeef.marker"
    stale.write_text(str(proc.pid))

    # cap is 2; after the stale marker is swept, live = 1 (m1), so there's room.
    g2, m2 = lim.acquire_slot(lim.TOTAL)
    assert g2 and not stale.exists()  # stale reclaimed

    g3, m3 = lim.acquire_slot(lim.TOTAL)
    assert not g3  # now genuinely at capacity

    lim.release_slot(m1)
    lim.release_slot(m2)


async def test_acquire_blocking_waits_for_a_free_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    g1, m1 = lim.acquire_slot(lim.TOTAL)
    assert g1  # the only slot is taken

    waiter = asyncio.ensure_future(lim.acquire_slot_blocking(lim.TOTAL))
    await asyncio.sleep(0.4)
    assert not waiter.done()  # still blocked while full

    lim.release_slot(m1)
    m2 = await asyncio.wait_for(waiter, timeout=3)
    assert m2 is not None  # resolved once a slot freed
    lim.release_slot(m2)


async def test_acquire_blocking_times_out_and_proceeds_over_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "1")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))
    monkeypatch.setenv("RLM_AGENT_WAIT_TIMEOUT", "0.5")

    g1, m1 = lim.acquire_slot(lim.TOTAL)
    assert g1  # full

    m2 = await asyncio.wait_for(lim.acquire_slot_blocking(lim.TOTAL), timeout=3)
    assert m2 is not None  # forced over-cap after the wait timeout
    lim.release_slot(m1)
    lim.release_slot(m2)
