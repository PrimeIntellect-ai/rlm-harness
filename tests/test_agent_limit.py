"""Tests for the cross-process live-agent cap (rlm._agent_limit)."""

import subprocess
import sys

from rlm import _agent_limit as lim


def test_cap_grants_up_to_limit_then_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "2")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    g1, m1 = lim.acquire_slot()
    g2, m2 = lim.acquire_slot()
    assert g1 and g2 and m1 is not None and m2 is not None

    g3, m3 = lim.acquire_slot()
    assert not g3 and m3 is None  # at capacity

    lim.release_slot(m1)
    g4, m4 = lim.acquire_slot()
    assert g4  # a slot freed up

    lim.release_slot(m2)
    lim.release_slot(m4)


def test_cap_disabled_when_limit_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("RLM_MAX_LIVE_AGENTS", raising=False)
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    for _ in range(20):
        granted, marker = lim.acquire_slot()
        assert granted and marker is None  # disabled -> always granted, no marker


def test_sweep_reclaims_dead_pid_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_MAX_LIVE_AGENTS", "2")
    monkeypatch.setenv("RLM_LIVE_AGENTS_DIR", str(tmp_path))

    g1, m1 = lim.acquire_slot()  # one live marker (this process)
    assert g1

    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()  # reaped -> pid is dead
    stale = tmp_path / f"{proc.pid}-deadbeef.marker"
    stale.write_text(str(proc.pid))

    # cap is 2; after the stale marker is swept, live = 1 (m1), so there's room.
    g2, m2 = lim.acquire_slot()
    assert g2 and not stale.exists()  # stale reclaimed

    g3, m3 = lim.acquire_slot()
    assert not g3  # now genuinely at capacity

    lim.release_slot(m1)
    lim.release_slot(m2)
