"""Tests for the background worker runtime (rlm.async_runtime).

Plain async functions; no kernel or engine — a worker is driven with trivial
in-process processors.
"""

import asyncio

import pytest

from rlm.async_runtime import (
    ERROR,
    FINISHED,
    RUNNING,
    BackgroundWorker,
    Registry,
)


class _Doubler:
    async def process(self, item):
        return item * 2

    async def teardown(self):
        pass


class _Gated:
    """Blocks in process() until the test releases the gate."""

    def __init__(self):
        self.gate = asyncio.Event()

    async def process(self, item):
        await self.gate.wait()
        return f"done:{item}"

    async def teardown(self):
        pass


class _Boom:
    async def process(self, item):
        raise ValueError("boom")

    async def teardown(self):
        pass


async def _settle(handle, *, want=FINISHED, tries=400):
    for _ in range(tries):
        if handle.poll().status == want:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"worker did not reach {want!r}: {handle.poll()!r}")


async def test_send_wait_returns_result():
    reg = Registry()
    handle = reg.send(5, name="w", worker_factory=lambda n: BackgroundWorker(n, _Doubler()))
    assert await handle.wait() == 10
    assert handle.poll().status == FINISHED


async def test_poll_does_not_consume_results():
    reg = Registry()
    handle = reg.send(3, name="w", worker_factory=lambda n: BackgroundWorker(n, _Doubler()))
    await _settle(handle)
    assert list(handle.poll().results) == [6]
    assert list(handle.poll().results) == [6]  # second poll still sees it
    assert handle.poll().results.popleft() == 6
    assert list(handle.poll().results) == []


async def test_running_status_while_processing():
    reg = Registry()
    proc = _Gated()
    handle = reg.send(1, name="w", worker_factory=lambda n: BackgroundWorker(n, proc))
    await asyncio.sleep(0.02)
    assert handle.poll().status == RUNNING
    proc.gate.set()
    await _settle(handle)
    assert list(handle.poll().results) == ["done:1"]


async def test_same_name_continues_one_worker():
    reg = Registry()
    created = []

    def factory(name):
        created.append(name)
        return BackgroundWorker(name, _Doubler())

    reg.send(2, name="spec", worker_factory=factory)
    handle = reg.send(3, name="spec", worker_factory=factory)
    assert created == ["spec"]  # factory built once; the second send reused the worker
    await _settle(handle)
    assert list(handle.poll().results) == [4, 6]  # FIFO order
    assert list(reg._workers) == ["spec"]


async def test_queued_is_editable():
    reg = Registry()
    proc = _Gated()
    handle = reg.send("a", name="w", worker_factory=lambda n: BackgroundWorker(n, proc))
    reg.send("b", name="w", worker_factory=lambda n: BackgroundWorker(n, proc))
    reg.send("c", name="w", worker_factory=lambda n: BackgroundWorker(n, proc))
    # "a" is being processed (gated); "b","c" are pending and editable.
    await asyncio.sleep(0.02)
    queued = handle.poll().queued
    assert queued == ["b", "c"]
    queued.remove("b")  # cancel a pending item directly on the live list
    proc.gate.set()
    await _settle(handle)
    assert list(handle.poll().results) == ["done:a", "done:c"]


async def test_error_surfaces_live_exception():
    reg = Registry()
    handle = reg.send(1, name="w", worker_factory=lambda n: BackgroundWorker(n, _Boom()))
    await _settle(handle, want=ERROR)
    state = handle.poll()
    assert state.status == ERROR
    assert isinstance(state.error, ValueError)
    with pytest.raises(ValueError, match="boom"):
        await handle.wait()
    # re-sending an errored name restarts it: the dead worker is evicted and a
    # fresh one is built under the same name.
    handle2 = reg.send(
        9, name="w", worker_factory=lambda n: BackgroundWorker(n, _Doubler())
    )
    assert await handle2.wait() == 18
    assert reg.get("w") is not None


async def test_close_all_tears_down_workers():
    class _P:
        def __init__(self):
            self.torn = False

        async def process(self, item):
            return item

        async def teardown(self):
            self.torn = True

    reg = Registry()
    proc = _P()
    handle = reg.send(1, name="w", worker_factory=lambda n: BackgroundWorker(n, proc))
    await _settle(handle)
    await reg.close_all()
    assert reg.get("w") is None
    assert proc.torn is True


async def test_auto_named_send_registers_a_worker():
    reg = Registry()
    handle = reg.send(4, name=None, worker_factory=lambda n: BackgroundWorker(n, _Doubler()))
    assert reg.get(handle.name) is not None
    assert await handle.wait() == 8


async def test_attach_background_adds_send():
    import types

    from rlm.async_runtime import attach_background

    mod = types.ModuleType("fakeskill")

    async def run(x, *, plus=1):
        return x + plus

    mod.run = run
    attach_background(mod, mod.run)

    assert await mod.send(41).wait() == 42  # forwards positional args
    assert await mod.send(10, plus=5).wait() == 15  # forwards kwargs

    # ephemeral: a send is a standalone worker that runs once and ends — it is
    # not held in any registry, so it's GC'd once the handle is dropped.
    handle = mod.send(7)
    assert await handle.wait() == 8
    await asyncio.sleep(0)
    assert handle._worker._task.done()
