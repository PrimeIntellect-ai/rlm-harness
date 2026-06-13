"""Background workers for programmatic tool calls inside the IPython kernel.

``send(...)`` schedules a tool invocation as a background asyncio task on the
kernel's event loop and returns a :class:`Handle` the model can ``poll()`` across
cells. The kernel loop keeps these tasks progressing between cells (see
``docs/background-subagents-and-tools.md``).

A worker owns an inbox and drains it sequentially via a :class:`Processor`:

- the default stateless processor runs each item as an independent call;
- ``rlm`` supplies a stateful processor (a live agent) so re-sending the same
  name continues a multi-turn conversation.
"""

from __future__ import annotations

import asyncio
import uuid
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

RUNNING = "running"
FINISHED = "finished"  # idle: inbox empty, ready for more
ERROR = "error"

_NO_ITEM = object()  # omitted from BackgroundWorker(item=...) -> resident worker


@dataclass
class ToolState:
    """A ``poll()`` snapshot.

    ``results`` and ``queued`` are the worker's **live** structures, not copies:
    the model drains ``results`` (``.popleft()``) and edits ``queued`` directly
    (cancel / reorder / edit pending items). ``poll()`` itself never consumes.
    Editing is race-free because the kernel loop is single-threaded cooperative —
    a sync edit in a cell can't interleave with the drain coroutine.
    """

    status: str
    results: deque
    queued: list
    error: BaseException | None = None

    def __repr__(self) -> str:
        # Bounded: a bare ``handle.poll()`` echoes this repr into model context.
        parts = [f"status={self.status!r}", f"results={len(self.results)}"]
        if self.queued:
            parts.append(f"queued={len(self.queued)}")
        if self.error is not None:
            parts.append(f"error={type(self.error).__name__}")
        return f"ToolState({', '.join(parts)})"


@runtime_checkable
class Processor(Protocol):
    """Turns one queued input into a result, plus one-shot teardown.

    ``process`` is called once per queued item, in order; ``teardown`` runs once
    when the worker is closed.
    """

    async def process(self, item: Any) -> Any: ...

    async def teardown(self) -> None: ...


class FnProcessor:
    """Default stateless processor: each item is an independent call to ``fn``.

    Items are ``(args, kwargs)`` tuples produced by :meth:`Registry.send`.
    """

    def __init__(self, fn: Callable[..., Awaitable[Any]]):
        self._fn = fn

    async def process(self, item: Any) -> Any:
        args, kwargs = item
        return await self._fn(*args, **kwargs)

    async def teardown(self) -> None:
        return None


class BackgroundWorker:
    """A background task that runs a processor on the kernel loop.

    Two lifecycles, selected by the ``item`` constructor arg: a *resident* worker
    drains an inbox sequentially and parks on ``_wake`` when idle (named rlm
    agents); an *ephemeral* worker runs its single item once and ends (general
    tools). A processor exception halts the worker in ``ERROR`` state, surfaced
    to the model via ``poll().error``.
    """

    def __init__(
        self,
        name: str,
        processor: Processor,
        *,
        session_dir: Path | None = None,
        item: Any = _NO_ITEM,
    ):
        self.name = name
        self.session_dir = session_dir
        self.results: deque = deque()
        self._processor = processor
        self._error: BaseException | None = None
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._closing = False
        # A resident worker (named rlm agent) starts empty and parks for more
        # sends until close(); an ephemeral worker (general tool) is created with
        # its one item and ends as soon as the inbox drains — it lives only as
        # long as its Handle, never registered, so it's GC'd when dropped.
        self._ephemeral = item is not _NO_ITEM
        self.queued: list = [item] if self._ephemeral else []
        self._status = RUNNING if self._ephemeral else FINISHED
        self._task: asyncio.Future = asyncio.ensure_future(self._drain())

    @property
    def status(self) -> str:
        return self._status

    def state(self) -> ToolState:
        return ToolState(
            status=self._status,
            results=self.results,
            queued=self.queued,
            error=self._error,
        )

    def submit(self, item: Any) -> None:
        self.queued.append(item)
        self._status = RUNNING
        self._wake.set()

    async def _drain(self) -> None:
        while not self._closing:
            if not self.queued:
                self._status = FINISHED
                if self._ephemeral:
                    return  # one-shot: its item is done, let the task end
                self._wake.clear()
                if self.queued or self._closing:  # re-check after clear
                    continue
                await self._wake.wait()
                continue
            item = self.queued.pop(0)
            self._status = RUNNING
            try:
                result = await self._processor.process(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # surfaced to the model via poll(); halts worker
                self._error = exc
                self._status = ERROR
                self._progress.set()
                return
            self.results.append(result)
            self._progress.set()

    async def wait(self) -> Any:
        """Await and return the next result (consumes it); raise if the worker errors."""
        while True:
            if self.results:
                return self.results.popleft()
            if self._status == ERROR:
                assert self._error is not None
                raise self._error
            if self._task.done():
                raise RuntimeError(f"worker {self.name!r} ended without a result")
            self._progress.clear()
            if self.results or self._status == ERROR:  # re-check after clear
                continue
            await self._progress.wait()

    async def close(self) -> None:
        """Cancel the drain task and run processor teardown (idempotent)."""
        self._closing = True
        self._wake.set()
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self._processor.teardown()


class Handle:
    """Model-facing reference to a background worker."""

    def __init__(self, worker: BackgroundWorker):
        self._worker = worker

    @property
    def name(self) -> str:
        return self._worker.name

    @property
    def session_dir(self) -> Path | None:
        """Session dir of the worker, if any. ``session_dir/'messages.jsonl'`` is
        the live transcript (rlm); ``None`` for tools without one."""
        return self._worker.session_dir

    def poll(self) -> ToolState:
        return self._worker.state()

    async def wait(self) -> Any:
        return await self._worker.wait()

    def __repr__(self) -> str:
        return f"Handle(name={self._worker.name!r}, status={self._worker.status!r})"


_ALL_REGISTRIES: "weakref.WeakSet[Registry]" = weakref.WeakSet()


async def close_all_registries() -> None:
    """Gracefully close every live registry's workers (used on kernel teardown)."""
    for registry in list(_ALL_REGISTRIES):
        await registry.close_all()


class Registry:
    """Per-tool, per-kernel collection of named workers."""

    def __init__(self):
        self._workers: dict[str, BackgroundWorker] = {}
        _ALL_REGISTRIES.add(self)

    def get(self, name: str) -> Handle | None:
        worker = self._workers.get(name)
        return Handle(worker) if worker is not None else None

    def send(
        self,
        item: Any,
        *,
        name: str | None,
        processor_factory: Callable[[str], Processor],
        session_dir_factory: Callable[[str], Path | None] | None = None,
    ) -> Handle:
        """Enqueue ``item`` to a worker named ``name`` (creating it if needed).

        A new (or replacement-of-errored) worker is built via
        ``processor_factory(name)``; ``session_dir_factory(name)`` supplies its
        session dir. Re-sending an existing name continues that worker.
        """
        if name is None:
            name = uuid.uuid4().hex
        worker = self._workers.get(name)
        if worker is not None and worker.status == ERROR:
            raise RuntimeError(
                f"agent {name!r} halted with an error; start a fresh one under a different name"
            )
        if worker is None:
            session_dir = (
                session_dir_factory(name) if session_dir_factory is not None else None
            )
            worker = BackgroundWorker(
                name, processor_factory(name), session_dir=session_dir
            )
            self._workers[name] = worker
        worker.submit(item)
        return Handle(worker)

    async def close_all(self) -> None:
        """Gracefully close every worker (used on kernel/agent teardown)."""
        workers = list(self._workers.values())
        self._workers.clear()
        for worker in workers:
            try:
                await worker.close()
            except Exception:
                pass


def attach_background(module, run_callable):
    """Give a wrapped callable module a stateless ``.send(*a, **kw) -> Handle``.

    Each ``send`` runs ``run_callable(*a, **kw)`` on its own ephemeral worker and
    returns a handle. The worker runs the call once and ends; it is not
    registered, so it and its result live only as long as the model holds the
    handle. A tool that wants state across calls keeps its own cache.
    """

    def send(*args, **kwargs):
        worker = BackgroundWorker(
            uuid.uuid4().hex, FnProcessor(run_callable), item=(args, kwargs)
        )
        return Handle(worker)

    module.send = send
    return module
