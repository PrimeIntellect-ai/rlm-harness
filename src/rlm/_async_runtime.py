"""Background workers for programmatic tool calls inside the IPython kernel.

``send(...)`` schedules a tool invocation as a background asyncio task on the
kernel's event loop and returns a :class:`Handle` the model can ``poll()`` across
cells. The kernel loop keeps these tasks progressing between cells (verified —
see ``docs/async-tools-and-interruptions.md``).

A worker owns an inbox and drains it sequentially via a :class:`Processor`:

- the default stateless processor runs each item as an independent call;
- ``rlm`` supplies a stateful processor (a live agent) so re-sending the same
  name continues a multi-turn conversation.

This module is tool-agnostic: it knows nothing about ``rlm`` or the engine.
"""

from __future__ import annotations

import asyncio
import hashlib
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

RUNNING = "running"
FINISHED = "finished"  # idle: inbox empty, ready for more
ERROR = "error"


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
    """A per-name background task draining an inbox sequentially.

    Lives for the worker's lifetime on the kernel loop: parked on ``_wake`` when
    idle, running the processor when items arrive. An exception from the
    processor halts the worker (terminal ``ERROR`` state); the live exception is
    surfaced via :attr:`error`.
    """

    def __init__(
        self,
        name: str,
        processor: Processor,
        *,
        session_dir: Path | None = None,
    ):
        self.name = name
        self.session_dir = session_dir
        self.queued: list = []
        self.results: deque = deque()
        self._processor = processor
        self._status = FINISHED
        self._error: BaseException | None = None
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()
        self._closing = False
        self._task: asyncio.Future = asyncio.ensure_future(self._drain())

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> BaseException | None:
        return self._error

    def state(self) -> ToolState:
        return ToolState(
            status=self._status,
            results=self.results,
            queued=self.queued,
            error=self._error,
        )

    def submit(self, item: Any) -> None:
        if self._status == ERROR:
            raise RuntimeError(
                f"worker {self.name!r} halted with an error; send to a different name"
            )
        self.queued.append(item)
        self._status = RUNNING
        self._wake.set()

    async def _drain(self) -> None:
        while not self._closing:
            if not self.queued:
                self._status = FINISHED
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
    """Per-tool, per-kernel collection of named workers.

    Each agent owns the registry of the children *it* spawned (the module that
    holds this instance lives in the kernel process), so nesting is naturally
    hierarchical with no global registry.
    """

    def __init__(self, *, name_seed: str = ""):
        self._workers: dict[str, BackgroundWorker] = {}
        self._name_seed = name_seed
        self._auto_counter = 0
        _ALL_REGISTRIES.add(self)

    def get(self, name: str) -> Handle | None:
        worker = self._workers.get(name)
        return Handle(worker) if worker is not None else None

    def list(self) -> list[str]:
        return sorted(self._workers)

    def _auto_name(self) -> str:
        while True:
            candidate = _auto_name(self._auto_counter, seed=self._name_seed)
            self._auto_counter += 1
            if candidate not in self._workers:
                return candidate

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
            name = self._auto_name()
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


def attach_background(module, run_callable, *, name_seed: str = ""):
    """Give a wrapped callable module a stateless ``.send(*a, **kw) -> Handle``.

    ``send`` runs ``run_callable(*a, **kw)`` on a background worker (auto-named,
    no persistence, no live-agent cap — skills are cheap coroutines, not
    kernels). The model holds the returned handle and polls it. Used to give
    uploaded skills the same background/poll lifecycle as sub-agents.
    """
    registry = Registry(name_seed=name_seed)

    def send(*args, **kwargs):
        return registry.send(
            (args, kwargs),
            name=None,
            processor_factory=lambda _name: FnProcessor(run_callable),
        )

    module.send = send
    return module


_NAME_ADJECTIVES = (
    "amber",
    "brave",
    "calm",
    "dapper",
    "eager",
    "fluffy",
    "gentle",
    "hidden",
    "jolly",
    "keen",
    "lucky",
    "mellow",
    "noble",
    "quiet",
    "rapid",
    "sunny",
    "tidy",
    "vivid",
    "witty",
    "zesty",
)
_NAME_NOUNS = (
    "sky",
    "river",
    "forest",
    "meadow",
    "canyon",
    "harbor",
    "glacier",
    "ember",
    "comet",
    "willow",
    "summit",
    "lagoon",
    "tundra",
    "prairie",
    "delta",
    "cove",
)
_NAME_ANIMALS = (
    "bison",
    "otter",
    "falcon",
    "lynx",
    "heron",
    "marten",
    "ibis",
    "tapir",
    "gecko",
    "quokka",
    "narwhal",
    "puffin",
    "badger",
    "civet",
    "wren",
    "shrew",
)


def _auto_name(index: int, *, seed: str = "") -> str:
    """Deterministic ``adjective-noun-animal`` name from ``(seed, index)``.

    Deterministic so rollouts of the same prompt (same seed) draw identical
    names — important for reproducibility in a training harness.
    """
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    adjective = _NAME_ADJECTIVES[digest[0] % len(_NAME_ADJECTIVES)]
    noun = _NAME_NOUNS[digest[1] % len(_NAME_NOUNS)]
    animal = _NAME_ANIMALS[digest[2] % len(_NAME_ANIMALS)]
    return f"{adjective}-{noun}-{animal}"
