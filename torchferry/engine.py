"""Single-worker, deadline-aware dynamic inference batching."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapters import Adapter
from .errors import BatchExecutionError, DeadlineExceeded, EngineClosed, QueueFull, RequestTooLarge


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_batch_size: int = 8
    max_batch_cost: int = 1024
    max_wait: float = 0.005
    max_pending: int = 128

    def __post_init__(self) -> None:
        for name in ("max_batch_size", "max_batch_cost", "max_pending"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.max_wait) or self.max_wait < 0:
            raise ValueError("max_wait must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    accepted: int
    completed: int
    rejected: int
    expired: int
    cancelled: int
    failed: int
    batches: int
    batch_items: int
    queued: int
    in_flight: int
    aborted: int


@dataclass(eq=False, slots=True)
class _Request:
    payload: Tensor
    cost: int
    enqueued: float
    deadline: float | None
    future: asyncio.Future[Tensor]
    timer: asyncio.TimerHandle | None = None
    dispatched: bool = False


class InferenceEngine:
    """A single-use engine on one asyncio event loop.

    Inputs are borrowed until their physical batch retires. In particular, a
    timed-out or cancelled model invocation cannot be interrupted safely.
    """

    def __init__(self, adapter: Adapter, config: EngineConfig = EngineConfig()):
        self.adapter = adapter
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake = asyncio.Event()
        self._queue: deque[_Request] = deque()
        self._active: list[_Request] = []
        self._executor: ThreadPoolExecutor | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closing = False
        self._counts = dict.fromkeys(
            ("accepted", "completed", "rejected", "expired", "cancelled",
             "failed", "batches", "batch_items", "aborted"), 0
        )

    @property
    def metrics(self) -> EngineMetrics:
        """Capture counters and physical queue/worker occupancy."""
        return EngineMetrics(**self._counts, queued=len(self._queue), in_flight=len(self._active))

    async def start(self) -> InferenceEngine:
        loop = asyncio.get_running_loop()
        if self._closing:
            raise EngineClosed("engine is closed")
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("engine belongs to a different event loop")
        if self._worker is None:
            self._loop = loop
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torchferry")
            self._worker = loop.create_task(self._serve(), name="torchferry-worker")
        return self

    async def __aenter__(self) -> InferenceEngine:
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self, *, drain: bool = True) -> None:
        """Stop admission; optionally fail all outstanding caller futures.

        Either mode waits for a running model call to finish. Cancellation of
        a waiter does not cancel the engine shutdown.
        """
        if self._loop is not None and self._loop is not asyncio.get_running_loop():
            raise RuntimeError("engine belongs to a different event loop")
        self._closing = True
        if not drain:
            while self._queue:
                self._finish(self._queue.popleft(), "aborted", EngineClosed("engine stopped"))
            for request in self._active:
                self._finish(request, "aborted", EngineClosed("engine stopped"))
        self._wake.set()
        if self._worker is not None:
            await asyncio.shield(self._worker)

    async def infer(self, payload: Tensor, *, timeout: float | None = None) -> Tensor:
        """Enqueue an inference; timeout includes both queue and model time."""
        loop = asyncio.get_running_loop()
        if self._closing or self._loop is None:
            self._counts["rejected"] += 1
            raise EngineClosed("engine is not running")
        if loop is not self._loop:
            raise RuntimeError("engine belongs to a different event loop")
        now = loop.time()
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        if len(self._queue) + len(self._active) >= self.config.max_pending:
            self._counts["rejected"] += 1
            raise QueueFull("pending capacity reached")
        cost = self.adapter.cost(payload)
        if type(cost) is not int or cost < 1:
            raise ValueError("adapter cost must be a positive integer")
        if cost > self.config.max_batch_cost:
            self._counts["rejected"] += 1
            raise RequestTooLarge("request exceeds the batch cost limit")

        request = _Request(payload, cost, now, None if timeout is None else now + timeout,
                           loop.create_future())
        self._queue.append(request)
        self._counts["accepted"] += 1
        if request.deadline is not None:
            request.timer = loop.call_at(request.deadline, self._expire, request)
        self._wake.set()
        try:
            return await asyncio.shield(request.future)
        except asyncio.CancelledError:
            if not request.future.done():
                self._remove_queued(request)
                request.future.cancel()
                self._counts["cancelled"] += 1
            elif not request.future.cancelled():
                request.future.exception()
            raise
        finally:
            if request.timer is not None:
                request.timer.cancel()

    def _remove_queued(self, request: _Request) -> None:
        if not request.dispatched:
            self._queue.remove(request)
            self._wake.set()

    def _expire(self, request: _Request) -> None:
        if not request.future.done():
            self._remove_queued(request)
            self._finish(request, "expired", DeadlineExceeded("request deadline elapsed"))

    def _finish(self, request: _Request, status: str, value: Tensor | Exception) -> None:
        if request.future.done():
            return
        self._counts[status] += 1
        if isinstance(value, Exception):
            request.future.set_exception(value)
        else:
            request.future.set_result(value)

    def _take_batch(self, now: float) -> tuple[int, float | None]:
        """Return a ready FIFO prefix, or the next dispatch time."""
        first = self._queue[0]
        ready_at = first.enqueued + self.config.max_wait
        width, count = 0, 0
        for request in self._queue:
            width = max(width, request.cost)
            if width * (count + 1) > self.config.max_batch_cost:
                return count, None
            count += 1
            if request.deadline is not None and request.deadline <= ready_at:
                ready_at = now
            if count == self.config.max_batch_size or width * count == self.config.max_batch_cost:
                return count, None
        if self._closing or ready_at <= now:
            return count, None
        return 0, ready_at

    @torch.inference_mode()
    def _run_batch(self, payloads: list[Tensor]) -> list[Tensor]:
        outputs = self.adapter.run(payloads)
        if (not isinstance(outputs, list) or len(outputs) != len(payloads)
                or any(not isinstance(output, Tensor) for output in outputs)):
            raise ValueError("adapter.run must return one tensor per request")
        return outputs

    async def _serve(self) -> None:
        assert self._loop is not None and self._executor is not None
        loop = self._loop
        try:
            while True:
                self._wake.clear()
                now = loop.time()
                # Timers and worker callbacks can both be ready in the same
                # turn. Do not dispatch work that has already expired.
                for request in tuple(self._queue):
                    if request.deadline is not None and now >= request.deadline:
                        self._expire(request)
                if not self._queue:
                    if self._closing:
                        return
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                count, ready_at = self._take_batch(now)
                if not count:
                    assert ready_at is not None
                    self._wake.clear()
                    alarm = loop.call_at(ready_at, self._wake.set)
                    try:
                        await self._wake.wait()
                    finally:
                        alarm.cancel()
                    continue

                self._active = [self._queue.popleft() for _ in range(count)]
                for request in self._active:
                    request.dispatched = True
                self._counts["batches"] += 1
                self._counts["batch_items"] += count
                error = None
                try:
                    results = await loop.run_in_executor(
                        self._executor, self._run_batch, [r.payload for r in self._active]
                    )
                except Exception as exc:
                    error = exc
                now = loop.time()
                for i, request in enumerate(self._active):
                    if request.deadline is not None and now >= request.deadline:
                        self._expire(request)
                    elif error is not None:
                        failure = BatchExecutionError("model batch failed")
                        failure.__cause__ = error
                        self._finish(request, "failed", failure)
                    else:
                        self._finish(request, "completed", results[i])
                self._active.clear()
        finally:
            self._closing = True
            while self._queue:
                self._finish(self._queue.popleft(), "aborted", EngineClosed("worker stopped"))
            for request in self._active:
                self._finish(request, "aborted", EngineClosed("worker stopped"))
            self._active.clear()
            self._executor.shutdown(wait=False)
