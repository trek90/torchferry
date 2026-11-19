"""Scheduler tests use real PyTorch and run each scenario on an isolated loop."""

import asyncio
import threading

import pytest
import torch

from torchferry.engine import EngineConfig, InferenceEngine
from torchferry.errors import (
    BatchExecutionError,
    DeadlineExceeded,
    EngineClosed,
    QueueFull,
    RequestTooLarge,
)


class RecordingAdapter:
    def __init__(self):
        self.calls = []
        self.owner = threading.get_ident()

    def cost(self, x):
        return x.numel()

    def run(self, items):
        assert threading.get_ident() != self.owner
        assert torch.is_inference_mode_enabled()
        self.calls.append([item.numel() for item in items])
        return [item.square() for item in items]


class GatedAdapter(RecordingAdapter):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.entered = asyncio.Event()
        self.release = threading.Event()

    def run(self, items):
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release.wait(3):
            raise RuntimeError("test did not release worker")
        return super().run(items)


def test_contiguous_fifo_batches_enforce_padded_cost():
    async def scenario():
        adapter = RecordingAdapter()
        cfg = EngineConfig(max_batch_size=3, max_batch_cost=9, max_wait=0.01)
        payloads = [torch.arange(n) + i for i, n in enumerate([2, 2, 3, 7, 1])]
        async with InferenceEngine(adapter, cfg) as engine:
            results = await asyncio.gather(*(engine.infer(x) for x in payloads))
        assert adapter.calls == [[2, 2, 3], [7], [1]]
        for x, y in zip(payloads, results):
            torch.testing.assert_close(y, x.square())

    asyncio.run(scenario())


def test_oversize_request_never_reaches_worker():
    async def scenario():
        adapter = RecordingAdapter()
        async with InferenceEngine(adapter, EngineConfig(max_batch_cost=3)) as engine:
            with pytest.raises(RequestTooLarge):
                await engine.infer(torch.ones(4))
        assert adapter.calls == []

    asyncio.run(scenario())


def test_bounded_pending_while_worker_is_busy():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        cfg = EngineConfig(max_pending=2, max_batch_size=1)
        async with InferenceEngine(adapter, cfg) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                second = asyncio.create_task(engine.infer(torch.tensor([2])))
                await asyncio.sleep(0)
                with pytest.raises(QueueFull):
                    await engine.infer(torch.tensor([3]))
            finally:
                adapter.release.set()
            assert (await first).item() == 1
            assert (await second).item() == 4

    asyncio.run(scenario())


def test_expired_queued_request_is_not_executed():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        async with InferenceEngine(adapter, EngineConfig(max_batch_size=1)) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                with pytest.raises(DeadlineExceeded):
                    await engine.infer(torch.tensor([2, 3]), timeout=0.03)
            finally:
                adapter.release.set()
            await first
        assert adapter.calls == [[1]]

    asyncio.run(scenario())


def test_cancelling_queued_request_skips_model_work():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        async with InferenceEngine(adapter, EngineConfig(max_batch_size=1)) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                queued = asyncio.create_task(engine.infer(torch.tensor([2, 3])))
                await asyncio.sleep(0)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
            finally:
                adapter.release.set()
            await first
        assert adapter.calls == [[1]]

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed", [False, True])
def test_failed_batch_does_not_poison_later_calls(malformed):
    class FlakyAdapter(RecordingAdapter):
        first = True

        def run(self, items):
            if self.first:
                self.first = False
                if malformed:
                    return []
                raise ValueError("model failed")
            return super().run(items)

    async def scenario():
        async with InferenceEngine(FlakyAdapter()) as engine:
            with pytest.raises(BatchExecutionError):
                await engine.infer(torch.tensor([2]))
            assert (await engine.infer(torch.tensor([3]))).item() == 9

    asyncio.run(scenario())


def test_drain_completes_accepted_requests_and_closes_admission():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        async with InferenceEngine(adapter, EngineConfig(max_batch_size=1)) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                second = asyncio.create_task(engine.infer(torch.tensor([2])))
                await asyncio.sleep(0)
                closing = asyncio.create_task(engine.close())
                await asyncio.sleep(0)
                with pytest.raises(EngineClosed):
                    await engine.infer(torch.tensor([3]))
            finally:
                adapter.release.set()
            assert (await first).item() == 1
            assert (await second).item() == 4
            await closing

    asyncio.run(scenario())


def test_close_without_drain_prevents_queued_model_work():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        async with InferenceEngine(adapter, EngineConfig(max_batch_size=1)) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                second = asyncio.create_task(engine.infer(torch.tensor([2, 3])))
                await asyncio.sleep(0)
                closing = asyncio.create_task(engine.close(drain=False))
                await asyncio.sleep(0)
            finally:
                adapter.release.set()
            await closing
            results = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(results[1], EngineClosed)
        assert adapter.calls == [[1]]

    asyncio.run(scenario())


def test_deadline_is_observed_while_model_is_running():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        cfg = EngineConfig(max_batch_size=1)
        async with InferenceEngine(adapter, cfg) as engine:
            pending = asyncio.create_task(engine.infer(torch.tensor([4]), timeout=0.25))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                with pytest.raises(DeadlineExceeded):
                    await pending
                assert not adapter.release.is_set()
            finally:
                adapter.release.set()
            assert (await engine.infer(torch.tensor([5]))).item() == 25
        assert adapter.calls == [[1], [1]]

    asyncio.run(scenario())


def test_cancellation_releases_queued_capacity():
    async def scenario():
        adapter = GatedAdapter(asyncio.get_running_loop())
        cfg = EngineConfig(max_pending=2, max_batch_size=1)
        async with InferenceEngine(adapter, cfg) as engine:
            first = asyncio.create_task(engine.infer(torch.tensor([1])))
            try:
                await asyncio.wait_for(adapter.entered.wait(), 2)
                dropped = asyncio.create_task(engine.infer(torch.ones(2)))
                await asyncio.sleep(0)
                dropped.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await dropped
                third = asyncio.create_task(engine.infer(torch.tensor([3])))
                await asyncio.sleep(0)
            finally:
                adapter.release.set()
            assert (await first).item() == 1
            assert (await third).item() == 9
        assert adapter.calls == [[1], [1]]

    asyncio.run(scenario())


def test_short_deadline_flushes_before_batch_window():
    async def scenario():
        adapter = RecordingAdapter()
        cfg = EngineConfig(max_wait=1.0)
        async with InferenceEngine(adapter, cfg) as engine:
            result = await engine.infer(torch.tensor([4]), timeout=0.2)
            assert result.item() == 16
        assert adapter.calls == [[1]]

    asyncio.run(scenario())
