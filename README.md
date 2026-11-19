# TorchFerry

TorchFerry runs PyTorch inference for concurrent asyncio callers. It collects
requests into batches, bounds outstanding work, and runs model calls on one
dedicated thread so the event loop stays responsive.

```python
import asyncio
import torch
from torchferry import EngineConfig, InferenceEngine, StackAdapter

async def main():
    model = torch.nn.Linear(16, 4).eval()
    adapter = StackAdapter(model)
    config = EngineConfig(max_batch_size=8, max_batch_cost=128)
    async with InferenceEngine(adapter, config) as engine:
        outputs = await asyncio.gather(
            *(engine.infer(torch.ones(16) * i, timeout=1.0) for i in range(12))
        )
        print(torch.stack(outputs).shape)  # torch.Size([12, 4])

asyncio.run(main())
```

## Scheduling

- FIFO admission. A batch closes when it reaches the row cap, when the next
  request would exceed the cost cap, when the oldest request has waited
  `max_wait` seconds, or when a deadline makes waiting that long unsafe. There is no reordering around an expensive request.
- A batch's cost is its largest input cost multiplied by its row count. For
  token sequences this is the number of padded slots, not bytes, attention
  work, or KV-cache usage.
- Admission is fail-fast under overload (`QueueFull`), not an unbounded wait.
  Requests that cannot fit even alone raise `RequestTooLarge`.
- Timeouts include both queueing and model execution. Cancellation or timeout
  does not preempt a PyTorch call already executing; its result is discarded.
- Use `async with` or explicitly start and close the single-use engine.
  Draining stops admission and lets accepted requests finish. Non-draining
  close fails queued and in-flight callers but still waits for an active forward.

## Adapters

`StackAdapter` stacks same-shaped inputs. `PaddedTokensAdapter` right-pads
token sequences and passes a boolean mask (True means valid). A custom adapter
provides a cheap, synchronous `cost(tensor)` and a blocking
`run(list_of_tensors)` that returns one tensor per input, in order.

Put the model in eval mode and move it and its inputs to the desired device
before submitting. Inputs are borrowed: do not mutate them until their model
call physically finishes, even if the request times out or is cancelled. Forward
runs under `torch.inference_mode()`. TorchFerry does not provide HTTP serving,
multi-GPU sharding, streaming decode, or hard cancellation of GPU kernels.

## Metrics

`engine.metrics` is an immutable snapshot. It reports accepted, completed,
rejected, expired, cancelled, failed and aborted requests; submitted batches
and batch items; and physical queue and in-flight sizes. An in-flight request
continues to occupy capacity even after its caller stops waiting. Metrics are
local to one engine; read them from its event loop. Asynchronous accelerator
work is not forcibly synchronized, so timings and deadlines reflect host-side
completion unless the model adapter synchronizes.

## Run locally

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python examples/transformer.py
```
