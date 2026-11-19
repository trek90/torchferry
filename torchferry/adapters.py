"""Tensor collation and per-request output extraction."""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class Adapter(Protocol):
    """The scheduler calls cost on its event loop and run on its worker.

    Cost must be fast and positive. Run receives a non-empty batch and must
    return one independent tensor per input, in the same order.
    """

    def cost(self, payload: Tensor) -> int: ...

    def run(self, payloads: list[Tensor]) -> list[Tensor]: ...


def _validate_tensor(payload: Tensor) -> int:
    if not isinstance(payload, Tensor) or payload.layout != torch.strided:
        raise ValueError("payload must be a strided tensor")
    if payload.is_nested or payload.is_quantized or payload.device.type == "meta":
        raise ValueError("nested, quantized, and meta tensors are not supported")
    if payload.numel() == 0:
        raise ValueError("payload must not be empty")
    return payload.numel()


def _split(output: Tensor, count: int) -> list[Tensor]:
    if not isinstance(output, Tensor) or not output.ndim or output.shape[0] != count:
        raise ValueError("forward must return a tensor with one leading row per input")
    return [row.clone() for row in output.unbind(0)]


class StackAdapter:
    """Stack equal-shaped tensors and call forward(batch).

    Each batch must have a common shape, dtype and device. Route incompatible
    requests to separate engines. The caller sets the model to eval mode.
    """

    def __init__(self, forward: Callable[[Tensor], Tensor]) -> None:
        self.forward = forward

    def cost(self, payload: Tensor) -> int:
        """Return the input element count, not bytes or model FLOPs."""
        return _validate_tensor(payload)

    def run(self, payloads: list[Tensor]) -> list[Tensor]:
        if not payloads:
            raise ValueError("batch must not be empty")
        prototype = payloads[0]
        for item in payloads:
            self.cost(item)
            if (item.shape, item.dtype, item.device) != (
                prototype.shape,
                prototype.dtype,
                prototype.device,
            ):
                raise ValueError("stack requests must have a common shape, dtype and device")
        return _split(self.forward(torch.stack(payloads)), len(payloads))


class PaddedTokensAdapter:
    """Right-pad token IDs and call forward(padded_ids, valid_mask).

    Inputs are 1D long tensors on a common device. True marks real tokens,
    including any real token equal to pad_id. The caller sets eval mode and
    trims any per-token outputs if needed. Neither adapter moves tensors or
    synchronizes accelerators: device and stream coordination remain with the
    caller. The token budget counts slots, not attention or KV-cache memory.
    """

    def __init__(
        self, forward: Callable[[Tensor, Tensor], Tensor], *, pad_id: int = 0
    ) -> None:
        if type(pad_id) is not int or not -(2**63) <= pad_id < 2**63:
            raise ValueError("pad_id must be an int64 integer")
        self.forward = forward
        self.pad_id = pad_id

    def cost(self, payload: Tensor) -> int:
        """Return the length of a non-empty token sequence."""
        length = _validate_tensor(payload)
        if payload.ndim != 1 or payload.dtype != torch.int64:
            raise ValueError("token payload must be a 1D int64 tensor")
        return length

    def run(self, payloads: list[Tensor]) -> list[Tensor]:
        if not payloads:
            raise ValueError("batch must not be empty")
        lengths = [self.cost(item) for item in payloads]
        device = payloads[0].device
        if any(item.device != device for item in payloads):
            raise ValueError("token requests must be on the same device")
        ids = pad_sequence(payloads, batch_first=True)
        mask = torch.arange(ids.shape[1], device=device)[None, :] < torch.tensor(
            lengths, device=device
        )[:, None]
        if self.pad_id:
            # pad_sequence takes a float padding value; preserve full int64 precision.
            ids.masked_fill_(~mask, self.pad_id)
        return _split(self.forward(ids, mask), len(payloads))
