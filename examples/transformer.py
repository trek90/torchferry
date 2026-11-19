"""Compare dynamically batched results against individual CPU inference."""

import asyncio

import torch
from torch import nn

from torchferry import EngineConfig, InferenceEngine, PaddedTokensAdapter


class MeanEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.layer = nn.TransformerEncoderLayer(16, 4, 32, batch_first=True)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.layer(self.embed(tokens), src_key_padding_mask=~valid)
        return (x * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)


async def main() -> None:
    torch.manual_seed(8)
    torch.set_num_threads(1)
    model = MeanEncoder().eval()
    requests = [torch.arange(n) for n in (3, 8, 1, 5, 2, 8, 4, 2, 6)]
    config = EngineConfig(max_batch_size=4, max_batch_cost=24)

    async with InferenceEngine(PaddedTokensAdapter(model), config) as engine:
        outputs = await asyncio.gather(*(engine.infer(x, timeout=5) for x in requests))

    with torch.inference_mode():
        expected = [
            model(x[None, :], torch.ones((1, x.numel()), dtype=torch.bool))[0]
            for x in requests
        ]
    torch.testing.assert_close(torch.stack(outputs), torch.stack(expected))
    print(f"{len(outputs)} dynamically batched results match the reference.")


if __name__ == "__main__":
    asyncio.run(main())
