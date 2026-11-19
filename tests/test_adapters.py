import pytest
import torch

from torchferry.adapters import PaddedTokensAdapter, StackAdapter


def test_stack_forward_and_output_storage():
    model = torch.nn.Linear(2, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))
        inputs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        outputs = StackAdapter(model).run(inputs)
    assert [row.tolist() for row in outputs] == [[8.0], [18.0]]
    assert outputs[0].untyped_storage().data_ptr() != outputs[1].untyped_storage().data_ptr()


@pytest.mark.parametrize("items", [
    [],
    [torch.ones(2), torch.ones(3)],
    [torch.ones(2), torch.ones(2, dtype=torch.int64)],
    [torch.empty(0)],
    [torch.ones(1, device="meta")],
    [torch.ones(1).to_sparse()],
])
def test_invalid_stack_input(items):
    with pytest.raises(ValueError):
        StackAdapter(lambda x: x).run(items)


def test_padded_tokens_mask_real_pad_id():
    pad = 2**63 - 1

    def forward(ids, valid):
        assert ids.tolist() == [[1, pad], [pad, pad]]
        assert valid.tolist() == [[True, True], [True, False]]
        assert valid.dtype == torch.bool
        return valid.sum(dim=1)

    adapter = PaddedTokensAdapter(forward, pad_id=pad)
    inputs = [torch.tensor([1, pad]), torch.tensor([pad])]
    assert [adapter.cost(item) for item in inputs] == [2, 1]
    assert [row.item() for row in adapter.run(inputs)] == [2, 1]


@pytest.mark.parametrize("payload", [torch.tensor([1.0]), torch.tensor([[1]]), torch.tensor([], dtype=torch.long)])
def test_invalid_token_input(payload):
    adapter = PaddedTokensAdapter(lambda x, mask: x)
    with pytest.raises(ValueError):
        adapter.run([payload])


@pytest.mark.parametrize("pad_id", [True, 2**63, -(2**63)-1, 0.5])
def test_invalid_pad_id(pad_id):
    with pytest.raises(ValueError, match="int64"):
        PaddedTokensAdapter(lambda x, mask: x, pad_id=pad_id)


@pytest.mark.parametrize("wrong", [None, torch.tensor(1), torch.ones(2)])
def test_adapter_rejects_non_batch_first_outputs(wrong):
    for adapter in [StackAdapter(lambda *_: wrong), PaddedTokensAdapter(lambda *_: wrong)]:
        with pytest.raises(ValueError, match="one leading row"):
            adapter.run([torch.tensor([1])])
