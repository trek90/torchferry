# Verification

Built and installed the TorchFerry 0.1.0 wheel. Ran the suite from outside
the source tree against the installed package.

- Python 3.12, PyTorch 2.8.0+cpu on Linux: 30 tests passed.
- Transformer example: nine variable-length inputs matched one-at-a-time inference.
- CUDA was unavailable and not tested.
