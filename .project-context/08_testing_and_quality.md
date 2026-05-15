# Testing and Quality Review

## Test Structure

Seven pytest files under `tests/` cover activations, data loading, network, training helpers, rendering, CLI renderable helpers, and checkpoints.

## Test Commands

- `pytest -q` -> `26 passed in 0.11s` during scan.
- `python -m compileall -q src tests scripts main.py` -> passed during scan.
- CLI launch smoke and visual training smoke commands passed during scan.

## Covered Areas

- Softmax/ReLU behavior.
- MNIST IDX image/label loading and some validation.
- One-hot label validation.
- Network topology and output shape.
- Gradient shape and one parameter update.
- Batching and evaluation.
- Rich renderable construction.
- Checkpoint happy-path round trip.

## Missing Coverage

- Deterministic seeded model initialization.
- Training convergence/loss decrease on a tiny deterministic sample.
- Numerical gradient checks.
- CLI command flows with mocked prompts.
- Checkpoint malformed/missing key/shape mismatch handling.
- Standard `train.main()` smoke test.
- Visual command argument validation tests.
- Missing dataset/checksum error behavior.

## Brittle Tests

- Some tests use manual try/except rather than `pytest.raises`.
- No snapshot-like tests for important terminal text content.

## Quality Tooling

- Configured: pytest only.
- Missing: formatter/linter/type checker/coverage/CI.

## Recommended Test Improvements

1. Add a tiny deterministic training acceptance test that verifies loss decreases or accuracy improves.
2. Add checkpoint failure-mode tests before changing persistence.
3. Add CLI command tests by extracting prompt-free command functions or mocking Rich prompts.
4. Add `ruff` and optionally `mypy`/`pyright` in `pyproject.toml`.
5. Add GitHub Actions for `pytest` and dataset checksum verification.

## Unknowns

- Unknown: Required accuracy target for full MNIST training.
