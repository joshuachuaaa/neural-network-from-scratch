# Flow Trace: Persistence Paths

## Summary

Confirmed: There is no database. Persistence is file-based: tracked MNIST data is read, checksum manifest is read, and checkpoints are saved/loaded as NumPy `.npz` files.

## Entry Points

- `cli.save_current_checkpoint()`
- `cli.load_model_checkpoint()`
- `checkpoints.save_checkpoint()`
- `checkpoints.load_checkpoint()`
- `scripts/verify_mnist.py`

## Step-by-Step Flow

| Step | File | Function/Class | What Happens | Inputs | Outputs | Side Effects |
|---|---|---|---|---|---|---|
| 1 | `cli.py` | `save_current_checkpoint()` | Prompts path, ensures network, calls save helper. | AppState, prompt path. | Path stored in state. | File write via helper. |
| 2 | `checkpoints.py` | `save_checkpoint()` | Creates parent dir, builds payload, calls `np.savez()`. | Network, path. | NPZ checkpoint. | Creates directories and writes file. |
| 3 | `cli.py` | `load_model_checkpoint()` | Prompts path, loads network, resets metrics/history. | Prompt path. | Loaded network in state. | File read. |
| 4 | `checkpoints.py` | `load_checkpoint()` | Reads `hidden_layer_dims`, reconstructs network, assigns weights/biases. | NPZ path. | `NeuralNetwork`. | File read. |
| 5 | `scripts/verify_mnist.py` | `main()` | Reads checksum manifest and dataset files. | `SHA256SUMS`, gzip files. | Success/failure. | File reads. |

## Error Handling

- Checkpoint errors are raw file/NumPy/key errors.
- Checksum mismatches produce `SystemExit` with messages.

## Tests

- `tests/test_checkpoints.py` covers happy-path checkpoint round trip.
- No tests for checkpoint corruption, shape mismatch, missing keys, or missing path.

## Risks

- Medium: Checkpoints lack schema version and shape validation.
- Medium: `np.savez()` appends `.npz` if given a path without extension, but `save_checkpoint()` returns the original `Path`; this can confuse users.
- Low: `artifacts/**` ignored, so user checkpoints are not accidentally committed.

## Unknowns

- Unknown: Whether training epoch/history should be part of checkpoints.

## Confidence

Confirmed.
