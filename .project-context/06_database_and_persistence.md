# Database and Persistence Review

## Summary

Confirmed: There is no database, ORM, migration system, SQL, table schema, queue, or server-side persistence. Persistence is file-based: tracked MNIST data, README images, checksum manifest, and optional local model checkpoints.

## Persistence Mechanisms

- Tracked dataset files under `data/mnist/`.
- Tracked README screenshots under `docs/assets/`.
- Optional local checkpoints under prompt-selected paths, defaulting to `artifacts/model.npz`.
- Ignored local checkpoint artifact observed: `artifacts/menu-test.npz`.

## Data Models / Schemas

- MNIST IDX image format: magic `2051`, count, rows, cols, raw pixel bytes.
- MNIST IDX label format: magic `2049`, count, label bytes.
- Checkpoint NPZ implicit schema:
  - `hidden_layer_dims`
  - `weights_0..n`
  - `biases_0..n`

## Read Paths

- `data.py` reads MNIST gzip files.
- `scripts/verify_mnist.py` reads checksum manifest and dataset files.
- `checkpoints.py` reads NPZ checkpoints.
- README images are read by GitHub/docs tooling.

## Write Paths

| Write Target | File | Function | Trigger | Data Written | Failure Handling | Tests |
|---|---|---|---|---|---|---|
| Checkpoint NPZ | `checkpoints.py` | `save_checkpoint()` | CLI action 8 or direct call | Architecture and trainable layer weights/biases | Raw file/NumPy exceptions | Happy-path round trip in `tests/test_checkpoints.py` |
| Temporary gzip fixtures | `tests/test_data_loading.py` | test functions | pytest | Test IDX gzip files under `tmp_path` | pytest failures | Data loading tests |

## Destructive Operations

- No deletes, SQL destructive operations, or truncates found.
- Checkpoint save can overwrite an existing file path without helper-level confirmation.

## File-Based Persistence

Confirmed and intentional.

## Risks

- Medium: Checkpoint schema is implicit and unversioned.
- Medium: Checkpoint path extension behavior can be confusing if path lacks `.npz`.
- Low: Dataset committed to Git increases repo size but improves demo portability.

## Unknowns

- Unknown: Whether checkpoints should include metadata such as training epochs, metrics, timestamp, or data hash.
