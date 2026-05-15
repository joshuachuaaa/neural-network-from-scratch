# Reader Report: Data and Persistence

## Scope

`src/neural_network_from_scratch/data.py`, `scripts/verify_mnist.py`, `data/mnist/**`.

## Component Purpose

Loads and validates MNIST IDX gzip files, converts labels to one-hot matrices, and verifies tracked dataset checksums.

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `src/neural_network_from_scratch/data.py` | reviewed | Image/label IDX readers and one-hot encoder. |
| `scripts/verify_mnist.py` | reviewed | SHA256 manifest verification. |
| `data/mnist/SHA256SUMS` | reviewed | Four dataset hashes. |
| `data/mnist/*.gz` | summarized | Binary MNIST data, checksums verified. |

## File-by-File Review

### `src/neural_network_from_scratch/data.py`

- Purpose: Load and validate IDX gzip data.
- Key symbols: `IMAGE_MAGIC`, `LABEL_MAGIC`, `load_mnist_images`, `load_mnist_labels`, `one_hot_encode`.
- Inputs: gzip paths; labels arrays.
- Outputs: normalized image arrays shaped `(count, 784)` as `float64`; label arrays as `uint8`; one-hot matrices.
- Internal dependencies: `settings.MNIST_IMAGE_ROWS`, `settings.MNIST_IMAGE_COLS`, `settings.IN_DIMS`, `settings.OUT_DIM`.
- External dependencies: `gzip`, `struct`, `numpy`.
- Side effects: File reads only.
- Error handling: Raises `ValueError` for invalid magic, dimensions, image/label count, and labels outside range.
- Tests: `tests/test_data_loading.py`.
- Risks/unknowns: Does not check label values when loading raw labels; one-hot validation catches training labels later, but evaluation/inference labels are not range-validated.
- Confidence: Confirmed.

### `scripts/verify_mnist.py`

- Purpose: Verify SHA256 checksums for tracked MNIST files.
- Key symbols: `PROJECT_ROOT`, `MANIFEST`, `sha256`, `main`.
- Inputs: `data/mnist/SHA256SUMS`, dataset files.
- Outputs: prints success or exits with failure details.
- Side effects: File reads only.
- Error handling: Checksum mismatch handled; missing files/manifest produce raw file exceptions.
- Tests: no direct tests.
- Risks/unknowns: Not exposed as console script.
- Confidence: Confirmed.

### `data/mnist/SHA256SUMS`

- Purpose: Integrity manifest.
- Content: four SHA256 entries for train/test image/label gzip files.
- Confidence: Confirmed.

### `data/mnist/*.gz`

- Purpose: Runtime dataset.
- Verification: `python scripts/verify_mnist.py` returned `MNIST checksums verified.`
- Confidence: Confirmed.

## Key Flows

- Training and CLI call loaders directly with fixed file names under `data/mnist/`.
- CLI repeatedly reloads/decompresses data per train/eval/inference action.

## Data Flow

Gzip IDX bytes -> NumPy arrays -> normalized image matrix / label vector -> one-hot labels for training.

## Risks and Unknowns

- Medium: Repeated dataset loading/decompression in CLI can make interactive workflows slower than necessary.
- Low: Dataset files are committed; this improves demo reliability but increases repository size.
- Low: `verify_mnist.py` lacks friendly missing-file handling.

## Summary for Orchestrator

Data loading is focused and well-tested. Improve by adding a small dataset cache/session object, label range validation at load time, and friendlier dataset verification failures.
