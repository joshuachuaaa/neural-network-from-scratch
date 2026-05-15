# Flow Trace: Destructive Operations

## Summary

No database-destructive operations were found. File writes are limited to checkpoint creation and test temporary gzip files.

## Entry Points

- `checkpoints.save_checkpoint()`
- `tests/test_data_loading.py` temporary test file writes

## Step-by-Step Flow

| Step | File | Function/Class | What Happens | Inputs | Outputs | Side Effects |
|---|---|---|---|---|---|---|
| 1 | `checkpoints.py` | `save_checkpoint()` | Writes/overwrites checkpoint path via `np.savez()`. | Network, path. | NPZ file. | Creates parent dirs, writes file. |
| 2 | `tests/test_data_loading.py` | tests | Writes temporary gzip IDX fixtures under pytest `tmp_path`. | Test data. | Temp files. | Test-only writes. |

## Error Handling

- Checkpoint writes rely on NumPy/file exceptions.
- No explicit overwrite confirmation in checkpoint helper.

## Tests

- Checkpoint happy path covered.

## Risks

- Medium: Saving to an existing checkpoint path overwrites it without confirmation at helper level. The CLI asks for path but does not warn about overwrite.
- Low: No deletes/truncates/drop operations found.

## Unknowns

- Unknown: Desired checkpoint overwrite UX.

## Confidence

Confirmed by repository search.
