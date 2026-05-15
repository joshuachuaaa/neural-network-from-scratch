# Reader Report: Interactive CLI and Checkpoints

## Scope

`src/neural_network_from_scratch/cli.py`, `src/neural_network_from_scratch/checkpoints.py`.

## Component Purpose

Provides the polished Rich menu app and checkpoint save/load persistence.

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `cli.py` | reviewed | Interactive menu, model editing, train/eval/infer, checkpoint commands. |
| `checkpoints.py` | reviewed | NPZ checkpoint persistence. |

## File-by-File Review

### `cli.py`

- Purpose: Main interactive terminal app.
- Key symbols: `TrainingPoint`, `AppState`, `architecture_summary`, `parameter_count`, `architecture_table`, `status_table`, `menu_table`, `architecture_visual`, `training_history_panel`, `main_screen`, `reset_run_state`, `add_hidden_layer`, `change_hidden_layer`, `remove_hidden_layer`, `ensure_network`, `load_data`, `train_current_model`, `run_inference`, `evaluate_current_model`, `save_current_checkpoint`, `load_model_checkpoint`, `run_app`, `main`.
- Inputs: Rich prompts, MNIST data files, optional checkpoint path.
- Outputs: Rich terminal UI, saved checkpoint files.
- Internal dependencies: settings, checkpoints, data, network, rendering, training helpers, `layer_update_norms`.
- Side effects: Reads dataset, mutates app state and network, writes checkpoint files, clears console.
- Config/env: Uses `DEFAULT_DATA_DIR`, settings constants; no environment variables.
- Persistence: `save_current_checkpoint()` writes to `artifacts/model.npz` by default through `save_checkpoint()`.
- External I/O: File system reads/writes only.
- Error handling: Top-level loop catches all `Exception` and prints message; individual prompts validate many numeric inputs.
- Tests: `tests/test_cli.py` covers renderable helpers and parameter counts, not full prompt command flows.
- Risks/unknowns: Large mixed-responsibility module; duplicated training loop; default watched sample is fixed; data is reloaded for each action; `Confirm` import is unused.
- Confidence: Confirmed.

### `checkpoints.py`

- Purpose: Save/load network architecture and parameters.
- Key symbols: `save_checkpoint`, `load_checkpoint`.
- Inputs: `NeuralNetwork`, filesystem path.
- Outputs: `.npz` file and loaded `NeuralNetwork`.
- Dependencies: NumPy, `LayerType`, `NeuralNetwork`.
- Side effects: Creates parent dirs, writes NPZ, reads NPZ.
- Error handling: Relies on NumPy/file exceptions and key lookups; no custom validation.
- Tests: `tests/test_checkpoints.py` happy path.
- Risks/unknowns: No checkpoint version/schema; no shape validation; no explicit `.npz` extension normalization; no metadata for training epoch/history; no corrupted checkpoint tests.
- Confidence: Confirmed.

## Key Flows

- Menu flow: `run_app()` -> render `main_screen()` -> prompt action -> dispatch command -> catch exceptions.
- Training flow: `train_current_model()` -> ensure/build network -> prompt hyperparameters -> load data -> train per batch -> update Rich dashboard -> update session state.
- Checkpoint flow: `save_current_checkpoint()` -> `save_checkpoint()`; `load_model_checkpoint()` -> `load_checkpoint()` -> reset run state.

## Data Flow

AppState hidden layer list -> `NeuralNetwork` -> training updates -> metrics/history -> checkpoint file.

## Component API / Boundaries

- `cli.py` is both UI controller and training orchestrator.
- `checkpoints.py` is a clean separate module but lacks validation abstraction.

## Duplication Observed

- CLI data loading duplicates `visual_train.py`/`train.py` path logic.
- CLI training loop duplicates `visual_train.py` training loop and partially duplicates `train_epoch()`.
- Architecture/parameter helpers in CLI may eventually belong in model/config utilities.

## Risks and Unknowns

- High: CLI training can drift from non-interactive training behavior because it has a copied training loop.
- Medium: Checkpoint load can fail with raw errors or silently accept semantically wrong files if shapes happen to fit.
- Medium: Reloading full data on every interaction is inefficient.

## Summary for Orchestrator

The CLI is the product surface and works, but it should be split into state, commands, menus/rendering, and training-session orchestration. Checkpoints should gain schema validation.
