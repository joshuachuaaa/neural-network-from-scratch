# Reader Report: Training and Visualization

## Scope

`src/neural_network_from_scratch/train.py`, `src/neural_network_from_scratch/visual_train.py`, `src/neural_network_from_scratch/rendering.py`.

## Component Purpose

Provides standard training/evaluation helpers and the Rich live visualization dashboard.

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `train.py` | reviewed | Batch iteration, loss, epoch training, eval, standard main. |
| `visual_train.py` | reviewed | Argparse/prompt visual training flow. |
| `rendering.py` | reviewed | Rich renderables for digits, probabilities, layers, metrics, dashboard. |

## File-by-File Review

### `train.py`

- Purpose: Standard non-interactive training.
- Key symbols: `DEFAULT_DATA_DIR`, `iter_batches`, `calculate_loss`, `train_epoch`, `evaluate_accuracy`, `main`.
- Inputs: NumPy features/targets, data directory, settings.
- Outputs: Printed metrics and final accuracy.
- Dependencies: `data`, `settings`, `NeuralNetwork`.
- Side effects: Reads dataset, prints to stdout, mutates network parameters.
- Tests: `tests/test_training.py`.
- Risks/unknowns: No direct test for `train_epoch()` improving loss; `main()` not smoke-tested; no CLI args for data path/hyperparameters.
- Confidence: Confirmed.

### `visual_train.py`

- Purpose: Live training dashboard entrypoint.
- Key symbols: `parse_args`, `validate_args`, `apply_interactive_prompts`, `layer_update_norms`, `main`.
- Inputs: CLI args/prompts, MNIST data.
- Outputs: Rich live dashboard.
- Dependencies: Rich, data loaders, `NeuralNetwork`, rendering, `calculate_loss`, `iter_batches`.
- Side effects: Reads dataset, mutates network, terminal rendering.
- Tests: `tests/test_rendering.py` covers `layer_update_norms`; command smoke run passed.
- Risks/unknowns: Duplicates training logic from `train_epoch()` and `cli.train_current_model()`; default `sample_index=0` always watches the test label 7.
- Confidence: Confirmed.

### `rendering.py`

- Purpose: Builds Rich renderables.
- Key symbols: `VisualMetrics`, `render_digit`, `_bar`, `sparkline`, `layer_table`, `probability_table`, `metrics_panel`, `dashboard`.
- Inputs: network, sample image, probabilities, metrics, update norms.
- Outputs: Rich `Text`, `Table`, `Panel`, `Group`.
- Side effects: none beyond renderable construction.
- Tests: `tests/test_rendering.py`.
- Risks/unknowns: Rendering compresses values for terminal width; this is appropriate but can hide precision.
- Confidence: Confirmed.

## Key Flows

- Standard training: `train.main()` -> load data -> one-hot labels -> `NeuralNetwork()` -> repeated `train_epoch()` -> `evaluate_accuracy()`.
- Visual training: parse args/prompts -> load limited data -> train loop -> `dashboard()` updates.

## Data Flow

MNIST arrays -> batches -> predictions -> gradients -> updates -> metrics -> Rich renderables.

## Component API / Boundaries

- `train.py` exposes reusable `iter_batches`, `calculate_loss`, `train_epoch`, `evaluate_accuracy`.
- `visual_train.py` reuses only some helpers, not `train_epoch`, because it needs per-batch visualization data.

## Duplication Observed

- Training-loop body is duplicated in `train.py`, `visual_train.py`, and `cli.py`.
- Data loading paths are repeated in `train.py`, `visual_train.py`, and `cli.py`.

## Risks and Unknowns

- High: Training behavior can diverge between standard, visual, and CLI paths due to duplicated loops.
- Medium: No empirical training quality gate.
- Medium: Default visual training uses tiny data subsets, which is good for demos but can make users think the model is broken.

## Summary for Orchestrator

Training/visualization works, but needs a shared training-session/event abstraction so CLI and visual dashboards observe the same canonical training loop.
