# Cross-Component Flows

## System-Level Mental Model

Confirmed: The system is a Python terminal neural-network lab. Data loaders produce MNIST arrays, core network classes train on those arrays, training helpers update model parameters, Rich renderers display state, and the CLI orchestrates interactive commands.

## Component Interaction Map

- `package-config-entrypoints` exposes `nnfs`, `nnfs-train`, and `nnfs-visual`.
- `data-and-persistence` feeds `training-visualization` and `interactive-cli-checkpoints`.
- `neural-network-core` is used by training, visualization, CLI, checkpoints, and tests.
- `training-visualization` provides helpers and renderables; `cli.py` imports `layer_update_norms()` from `visual_train.py`.
- `interactive-cli-checkpoints` controls user workflow and persists model weights.
- `tests` verify pieces across all components.

## Runtime Startup Order

`nnfs` -> `cli.main()` -> `run_app()` -> render menu -> user command -> build/load/train/eval/infer/save.

## Data Lifecycle

Tracked gzip files -> data loaders -> normalized arrays -> one-hot training labels -> network forward/backprop -> metrics/rendering -> optional checkpoint file.

## Control Flow

Control is mostly procedural. CLI command dispatch is a large `if/elif` chain in `run_app()`. Training loops are implemented separately in standard, visual, and CLI contexts.

## Error Propagation

- Data/core errors propagate in `train.py` and `visual_train.py`.
- CLI catches all exceptions in the menu loop and prints them.
- Checkpoint errors propagate to CLI catch block.

## Shared Models and Utilities

- `NeuralNetwork`, `Layer`, `LayerType`.
- `settings` constants.
- `iter_batches`, `calculate_loss`, `evaluate_accuracy`.
- `dashboard`, `probability_table`, `render_digit`.

## Hidden Coupling

- `backProp()` requires a previous `predict()` call because gradients depend on stored layer activations.
- CLI imports `layer_update_norms()` from `visual_train.py`, coupling the app to a command-entrypoint module.
- Multiple modules assume MNIST file names directly.

## Circular or Suspicious Dependencies

- No circular imports observed.
- Suspicious dependency: `cli.py` -> `visual_train.py` for `layer_update_norms()`.

## Multiple Sources of Truth

- Data file names repeated in `train.py`, `visual_train.py`, `cli.py`.
- Training loop duplicated in three places.
- Dependency declarations split between `pyproject.toml` and `requirements.txt`.
- Project name differs between README/repo direction and package metadata.

## Risks

- Training behavior divergence.
- Reproducibility gap from unseeded weight initialization.
- Checkpoint schema fragility.
- Weak empirical quality gates.

## Unknowns

- Desired target training accuracy and preferred project/package name.
