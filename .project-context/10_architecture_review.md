# Architecture Review

## Current Architecture Map

### What the project appears to do

Confirmed: A local Python/Rich terminal lab for building, training, visualizing, evaluating, and checkpointing a from-scratch NumPy MNIST classifier.

### Main runtime entrypoints

`nnfs`, `nnfs-train`, `nnfs-visual`, `python main.py`, `python -m neural_network_from_scratch`.

### Major components

- Core neural network.
- Data loading and verification.
- Training helpers.
- Rich rendering.
- Interactive CLI.
- Checkpoints.
- Tests/docs.

### How components communicate

Direct Python imports and in-memory NumPy arrays. No service boundaries.

### Important data models

- `NeuralNetwork`
- `Layer`
- `LayerType`
- `AppState`
- `TrainingPoint`
- `VisualMetrics`
- implicit checkpoint NPZ payload

### Important external dependencies

NumPy and Rich.

### Config/environment assumptions

Global `settings.py`; current working directory contains `data/mnist`; no env vars.

### Current deployment/runtime model

Local CLI package only.

## Component Boundary Review

Core math is reasonably separated. CLI boundary is too broad: `cli.py` mixes UI, app state, data loading, training loop, persistence, and command dispatch.

## Dependency Direction Review

Mostly sane. The main issue is `cli.py` importing `layer_update_norms()` from `visual_train.py`, a command module. That utility should live in a shared metrics/training module.

## Data Flow Review

Simple and inspectable. Repeated data loads and relative paths are the main issues.

## State and Persistence Review

In-memory state is held by `AppState`; persistence is checkpoint NPZ. Checkpoints store architecture and weights/biases but not run metadata.

## Error Handling and Observability Review

Observability is good for a terminal educational tool. Error handling is basic; CLI catches broad exceptions and prints them. Checkpoint and file errors could be friendlier.

## Testing Architecture Review

Fast unit tests exist and pass. Behavioral ML and CLI-flow tests are missing.

## Operational Architecture Review

Good for local dev. Missing CI, lockfile, and robust installed-package data path story.

## Scalability Review

This does not need service-scale architecture. Performance improvements should focus on data caching, `float32`, training-loop consolidation, and initialization.

## Maintainability Review

Maintainability is good for the core modules but weakening in CLI/training duplication. Refactor before adding more product features.

## Architectural Strengths

- Clear educational separation between data, core, training, rendering, CLI, checkpoints.
- Simple local runtime.
- Fast tests.
- Rich UI is a strong differentiator.

## Architectural Weaknesses

- Training loop duplicated.
- CLI module too large.
- Reproducibility gap.
- Checkpoint schema weak.
- No explicit model/config objects.

## Architectural Risks

| Issue | Evidence | Why It Matters | Severity | Recommendation |
|---|---|---|---:|---|
| Training loop duplication | `train.py`, `cli.py`, `visual_train.py` | Behavior and bug fixes can diverge. | High | Extract shared training event generator. |
| Non-seeded initialization | `layers.py:23` | Runs are not reproducible despite seed constant. | High | Pass RNG/seed through `NeuralNetwork` and `Layer`. |
| CLI mixed responsibilities | `cli.py` 420 lines | More features will make it harder to test/change. | Medium | Split state, commands, menus, training session. |
| Checkpoint fragility | `checkpoints.py` | Bad user files fail poorly and format cannot evolve safely. | Medium | Add schema version and validation. |
| Weak behavioral tests | tests | Refactors can preserve shapes while breaking learning. | Medium | Add convergence and CLI flow tests. |

## Open Architecture Questions

- Should `nnfs` become the only primary user entrypoint?
- Should checkpoints include training history and settings?
- Should project/package be renamed to `neuralnet-terminal-visualizer`?
