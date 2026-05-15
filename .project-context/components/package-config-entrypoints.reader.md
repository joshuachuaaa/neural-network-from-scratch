# Reader Report: Package Config and Entrypoints

## Scope

`.gitignore`, `pyproject.toml`, `requirements.txt`, `main.py`, `src/neural_network_from_scratch/__init__.py`, `src/neural_network_from_scratch/__main__.py`, `src/neural_network_from_scratch/settings.py`.

## Component Purpose

Defines how the project is installed, run, tested, ignored, and configured by default.

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `.gitignore` | reviewed | Ignores caches, env files, reports, artifacts, egg-info. |
| `pyproject.toml` | reviewed | Package metadata, dependencies, scripts, dev extra, pytest config. |
| `requirements.txt` | reviewed | Alternate dependency list. |
| `main.py` | reviewed | Legacy script wrapper. |
| `src/neural_network_from_scratch/__init__.py` | reviewed | Exports `NeuralNetwork`. |
| `src/neural_network_from_scratch/__main__.py` | reviewed | Module entrypoint. |
| `src/neural_network_from_scratch/settings.py` | reviewed | Constants for model dimensions and training defaults. |

## File-by-File Review

### `.gitignore`

- Purpose: Keeps generated files out of Git.
- Key entries: `__pycache__/`, `.pytest_cache/`, `.project-context/`, `artifacts/`, `*.egg-info/`.
- Risks/unknowns: No major issue; local checkpoint `artifacts/menu-test.npz` is ignored as intended.
- Confidence: Confirmed.

### `pyproject.toml`

- Purpose: Build metadata and console scripts.
- Key config: package `neural-network-from-scratch`, Python `>=3.10`, dependencies `numpy`, `rich`, scripts `nnfs`, `nnfs-train`, `nnfs-visual`, dev extra `pytest`, pytest `pythonpath = ["src"]`.
- Risks/unknowns: No lint/type tooling; repo remote now uses `neuralnet-terminal-visualizer` while package remains `neural-network-from-scratch`.
- Confidence: Confirmed.

### `requirements.txt`

- Purpose: Alternate non-editable install dependencies.
- Key contents: `numpy`, `pytest`, `rich`.
- Risks/unknowns: `pytest` is listed as a base requirement here but only a dev extra in `pyproject.toml`.
- Confidence: Confirmed.

### `main.py`

- Purpose: Backwards-compatible script entrypoint.
- Key symbols: imports `Path`, mutates `sys.path`, imports `train.main`, runs it under `__main__`.
- Risks/unknowns: `sys.path.insert()` is a compatibility hack; package scripts are cleaner.
- Confidence: Confirmed.

### `__init__.py`

- Purpose: Package export.
- Key symbols: `NeuralNetwork`, `__all__`.
- Confidence: Confirmed.

### `__main__.py`

- Purpose: `python -m neural_network_from_scratch` entrypoint.
- Key symbols: delegates to `train.main()`.
- Confidence: Confirmed.

### `settings.py`

- Purpose: Global constants.
- Key config: `MNIST_IMAGE_ROWS=28`, `MNIST_IMAGE_COLS=28`, `IN_DIMS=784`, `OUT_DIM=10`, `HIDDEN_LAYERS=4`, `HIDDEN_LAYER_DIM=128`, `LEARNING_RATE=0.001`, `BATCH_SIZE=64`, `EPOCHS=10`, `RANDOM_SEED=42`.
- Risks/unknowns: Defaults are global rather than passed through a typed config object; `RANDOM_SEED` controls shuffle order but not model initialization because `Layer` uses global `np.random.randn()`.
- Confidence: Confirmed.

## Key Flows

- Install exposes console commands through `pyproject.toml`.
- `main.py`, `__main__.py`, and `nnfs-train` all route to standard training.

## Component API / Boundaries

- Public entrypoints are console scripts and `NeuralNetwork` export.
- Config is global constants, not dependency-injected.

## Risks and Unknowns

- Medium: Project/repo/package naming mismatch can confuse users and packaging.
- Low: Dependency definitions are duplicated and inconsistent between `requirements.txt` and `pyproject.toml`.
- Low: No configured lint/type/format tool.

## Summary for Orchestrator

The project is installable and runnable, but packaging hygiene can improve with aligned naming, unified dependency management, optional lint/type tooling, and eventual removal or de-emphasis of `main.py`'s `sys.path` wrapper.
