# Repository Map

## High-Level Summary

Confirmed: This repository is a Python package for a from-scratch NumPy neural network with Rich-based terminal visualization. Evidence: `README.md:5`, `pyproject.toml:10-18`, and source modules under `src/neural_network_from_scratch/`.

Confirmed: There is no web server, database server, Docker config, CI config, or external network integration in tracked files. Evidence: `git ls-files`, `pyproject.toml`, and repository searches for HTTP/database/queue patterns.

## Top-Level Structure

| Path | Purpose | Runtime Relevance | Build/Test Relevance | Needs Deep Review? | Notes |
|---|---|---|---|---|---|
| `.gitignore` | Ignore generated/cache/local outputs. | Indirect. | Important for repo hygiene. | Yes | Ignores `.project-context/`, artifacts, egg-info. |
| `README.md` | Landing page and usage docs. | No direct runtime effect. | Developer onboarding. | Yes | Good overview, but naming still differs from package metadata. |
| `data/mnist/` | Tracked MNIST gzip data and checksum manifest. | Required by default training, CLI, and visualization flows. | Test-independent; runtime verification via script. | Summarize binaries, review manifest. | Dataset files are committed; clone size tradeoff is acceptable for demo but should be intentional. |
| `docs/assets/` | README screenshots. | None. | Docs only. | Summarize binaries. | PNG screenshots tracked. |
| `scripts/` | Operational helper scripts. | `verify_mnist.py` validates dataset integrity. | Useful pre-run check. | Yes | Script does not handle missing manifest/path with friendly messages. |
| `src/neural_network_from_scratch/` | Application/package source. | Primary runtime. | Primary test target. | Yes | Core, CLI, training, visualization all live here. |
| `tests/` | Unit and smoke tests. | None. | Primary quality gate. | Yes | 26 tests pass; coverage lacks real training convergence checks. |
| `main.py` | Backwards-compatible script entrypoint. | Runs training via `train.main()`. | Not directly tested. | Yes | Uses `sys.path.insert`, less clean than package scripts. |
| `pyproject.toml` | Build metadata, deps, console scripts, pytest config. | Defines installable commands. | Defines pytest behavior. | Yes | No lint/type/CI tooling configured. |
| `requirements.txt` | Alternate dependency install path. | Install support. | Includes `pytest` as regular dependency. | Yes | Duplicates dependency info from `pyproject.toml`. |

## Important Documentation

- `README.md` states the project purpose, quick start, commands, terminal lab behavior, data expectations, and project structure.
- No separate architecture docs, setup docs, CI docs, Docker docs, or API docs were found.

## Important Config Files

- `pyproject.toml`: package name `neural-network-from-scratch`, dependencies `numpy` and `rich`, console scripts `nnfs`, `nnfs-train`, `nnfs-visual`, dev extra `pytest`, pytest path config.
- `requirements.txt`: lists `numpy`, `pytest`, `rich`.
- `.gitignore`: ignores generated Python files, caches, virtualenvs, `.env`, `.project-context/`, `artifacts/`, and `*.egg-info/`.

## Important Entrypoints

- `nnfs` -> `neural_network_from_scratch.cli:main` (`pyproject.toml:16`).
- `nnfs-train` -> `neural_network_from_scratch.train:main` (`pyproject.toml:17`).
- `nnfs-visual` -> `neural_network_from_scratch.visual_train:main` (`pyproject.toml:18`).
- `python main.py` -> inserts `src` into `sys.path`, then runs `train.main()` (`main.py:1-10`).
- `python -m neural_network_from_scratch` -> runs `train.main()` (`__main__.py:1-5`).

## Initial Observations

- Confirmed: The CLI is the largest module at 420 lines and combines app state, prompts, data loading, training loop, rendering composition, inference, eval, checkpointing, and command dispatch.
- Confirmed: Training loop logic appears in `train.py`, `cli.py`, and `visual_train.py`, with duplicated forward/backprop/update/metric logic.
- Confirmed: Runtime file writes are limited to checkpoint files via `save_checkpoint()` and generated caches outside application code.
- Confirmed: Tests pass quickly and verify many unit-level behaviors.

## Initial Unknowns

- Unverified: Expected target accuracy and training hyperparameters for “good enough” project behavior.
- Unverified: Whether MNIST data should remain committed or be downloaded on demand.
- Unknown: Whether the package/project should be renamed to match the moved repository `neuralnet-terminal-visualizer`.
