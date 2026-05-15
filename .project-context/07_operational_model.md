# Operational Model

## Build

- Build backend: setuptools (`pyproject.toml`).
- Editable install: `python -m pip install -e ".[dev]"`.
- No wheel/build scripts beyond standard Python packaging.

## Run

- Interactive app: `nnfs` or `PYTHONPATH=src python -m neural_network_from_scratch.cli`.
- Standard train: `nnfs-train` or `python main.py`.
- Visual train: `nnfs-visual --interactive`.
- Dataset verification: `python scripts/verify_mnist.py`.

## Test

- `pytest`
- Scan result: `26 passed`.

## Deploy

No deployment process found. This is a local CLI/package project.

## Runtime Processes

Single foreground Python process for each command.

## Background Jobs

None found.

## Logging

- No logging module usage.
- Standard training prints epoch/test metrics.
- CLI and visual app render through Rich.

## Metrics / Health Checks

- Training loss/accuracy displayed.
- No external health checks.
- Dataset checksum script acts as a local integrity check.

## Debugging Tools

- Tests.
- Dataset verification script.
- Rich visual dashboard.

## Common Failure Modes

- Running from wrong directory can make relative `data/mnist` path fail.
- Missing/corrupt MNIST files.
- Terminal too narrow for ideal Rich rendering.
- Invalid/corrupt checkpoint file.
- User misunderstanding fixed watched MNIST sample as training label.

## Unknowns

- Unknown: Desired support for packaged data paths or user-specified global data directory.
