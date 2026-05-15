# Flow Trace: Startup Flow

## Summary

Confirmed: The project has three package console entrypoints plus two Python module/script entrypoints. The primary user-facing entrypoint is `nnfs`.

## Entry Points

- `nnfs` -> `neural_network_from_scratch.cli:main` (`pyproject.toml:16`).
- `nnfs-train` -> `neural_network_from_scratch.train:main` (`pyproject.toml:17`).
- `nnfs-visual` -> `neural_network_from_scratch.visual_train:main` (`pyproject.toml:18`).
- `python main.py` -> `train.main()` (`main.py:1-10`).
- `python -m neural_network_from_scratch` -> `train.main()` (`__main__.py:1-5`).

## Step-by-Step Flow

| Step | File | Function/Class | What Happens | Inputs | Outputs | Side Effects |
|---|---|---|---|---|---|---|
| 1 | `pyproject.toml` | `[project.scripts]` | Console script resolves to module function. | Installed package. | `nnfs`, `nnfs-train`, `nnfs-visual`. | None. |
| 2 | `cli.py` | `main()` | Calls `run_app()`. | None. | Interactive app. | Terminal I/O. |
| 3 | `cli.py` | `run_app()` | Creates `Console`, `AppState`, renders `main_screen()`, prompts action. | Terminal input. | Menu loop. | Clears terminal and dispatches commands. |
| 4 | `train.py` | `main()` | Loads data, trains for settings epochs, evaluates. | Data files, settings. | Printed metrics. | Reads dataset, mutates model. |
| 5 | `visual_train.py` | `main()` | Parses args, loads limited data, renders live training dashboard. | Args/prompts, data files. | Live Rich dashboard. | Reads dataset, mutates model. |

## Error Handling

- CLI catches broad `Exception` in `run_app()` and prints an error message.
- `visual_train.validate_args()` raises `SystemExit` for invalid numeric args.
- `train.main()` lets file/data errors propagate.

## Tests

- CLI launch smoke command passed during scan.
- Visual training smoke command passed during scan.
- No direct tests for `train.main()` or console-script packaging.

## Risks

- `main.py` uses `sys.path.insert()` and is less clean than package entrypoints.
- Startup paths lead into different training implementations.

## Unknowns

- Unverified: Desired long-term support for `main.py`.

## Confidence

Confirmed for static entrypoint mapping and smoke command results.
