# Configuration and Environment Review

## Summary

Confirmed: Configuration is code-based constants in `settings.py` plus package metadata in `pyproject.toml`. No environment variables, secrets, `.env.example`, Docker, CI, or deployment config were found.

## Config Files

- `src/neural_network_from_scratch/settings.py`: model dimensions and training defaults.
- `pyproject.toml`: package metadata, dependencies, console scripts, pytest config.
- `requirements.txt`: alternate dependency list.
- `.gitignore`: generated/local file exclusions.

## Environment Variables

| Name | Used In | Required? | Default | Purpose | Risk |
|---|---|---|---|---|---|
| N/A | N/A | No | N/A | No environment variables found in tracked code. | None. |

## Secrets

No secrets or secret-loading paths found.

## Environment-Specific Behavior

- Local/dev usage is documented through `python -m pip install -e ".[dev]"`.
- Runtime assumes commands are run from repository root or with package installed, because default data path is relative `data/mnist`.
- `main.py` manually inserts local `src` into `sys.path`.

## Risks

- Medium: Relative `DEFAULT_DATA_DIR = Path("data") / "mnist"` depends on working directory for installed package usage unless user runs from repo root.
- Low: No lint/type config.
- Low: Dependency declarations duplicated.

## Unknowns

- Unknown: Whether installed command should locate packaged data, user data dir, or repo-relative data.
