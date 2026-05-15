# External Integrations

## Summary

No external services or network APIs are used. External libraries are local Python dependencies.

| Integration | Used In | Purpose | Auth | Failure Handling | Notes |
|---|---|---|---|---|---|
| NumPy | Core, data, training, rendering, checkpoints, tests | Numeric arrays, math, random, checkpoint NPZ | None | Mostly implicit exceptions | Required runtime dependency. |
| Rich | CLI, rendering, visual training, tests | Terminal UI/rendering/prompts | None | Prompt/render exceptions implicit | Required runtime dependency. |
| pytest | Tests | Test runner | None | Test failures | Dev dependency in `pyproject.toml`; listed in `requirements.txt`. |
| gzip/struct/hashlib/pathlib | Data and scripts | Standard library file parsing/checksums | None | ValueError/SystemExit/file exceptions | No network. |

## Integration Details

### NumPy

- Purpose: Neural network math, arrays, RNG, checkpoints.
- Files/functions: `activations.py`, `layers.py`, `network.py`, `data.py`, `train.py`, `visual_train.py`, `rendering.py`, `checkpoints.py`, tests.
- Auth/config: none.
- Request/response: not applicable.
- Retry/timeout: not applicable.
- Failure behavior: NumPy errors propagate.
- Risks: Version not pinned; no deterministic model init.
- Unknowns: None.

### Rich

- Purpose: Terminal menus, prompts, panels, tables, live dashboard.
- Files/functions: `cli.py`, `visual_train.py`, `rendering.py`.
- Auth/config: none.
- Failure behavior: Exceptions propagate to CLI catch block or command exit.
- Risks: Terminal width/TTY behavior can affect display.

## No External Network Calls

Confirmed by search: no `requests`, `httpx`, `urllib`, `aiohttp`, `fetch`, `axios`, websocket, gRPC, or similar runtime network usage.
