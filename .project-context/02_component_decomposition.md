# Component Decomposition

## Component List

| Component | Purpose | Files/Folders | Reader Pass | QA Pass | Notes |
|---|---|---|---|---|---|
| package-config-entrypoints | Packaging, defaults, entrypoints, ignore rules. | `.gitignore`, `pyproject.toml`, `requirements.txt`, `main.py`, `src/neural_network_from_scratch/__init__.py`, `src/neural_network_from_scratch/__main__.py`, `src/neural_network_from_scratch/settings.py` | package-config-reader | package-config-qa | Includes project defaults and install scripts. |
| documentation-assets | User-facing docs and screenshots. | `README.md`, `docs/assets/terminal-ui.png`, `docs/assets/terminal-ui-latest.png` | documentation-reader | documentation-qa | Binary screenshots summarized. |
| data-and-persistence | MNIST loading, checksum verification, dataset files. | `src/neural_network_from_scratch/data.py`, `scripts/verify_mnist.py`, `data/mnist/**` | data-reader | data-qa | Gzip files summarized, not read byte-by-byte beyond checksum verification. |
| neural-network-core | Mathematical model implementation. | `src/neural_network_from_scratch/activations.py`, `src/neural_network_from_scratch/layers.py`, `src/neural_network_from_scratch/network.py` | core-reader | core-qa | Core business/domain logic. |
| training-visualization | Standard training helpers and Rich live dashboard. | `src/neural_network_from_scratch/train.py`, `src/neural_network_from_scratch/visual_train.py`, `src/neural_network_from_scratch/rendering.py` | visualization-reader | visualization-qa | Contains duplicated visual training loop. |
| interactive-cli-checkpoints | Rich menu app, interactive commands, checkpoint IO. | `src/neural_network_from_scratch/cli.py`, `src/neural_network_from_scratch/checkpoints.py` | cli-reader | cli-qa | Largest component; strongest refactor candidate. |
| tests | Automated tests. | `tests/*.py` | tests-reader | tests-qa | 26 tests pass. |

## Component Boundaries

- Confirmed: Core math (`activations.py`, `layers.py`, `network.py`) does not import CLI or Rich.
- Confirmed: Rendering imports core layer metadata (`LayerType`) but not CLI state.
- Confirmed: CLI imports almost every runtime component: data loaders, network, rendering, training helpers, visual update norms, checkpoint helpers.
- Confirmed: `visual_train.py` imports `dashboard()` from rendering and training helper functions from `train.py`, but reimplements its own training loop.

## Ambiguous Files

- `settings.py`: Classified under package-config-entrypoints because it holds global defaults; it is also consumed by core, training, data, CLI, and visualization.
- `main.py`: Classified under package-config-entrypoints as a backwards-compatible script. It is runtime-relevant but mostly delegates.

## Files Requiring Special Attention

- `src/neural_network_from_scratch/cli.py`: Large mixed-responsibility module and duplicated training loop.
- `src/neural_network_from_scratch/visual_train.py`: Duplicates CLI training loop behavior.
- `src/neural_network_from_scratch/checkpoints.py`: Lacks schema/version/shape validation and has a possible extension-return mismatch when saving to a path without `.npz`.
- `src/neural_network_from_scratch/layers.py`: Uses global unseeded `np.random.randn()` and fixed `0.01` initialization.

## Files Summarized Instead of Deep-Read

- `data/mnist/*.gz`: Binary dataset files. Checksums verified through `scripts/verify_mnist.py`.
- `docs/assets/*.png`: Binary README screenshots. File type and sizes inspected.

## Files Excluded With Justification

- `.git/**`: Git internals.
- Generated/ignored local files: `.project-context/**`, `.pytest_cache/**`, `src/**/*.pyc`, `src/*.egg-info/**`, `artifacts/**`.
