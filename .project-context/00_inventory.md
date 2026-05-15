# Repository Inventory

## Summary

- Repository root: `/home/joshua/Projects/neural-network-from-scratch`
- Current branch: `main`
- Git status at scan time: `## main...origin/main`
- Total tracked files: 28 (`git ls-files`)
- Untracked files observed: none from `git status`; ignored/generated files observed by `find`: `.project-context/**`, `.pytest_cache/**`, `src/neural_network_from_scratch.egg-info/**`, `src/neural_network_from_scratch/__pycache__/**`, `artifacts/menu-test.npz`.
- Generated/vendor/binary directories identified: `.git/**` excluded; `.project-context/**` generated scan output; `.pytest_cache/**` generated; `*.egg-info/**` generated editable-install metadata; `__pycache__/**` generated; `artifacts/**` local checkpoints; `data/mnist/*.gz` binary dataset files; `docs/assets/*.png` binary README screenshots.
- Verification commands run: `pytest -q` -> 26 passed; `python scripts/verify_mnist.py` -> MNIST checksums verified; `python -m compileall -q src tests scripts main.py` -> passed; CLI launch smoke and visual training smoke passed.

## Inventory

| Path | Type | Component | Status | Reader | QA | Notes |
|---|---|---|---|---|---|---|
| `.gitignore` | config | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Ignores caches, `.project-context/`, artifacts, egg-info. |
| `README.md` | docs | documentation-assets | reviewed | documentation-reader | documentation-qa | Project overview, screenshots, quick start, runtime model. |
| `data/mnist/SHA256SUMS` | checksum manifest | data-and-persistence | reviewed | data-reader | data-qa | Hash manifest used by `scripts/verify_mnist.py`. |
| `data/mnist/t10k-images-idx3-ubyte.gz` | binary data | data-and-persistence | summarized | data-reader | data-qa | MNIST test images; checksum verified. |
| `data/mnist/t10k-labels-idx1-ubyte.gz` | binary data | data-and-persistence | summarized | data-reader | data-qa | MNIST test labels; checksum verified. |
| `data/mnist/train-images-idx3-ubyte.gz` | binary data | data-and-persistence | summarized | data-reader | data-qa | MNIST train images; checksum verified. |
| `data/mnist/train-labels-idx1-ubyte.gz` | binary data | data-and-persistence | summarized | data-reader | data-qa | MNIST train labels; checksum verified. |
| `docs/assets/terminal-ui-latest.png` | binary image | documentation-assets | summarized | documentation-reader | documentation-qa | README screenshot; 899x130 PNG. |
| `docs/assets/terminal-ui.png` | binary image | documentation-assets | summarized | documentation-reader | documentation-qa | README screenshot; 914x1127 PNG. |
| `main.py` | Python entrypoint | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Backwards-compatible script; inserts `src` into `sys.path`. |
| `pyproject.toml` | package config | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Defines package, deps, console scripts, pytest config. |
| `requirements.txt` | dependency file | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Runtime-style requirements include `pytest`. |
| `scripts/verify_mnist.py` | script | data-and-persistence | reviewed | data-reader | data-qa | Hashes dataset files against manifest. |
| `src/neural_network_from_scratch/__init__.py` | Python package | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Exposes `NeuralNetwork`. |
| `src/neural_network_from_scratch/__main__.py` | Python module entrypoint | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Runs `train.main()`. |
| `src/neural_network_from_scratch/activations.py` | Python source | neural-network-core | reviewed | core-reader | core-qa | ReLU and Softmax. |
| `src/neural_network_from_scratch/checkpoints.py` | Python source | interactive-cli-checkpoints | reviewed | cli-reader | cli-qa | NPZ save/load. |
| `src/neural_network_from_scratch/cli.py` | Python source | interactive-cli-checkpoints | reviewed | cli-reader | cli-qa | Rich menu, training, inference, eval, checkpoint commands. |
| `src/neural_network_from_scratch/data.py` | Python source | data-and-persistence | reviewed | data-reader | data-qa | IDX gzip loading and label encoding. |
| `src/neural_network_from_scratch/layers.py` | Python source | neural-network-core | reviewed | core-reader | core-qa | Layer state, activation, gradient update. |
| `src/neural_network_from_scratch/network.py` | Python source | neural-network-core | reviewed | core-reader | core-qa | Network topology, prediction, backpropagation. |
| `src/neural_network_from_scratch/rendering.py` | Python source | training-visualization | reviewed | visualization-reader | visualization-qa | Rich tables, digit renderer, dashboard. |
| `src/neural_network_from_scratch/settings.py` | Python source | package-config-entrypoints | reviewed | package-config-reader | package-config-qa | Constants for dimensions, defaults, seed. |
| `src/neural_network_from_scratch/train.py` | Python source | training-visualization | reviewed | visualization-reader | visualization-qa | Batch iteration, loss, train/eval, standard entrypoint. |
| `src/neural_network_from_scratch/visual_train.py` | Python source | training-visualization | reviewed | visualization-reader | visualization-qa | Argparse visual training entrypoint and update norms. |
| `tests/test_activations.py` | test | tests | reviewed | tests-reader | tests-qa | Activation unit tests. |
| `tests/test_checkpoints.py` | test | tests | reviewed | tests-reader | tests-qa | Checkpoint happy-path round trip. |
| `tests/test_cli.py` | test | tests | reviewed | tests-reader | tests-qa | CLI renderable helpers and parameter counts. |
| `tests/test_data_loading.py` | test | tests | reviewed | tests-reader | tests-qa | IDX gzip loader validation. |
| `tests/test_network.py` | test | tests | reviewed | tests-reader | tests-qa | Network shape, topology, gradient, update tests. |
| `tests/test_rendering.py` | test | tests | reviewed | tests-reader | tests-qa | Rich renderables and update norm tests. |
| `tests/test_training.py` | test | tests | reviewed | tests-reader | tests-qa | Batching and evaluation helper tests. |

## Explicit Exclusions

| Pattern | Status | Reason |
|---|---|---|
| `.git/**` | excluded | Git internals, not application/runtime code. |
| `.project-context/**` | generated | Deep-read reports generated by this scan. |
| `.pytest_cache/**` | generated | Test runner cache. |
| `src/neural_network_from_scratch.egg-info/**` | generated | Editable install metadata ignored by `.gitignore`. |
| `src/neural_network_from_scratch/__pycache__/**` | generated | Python bytecode cache ignored by `.gitignore`. |
| `artifacts/menu-test.npz` | generated | Local checkpoint artifact ignored by `.gitignore`; not part of tracked source. |
