# Full Codebase Context Report

## 1. Executive Summary

Confirmed: This is a compact, local Python/Rich terminal app for visualizing and training a from-scratch NumPy MNIST classifier. It is installable, documented, test-backed, and has a strong terminal UI surface.

Main improvement priority: protect behavior with a few stronger tests, then remove duplicated training logic and improve reproducibility/checkpoint reliability.

## 2. Confidence Legend

- Confirmed: Directly observed in code, config, tests, docs, or command output.
- Inferred: Likely based on structure/naming/usage, but not directly documented.
- Unverified: Requires runtime execution beyond smoke checks, external context, or owner decision.
- Unknown: Not enough evidence found.

## 3. Repository Structure

Tracked source is under `src/neural_network_from_scratch/`; tests are under `tests/`; data is tracked under `data/mnist/`; screenshots are under `docs/assets/`; helper script is `scripts/verify_mnist.py`.

## 4. Runtime Architecture

Single-process local CLI. NumPy handles math/data; Rich handles terminal UI. No server, database, auth, network calls, queues, or background jobs.

## 5. Major Components

| Component | Purpose | Main Files | Key Symbols | QA Verdict | Risk Level |
|---|---|---|---|---|---|
| package-config-entrypoints | Packaging, defaults, commands | `pyproject.toml`, `settings.py`, `main.py`, `__main__.py` | `nnfs`, `nnfs-train`, `nnfs-visual`, constants | PASS WITH CORRECTIONS | Low |
| documentation-assets | README/screenshots | `README.md`, `docs/assets/*.png` | Quick start, terminal lab docs | PASS | Low |
| data-and-persistence | MNIST loading/checksums | `data.py`, `verify_mnist.py`, `data/mnist/**` | `load_mnist_images`, `load_mnist_labels`, `one_hot_encode` | PASS WITH CORRECTIONS | Low |
| neural-network-core | Model math | `activations.py`, `layers.py`, `network.py` | `Softmax`, `ReLU`, `Layer`, `NeuralNetwork` | PASS WITH CORRECTIONS | Medium |
| training-visualization | Training helpers/dashboard | `train.py`, `visual_train.py`, `rendering.py` | `train_epoch`, `dashboard`, `VisualMetrics` | PASS WITH CORRECTIONS | Medium |
| interactive-cli-checkpoints | Rich app/persistence | `cli.py`, `checkpoints.py` | `AppState`, `run_app`, `save_checkpoint`, `load_checkpoint` | PASS WITH CORRECTIONS | Medium |
| tests | Quality gate | `tests/*.py` | 26 tests | PASS WITH CORRECTIONS | Medium |

## 6. End-to-End Execution Flows

- Interactive: `nnfs` -> `cli.run_app()` -> model design/build/train/eval/infer/save/load.
- Standard: `nnfs-train` -> `train.main()` -> load data -> train epochs -> evaluate.
- Visual: `nnfs-visual` -> `visual_train.main()` -> load limited data -> live dashboard.

## 7. Data Model and Persistence

No database. Data is file-based MNIST gzip. Checkpoints are implicit NPZ payloads containing hidden dims and layer weights/biases.

## 8. External Integrations

No external services. Libraries: NumPy, Rich, pytest.

## 9. Configuration and Environment

Global constants in `settings.py`; package config in `pyproject.toml`; no env vars/secrets. Default data path is repo-relative `data/mnist`.

## 10. Testing and Quality

26 tests pass. Coverage is good for small units but missing convergence, CLI command-flow, checkpoint-failure, and deterministic initialization tests.

## 11. Observability and Operations

Rich UI gives strong local observability of layers/probabilities/metrics. No structured logs, CI, or coverage.

## 12. Security Review

Low conventional security risk. Main file-risk is arbitrary checkpoint path overwrite. No network/auth/secrets.

## 13. Performance Review

Potential improvements: cache data in CLI, use `float32`, improve initialization, avoid repeated decompression, document terminal width expectations.

## 14. Architecture Review

Strengths: simple, inspectable, good separation of core/rendering/data. Weaknesses: duplicated training loops, large CLI module, reproducibility gap, weak checkpoint schema.

## 15. Duplication and Abstraction Review

Highest-value abstraction is a shared training event/session generator used by standard, visual, and interactive flows. Dataset loading and checkpoint schema helpers are also worth extracting.

## 16. Codebase Quality Review

Clean small project with growing product surface. Refactor CLI/training boundaries before adding many more UI features.

## 17. Risk Register

| Risk | Area | Severity | Evidence | Recommendation |
|---|---|---:|---|---|
| Duplicated training loops | Architecture | High | `train.py`, `cli.py`, `visual_train.py` | Extract shared training event generator. |
| Unseeded initialization | Core ML | High | `layers.py` uses global `np.random.randn()` | Inject RNG/seed and add tests. |
| Weak checkpoint schema | Persistence | Medium | `checkpoints.py` implicit NPZ keys | Add version, validation, metadata. |
| Fixed watched sample UX | UI | Medium | `sample_index` defaults to 0 | Show index, rotate samples, separate watched sample from training labels. |
| No convergence test | Quality | Medium | Tests check shapes/updates only | Add tiny deterministic training-quality test. |
| Relative data path | Operations | Medium | `DEFAULT_DATA_DIR = Path("data") / "mnist"` | Add data-dir config and friendlier errors. |
| CLI mixed responsibilities | Architecture | Medium | `cli.py` 420 lines | Split after tests. |

## 18. Unknowns and Follow-Up Questions

Blocking: target accuracy; final project/package name. Non-blocking: checkpoint metadata, dataset distribution model, whether `nnfs-visual` remains separate.

## 19. Verified Component Reports

| Component | Reader Status | QA Verdict | Risk Level | Notes |
|---|---|---|---|---|
| package-config-entrypoints | reviewed | PASS WITH CORRECTIONS | Low | Naming/dependency cleanup. |
| documentation-assets | reviewed/summarized | PASS | Low | Add troubleshooting. |
| data-and-persistence | reviewed/summarized | PASS WITH CORRECTIONS | Low | Cache/helper improvements. |
| neural-network-core | reviewed | PASS WITH CORRECTIONS | Medium | Seed/init/shape validation. |
| training-visualization | reviewed | PASS WITH CORRECTIONS | Medium | Shared loop needed. |
| interactive-cli-checkpoints | reviewed | PASS WITH CORRECTIONS | Medium | Split CLI, validate checkpoints. |
| tests | reviewed | PASS WITH CORRECTIONS | Medium | Add behavioral coverage. |

## 20. Final Mental Model

Think of the codebase as a terminal-first educational ML lab. The core network is a small stateful NumPy model. Data loaders turn MNIST files into arrays. Training code performs mini-batch gradient descent. Rendering code turns internal model state into Rich tables/panels. The `nnfs` CLI ties everything together into an interactive app where users can change architecture, train, inspect, evaluate, infer, and checkpoint.

## 21. Recommended Next Actions

1. Medium-risk refactor: Add deterministic RNG/model initialization and He initialization.
2. Low-risk cleanup: Add convergence, checkpoint-invalid, and CLI command tests.
3. Medium-risk refactor: Extract shared training event generator.
4. Low-risk cleanup: Add dataset bundle/cache and centralize MNIST paths.
5. Low-risk cleanup: Add checkpoint schema/version validation.
6. Medium-risk refactor: Split `cli.py` into state, commands, menu rendering, and training orchestration.
7. Low-risk cleanup: Add CI plus `ruff`.
8. Needs more information: Decide package/repo display name alignment.
