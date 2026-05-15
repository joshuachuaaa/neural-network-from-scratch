# Duplication and Abstraction Review

## Summary

The most important duplication is not superficial: training behavior is implemented in multiple places and can diverge. This is the highest-priority refactor area. Avoid over-abstracting the core math; it is small and readable.

## Duplicated Logic Findings

### Finding: Training Batch Loop

- Duplicated logic: forward pass, backprop, update norms/updates, loss/accuracy accumulation.
- Evidence: `train.py:27-43`, `cli.py:287-320`, `visual_train.py:110-142`.
- Why it matters: Bug fixes or training behavior changes must be made multiple times.
- Should it be abstracted? Yes.
- Recommended abstraction: `TrainingSession` or `iter_training_events()` generator that yields per-batch metrics, probabilities, update norms, and epoch summaries.
- Possible location: `src/neural_network_from_scratch/training_session.py` or expanded `train.py`.
- Risk of abstraction: Too much generic callback machinery could obscure educational clarity.
- Risk of not fixing: CLI and visual training diverge as features grow.
- Migration approach: Add new shared generator with tests; update `visual_train.py`; update `cli.py`; keep `train_epoch()` as wrapper.
- Confidence: Confirmed.

### Finding: MNIST Path Loading

- Duplicated logic: hardcoded train/test gzip names.
- Evidence: `train.py:59-62`, `visual_train.py:85-88`, `cli.py:250-253`.
- Why it matters: Dataset path changes or validation changes require multiple edits.
- Should it be abstracted? Yes.
- Recommended abstraction: `load_mnist_dataset(data_dir, limit_train=None, limit_test=None)` returning a `DatasetBundle`.
- Possible location: `data.py`.
- Risk of abstraction: Low.
- Risk of not fixing: Repeated IO and path drift.
- Migration approach: Add helper and replace call sites.
- Confidence: Confirmed.

### Finding: Model Parameter/Architecture Metadata

- Duplicated logic: architecture dimensions and parameter counts in CLI while model has layer data.
- Evidence: `cli.py:46-63`, `cli.py:70-88`, `cli.py:132-149`.
- Why it matters: Future model types could require CLI helper changes.
- Should it be abstracted? Maybe.
- Recommended abstraction: small `architecture.py` or model metadata methods.
- Possible location: `network.py` for methods, or new `model_summary.py`.
- Risk of abstraction: Could overcomplicate a small app.
- Risk of not fixing: Low until more architectures exist.
- Migration approach: Wait until adding new layer/activation types.
- Confidence: Inferred.

### Finding: Dependency Declarations

- Duplicated logic: Dependencies in `pyproject.toml` and `requirements.txt`.
- Evidence: `pyproject.toml:10-23`, `requirements.txt:1-3`.
- Why it matters: Dependency drift.
- Should it be abstracted? No code abstraction; choose one source of truth.
- Recommended abstraction: Generate requirements from project metadata or keep README recommending editable install.
- Risk of abstraction: Low.
- Risk of not fixing: Low.
- Migration approach: Make `requirements.txt` runtime-only or remove it.
- Confidence: Confirmed.

## Repeated Patterns That Should Stay Separate

| Pattern | Locations | Why Not Abstract | Confidence |
|---|---|---|---|
| Rich table construction | `cli.py`, `rendering.py` | Different UI contexts; abstraction would be premature unless theming grows. | Confirmed |
| Small test arrays | tests | Test-local clarity is better than fixtures for now. | Confirmed |

## Abstraction Opportunities

| Opportunity | Current Locations | Suggested Abstraction | Benefit | Risk | Priority |
|---|---|---|---|---|---|
| Training event stream | `train.py`, `cli.py`, `visual_train.py` | `iter_training_events()` / `TrainingSession` | One canonical training loop; easier visual UI | Medium | High |
| Dataset bundle/cache | `train.py`, `cli.py`, `visual_train.py`, `data.py` | `MnistDataset` dataclass/load helper | Less IO, clearer paths | Low | High |
| Checkpoint schema | `checkpoints.py` | `CheckpointPayload` validation helpers | Safer persistence | Low | High |
| RNG/config object | `settings.py`, `network.py`, `layers.py` | `ModelConfig`, `TrainingConfig` | Reproducibility/testability | Medium | High |
| CLI command modules | `cli.py` | `app_state.py`, `commands.py`, `menus.py` | Testability and maintainability | Medium | Medium |

## Over-Abstraction Risks

- Do not introduce a framework-like neural net API before adding more layer/activation types.
- Keep the core math readable for educational value.
- Prefer simple dataclasses and generators over callback-heavy architecture.

## Recommended Refactor Sequence

1. Add tests: deterministic initialization, tiny convergence, checkpoint invalid files.
2. Introduce `ModelConfig`/seeded RNG and He initialization.
3. Extract `load_mnist_dataset()` and optional CLI cache.
4. Extract shared training event generator.
5. Split `cli.py` after behavior is protected by tests.
6. Improve checkpoint schema/version.

## Open Questions

- Should checkpoint files include run history and settings?
- Should visual training and interactive CLI merge into one primary app path?
