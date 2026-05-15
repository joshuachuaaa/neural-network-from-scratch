# Security and Risk Review

## Summary

This is a local educational CLI with no network server, auth, secrets, or external service calls. Main risks are reliability, confusing UX, file overwrite behavior, dependency hygiene, and model correctness rather than conventional web security.

## Secrets Handling

No secrets found.

## Input Validation

- Good: Data loaders validate IDX magic/dimensions/count.
- Good: CLI validates positive hidden layer counts and positive train sample counts.
- Gap: Checkpoint schema/shape validation is missing.
- Gap: File path prompts allow arbitrary local paths.

## Auth / Permissions

No auth/permission model.

## Dangerous Operations

- Checkpoint save can overwrite files at the user-provided path.
- No delete/truncate/database destructive operations.

## Dependency Risks

- Dependencies are unpinned (`numpy`, `rich`, `pytest`).
- No lockfile.
- No automated vulnerability check.

## Performance Risks

- Repeated gzip dataset loading in CLI.
- `float64` images.
- Deep ReLU network with fixed small initialization may train poorly or slowly.
- Full dataset standard training can be CPU-bound.

## Operational Risks

- Relative data path depends on working directory.
- No CI.
- No target accuracy documented.

## Risk Register

| Risk | Area | Severity | Evidence | Recommendation |
|---|---|---:|---|---|
| Duplicated training loops can diverge | Architecture | High | `train.py:27-43`, `cli.py:287-320`, `visual_train.py:110-142` | Extract shared training event/session abstraction. |
| Non-reproducible model initialization | Core ML | High | `layers.py:23` uses global `np.random.randn`; `settings.RANDOM_SEED` only used in train/CLI/visual shuffling | Inject RNG into network/layers and test deterministic initialization. |
| Checkpoint schema is implicit and unvalidated | Persistence | Medium | `checkpoints.py:16-24`, `checkpoints.py:31-38` | Add version, shape/key validation, metadata, invalid-checkpoint tests. |
| CLI reloads data repeatedly | Performance/UX | Medium | `cli.py:249-261`, called by train/infer/eval | Add `DatasetBundle` cache in `AppState`. |
| Fixed watched sample confuses users | UX | Medium | `cli.py:271-285`, `visual_train.py:29`, `visual_train.py:93-108` | Show sample index clearly, rotate watched samples, separate batch labels from watched sample. |
| No training-quality regression test | Quality | Medium | Tests cover shapes/update but not convergence | Add tiny deterministic convergence test and optional full smoke benchmark. |
| Relative data path can fail after install | Operations | Medium | `DEFAULT_DATA_DIR = Path("data") / "mnist"` | Accept `--data-dir` in all commands, support env/config or package-relative resolution. |
| Package/repo naming mismatch | Packaging | Low | `pyproject.toml:6`, README/repo direction | Decide final name and align package metadata/docs. |

## Unknowns

- Desired checkpoint metadata and target accuracy.
