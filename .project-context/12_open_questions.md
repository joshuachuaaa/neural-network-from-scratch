# Open Questions

## Blocking Unknowns

| Question | Area | Why It Matters | Evidence Needed |
|---|---|---|---|
| What target accuracy or training behavior should be considered acceptable? | ML quality | Needed for meaningful convergence tests and defaults. | Owner expectation or benchmark run. |
| Should the project/package be renamed to match `neuralnet-terminal-visualizer`? | Packaging/docs | Affects PyPI/package name, imports, README, repo identity. | Naming decision. |

## Non-Blocking Unknowns

| Question | Area | Why It Matters | Evidence Needed |
|---|---|---|---|
| Should checkpoints include training history, settings, and data hashes? | Persistence | Determines checkpoint schema design. | Desired UX. |
| Should MNIST files stay committed or become downloadable artifacts? | Repo ops | Affects clone size and offline demo reliability. | Project distribution goal. |
| Should `nnfs-visual` remain separate from `nnfs`? | UX/API | Affects command surface and training loop refactor. | Preferred user workflow. |
| Should installed commands work outside repo root with bundled or configured data? | Operations | Current default relative path assumes repo root. | Packaging/distribution goal. |

## Nice-to-Clarify Items

| Question | Area | Why It Matters | Evidence Needed |
|---|---|---|---|
| Preferred lint/format/type tooling? | Quality | Helps enforce style without surprising contributors. | Owner preference. |
| Should the UI support rotating watched samples during training? | UX | Reduces confusion about fixed label/prediction display. | Desired visual behavior. |
| Should future model types be supported? | Architecture | Determines whether to abstract layer/config now or later. | Roadmap. |
