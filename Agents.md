# Agent Rules

当前项目的conda环境为`rfsig`

- Read the current repository and relevant documentation before changing code.
- Keep signal/data generation (`src/rsig`) independent from reconstruction algorithms (`src/rdps`).
- Prefer small, testable changes that preserve reproducibility and explicit configuration.
- Do not mix unrelated refactors into an experiment change.
- Every experiment must record its configuration, random seed, dataset split, metrics, and output location.
- Validate new behavior with focused tests and report any unrun checks.
- Update the relevant documentation when an interface, assumption, or research milestone changes.

