## What and why

<!-- What this changes and why. Link the issue: "Fixes #123". One purpose per pull request. -->

## How it was checked

<!-- The tests you added or ran, and any numbers (fidelity, parity, timings) that support the change. -->

## Checklist

- [ ] Branched from `dev` and targeting `dev`.
- [ ] `pytest -q` passes; new code is tested (a bug fix adds a test that failed before the fix).
- [ ] `ruff check` passes.
- [ ] `sphinx-build -b html -W --keep-going docs docs/_build/html` passes.
- [ ] Reuses existing helpers in `xnn.common` or the family package; no second implementation of an existing component.
- [ ] No data files in `tests/` or `notebooks/`; examples build their inputs in code or load them through the hubs.
- [ ] `README.md` and the documentation are updated where behaviour, models or options change (including the model tables).
- [ ] For a model: registration, invariance and equivariance, periodic stress, finite-difference forces, script parity if it exports, and fidelity to its reference.
- [ ] No code copied from other projects; any reference implementation I consulted is not under a copyleft license; the sources and licenses of new data files are stated below.

## Sources and licenses

<!-- Papers, reference implementations (with version and license) and data files this change relies on. -->
