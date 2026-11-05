# Contributing to xnn

We always appreciate your contributions. This guide explains what we need from
you so that an issue can be reproduced and a pull request can be merged without
rounds of back-and-forth. Contributions that follow this guide are often handled
first. Everyone taking part in the project is expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

- [Before you start](#before-you-start)
- [Reporting a bug](#reporting-a-bug)
- [Reporting a numerical discrepancy](#reporting-a-numerical-discrepancy)
- [Requesting a feature or a model](#requesting-a-feature-or-a-model)
- [How issues are triaged](#how-issues-are-triaged)
- [Setting up a development environment](#setting-up-a-development-environment)
- [Making a change](#making-a-change)
- [Adding a model](#adding-a-model)
- [Tests](#tests)
- [Documentation and examples](#documentation-and-examples)
- [Code style](#code-style)
- [Pull requests](#pull-requests)
- [Licensing and third-party code](#licensing-and-third-party-code)
- [Security issues](#security-issues)

## Before you start

- Search the [open and closed issues](https://github.com/molssi-ai/xnn/issues?q=is%3Aissue) as
  your problem may already be known or fixed.
- Check that contribution uses the latest release (`pip install -U xnns`) or with `dev`.
- For anything more significant or larger than a small fix, open an issue first
  and wait for a maintainer to agree on the approach. A pull request that
  changes the design without that agreement may be closed, however good the code
  may be.

## Reporting a bug

Use the **Bug report** form. A bug is worked on once a maintainer can run your example and see
the failure. Please provide:

1. **A minimal, self-contained script** that fails. It must run on its own, with no files from
   your machine:
   - build structures in code (for example with `ase.build.molecule`, `ase.build.bulk`, or a few
     lines of positions and atomic numbers), or load a public dataset through
     `xnn.common.data.load_dataset(...)`;
   - use a randomly initialised model or a public pretrained one (`from_pretrained(...)`), not
     a private checkpoint;
   - keep it as short as possible: the fewest atoms, the smallest model, the fewest steps that
     still show the problem;
   - seed everything that is random (`torch.manual_seed(0)`).
   
   If the problem only appears with your own data or checkpoint, say so, and include the
   smallest file that shows it, attached to the issue.
2. **What you expected and what happened**, with numbers where there are numbers.
3. **The full traceback**, as text, not a screenshot.
4. **Your environment**, printed by:

   ```bash
   python -c "import sys, platform, torch, xnn; print('xnn', xnn.__version__, xnn.__file__); print('python', sys.version.split()[0], platform.platform()); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
   pip list 2>/dev/null | grep -i -E "e3nn|ase|vesin|cuequivariance|mace|nequip|numpy|pymdi"
   ```

5. **CPU or GPU, float32 or float64**, and whether the failure depends on either.

A good report looks like this:

```python
import torch
from ase.build import molecule
from xnn.common.config import Config
from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput, build_model

torch.manual_seed(0)
cfg = Config()
cfg.model.name = "schnet"
cfg.model.cutoff = 5.0
model = ForceStressOutput(build_model(cfg.model)).eval()

atoms = molecule("H2O")
graph = structure_to_graph({"pos": atoms.positions, "atomic_numbers": atoms.numbers}, cfg.model.cutoff)
out = model(graph)
print(out["energy"], out["forces"])   # expected finite forces; got nan for the oxygen
```

Adapt the model and data to your case; the shape (imports, seeded model, one structure built in
code, the failing call, the expected result in a comment) is what makes it reproducible.

## Reporting a numerical discrepancy

Many xnn models reproduce an upstream code or a paper to round-off. If you find a difference,
use the **Numerical discrepancy** form and include:

- the reference: the upstream package and its exact version, or the paper and the equation,
  table or figure;
- a script that computes the same quantity with both, on the same structure and the same
  weights (the weights transplanted from the reference where it has any);
- both numbers, their difference, the precision (float32 or float64) and the device.

Differences at the level of float32 round-off (about 1e-6 relative) are expected in float32;
please report float64 results where you can.

## Requesting a feature or a model

Use the **Feature or model request** form. Describe the problem you want to solve before the solution
you have in mind. For a new model, give the paper, the reference implementation if there is
one (with its license), the data it was published on, and a number from the paper we can
reproduce to check the implementation.

## How issues are triaged

1. A maintainer runs your example. If it runs and shows the problem, the issue is labelled
   `confirmed`.
2. An issue without a runnable reproduction is labelled `needs-reproduction`. If none arrives
   within *2-3 weeks**, it is closed; it can be reopened as soon as one is posted.
3. A confirmed bug is fixed by first turning your example into a failing test, then making it
   pass. Your example is what we need most; a pull request is welcome but not required.
4. Questions about how to use xnn belong in
   [Discussions](https://github.com/molssi-ai/xnn/discussions), not issues.

## Setting up a development environment

```bash
git clone https://github.com/molssi-ai/xnn.git
cd xnn
git switch dev
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[ase,gnn,ffnn,vesin,dev]" "e3nn==0.4.4"
pytest -q
```

This is the environment the continuous integration uses. Add `[docs]` to build
the documentation and `[examples]` to run the example notebooks. The fused GPU
kernels (cuEquivariance) are optional and installed manually, as described in
`docs/how_tos/fast_paths.rst`.

## Making a change

- Branch from `dev` and open the pull request against `dev`. `main` holds released code.
- Keep one pull request to one purpose. A bug fix, a refactor and a new feature are three
  pull requests.
- Run the tests and the documentation build locally before pushing.

The design principles in the README and in `docs/developer_guide/` apply to every change.
The ones that matter most for review:

- **One implementation per component.** Before writing a helper, look for an existing one in
  `xnn.common` or in the model's family package and reuse or extend it. Duplicate
  implementations of the same thing are not merged.
- **Code lives with the family that uses it**: `gnn` (graph), `dnn` (dense), `cnn`
  (convolutional), `ffnn` (force fields), `sqmnn` (semi-empirical), `trfnn` (transformers),
  `hybnn` (hybrid); code shared by several families goes to `common`.
- **One data object.** Models take an `AtomicGraph` and return `node_energy` and `energy`;
  forces and stress are added by `ForceStressOutput`, never inside a model.
- **Canonical configuration keys.** Upstream spellings (MACE, NequIP, ...) are accepted through
  the key-translation registry (`register_key_translation`), never through aliases in
  `from_config`.

## Adding a model

Read `docs/developer_guide/extending.rst` first. A new model needs:

1. the module in its family package, registered with `@register_model`, and
   `configs/model/<name>.yaml`;
2. its fidelity reference, reproduced to float64 and float32 round-off whatever the implementation
   method: the upstream code under transplanted weights, the reference program, or, where neither
   exists, the paper's equations and published numbers.
   Every model has a **reference implementation** (`use_fast=False`): the eager composition of its
   equations, which runs on any device, has second derivatives and is the oracle of the tests. Its
   priorities, in order: *precision*, to round-off in float32 and float64; *compute efficiency*, at least that of the
   upstream code when the upstream is a comparable implementation; and *memory usage*, preferably similar to the
   upstream's or better. A model may add a **fast implementation**  (`use_fast=True`; `"auto"`, the
   default, takes it on a GPU in evaluation mode): fused kernels or better algorithms that compute
   the same numbers to round-off with first derivatives, faster and lighter than the reference, and
   at least as fast as a hand-optimized upstream code where one exists. Approximations that change
   the results (reduced tolerances, lower precision, extrapolation) are explicit options with a
   stated tolerance against the reference, never part of `use_fast`. Fast paths can also allow users to benefit from additional algorithmic, software and hardware optimizations, such as mixed precision, by changing different options without changing the model's equations.
   The defaults are the upstream code's as shipped, with the paper's values as options that can be set by the user where they differ;
3. the tests listed under [Tests](#tests), including the fidelity test against the reference;
4. its row in the model tables of `README.md` and the documentation;
5. if it is user-facing, a short example under `notebooks/examples/<family>/<model>/`, and its fidelity check under `notebooks/fidelity_checks/<family>/<model>/`, with the speed and memory comparison against the upstream code.

## Tests

```bash
pytest -q                     # the whole suite, as in CI (CPU)
pytest tests/test_mace.py -q  # one file
```

The suite is a flat `tests/` directory, one file per model or subsystem. Tests that need an
optional reference package skip when it is missing; GPU-only tests skip without CUDA. CI runs
on CPU with `XNN_OFFLINE=1`, so tests must not download anything.

Every new model gets at least:

- a registration test;
- invariance of the energy and equivariance of the forces under rotation and translation;
- periodic stress, and forces against finite differences;
- script-versus-eager parity if the model exports to TorchScript;
- fidelity to the reference to round-off: the upstream code under transplanted weights (guarded
  by an import check), the reference program, or an independent implementation of the paper's
  equations where neither exists;
- for a model with a fast implementation, its agreement with the reference implementation:
  energies, forces, stress and every intermediate it returns, to round-off in float64 and within
  the float32 reference's own deviation in float32 (GPU tests skip without CUDA).

Every bug fix gets a test that fails before the fix and passes after it.

Test conventions:

- no data files in `tests/`: build the inputs in the test;
- a one-line docstring per test, saying what it checks;
- no dates, timings, job numbers or narrative of how a bug was found in test code or comments;
  that belongs in the pull request description.

## Documentation and examples

- Docstrings use the NumPy style and are rendered by Sphinx. The CI builds the documentation
  with `-W`, so any warning, including malformed reStructuredText in a docstring or a `#:`
  comment, fails the build. Check locally:

  ```bash
  pip install -e ".[docs,gnn,ase]"
  sphinx-build -b html -W --keep-going docs docs/_build/html
  ```

- Documentation pages are concise and describe how things work and how to use them. Explain
  the reason for an internal fix in the docstring, the test or the pull request, not in the
  user documentation.
- Examples must be self-contained: no data files; build structures in code or load them
  through the dataset hub. A feature's example must exercise that feature (a long-range
  example uses charged or polar systems, not argon).
- Example notebooks are committed executed, with the import and TorchScript warnings
  suppressed in the first cell. Each notebook is added to `docs/examples/index.rst`.

## Code style

- Follow the style of the surrounding code: its naming, comment density and idioms.
- Comments explain why, briefly. No decorative comment banners (`# ---- section ----`); a plain
  comment is enough.
- Type hints on public functions; NumPy-style docstrings on every public class and function.
- Keep numerical constants derived from their sources (CODATA, the paper's tables), not
  retyped with fewer digits.
- `ruff check` must pass (the Lint workflow runs it; the rules are in `pyproject.toml`).
  Run it before pushing: `pip install ruff` (or `uvx ruff`), then `ruff check` at the repository
  root. The formatter is not enforced: `ruff format --check` only reports in CI.

## Pull requests

The pull request template lists what reviewers check. In short:

- the description says what changes and why, and links the issue (`Fixes #123`);
- the tests pass and new code is tested;
- the strict documentation build passes;
- `README.md` and the documentation are updated where behaviour, models or options change;
- the license of anything you drew on is stated (next section).

Small, focused pull requests are reviewed fastest. Maintainers may push small fixes to your
branch before merging.

## Licensing and third-party code

xnn is released under the MIT license, and contributions are accepted under it.

- **Do not copy code from other projects**, whatever their license. Implement from the paper
  and from the documented behaviour, and prove it with fidelity tests against the
  reference implementation installed as a test dependency. Upstream code may be imported in
  tests; it is not vendored.
- **Do not read or consult implementations under copyleft licenses (GPL, AGPL, LGPL)** while
  writing code for xnn. If the only reference implementation is copyleft, say so in the issue
  first.
- Data and parameter files (force-field tables, dispersion coefficients) may be included only
  under a license that allows redistribution; state the source, version and license in the
  pull request and keep the notice next to the file.

## Security issues

Do not open a public issue for a security problem. Report it privately through
**Security > Report a vulnerability** on the GitHub repository.
