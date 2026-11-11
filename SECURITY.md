# Security policy

xnn is a research library. It reads model files, configuration files and datasets, and it compiles
GPU and CPU kernels at run time. This page explains how to report a vulnerability, which versions
receive fixes, and which inputs xnn treats as trusted.

- [Reporting a vulnerability](#reporting-a-vulnerability)
- [What happens next](#what-happens-next)
- [Supported versions](#supported-versions)
- [What xnn trusts](#what-xnn-trusts)
- [Out of scope](#out-of-scope)
- [Dependencies and releases](#dependencies-and-releases)

## Reporting a vulnerability

Report it privately. Do not open a public issue, discussion or pull request.

1. Open [Report a vulnerability](https://github.com/molssi-ai/xnn/security/advisories/new) on the
   repository (**Security > Advisories > Report a vulnerability**).
2. If you cannot use GitHub, write to the project team at ai@molssi.org with "xnn security" in
   the subject.

Please include:

- the xnn version (`python -c "import xnn; print(xnn.__version__)"`), the Python and PyTorch
  versions, and the operating system;
- the affected function, command or file format;
- a minimal script that shows the problem. As for bug reports, it builds its inputs in code. For a
  malicious file, the script writes that file;
- what an attacker gains, and under which conditions;
- whether the problem is already public, and any disclosure date you need.

Do not send private data, unpublished models or credentials. A constructed file that shows the
problem is enough.

## What happens next

- We acknowledge your report within 5 working days.
- Within 14 days we tell you whether we accept it and how severe we judge it.
- We work on the fix in a private fork attached to the GitHub advisory. A test that fails before
  the fix and passes after it comes with every fix.
- The fix ships in a new release on PyPI. We aim to release within 90 days of the report, and
  sooner for severe problems.
- We then publish the GitHub security advisory and request a CVE when one is warranted. We credit
  you in the advisory unless you prefer not to be named.

Please keep the report private until the advisory is published. If we miss these dates, tell us
through the same private channel before you disclose.

## Supported versions

Fixes go into the latest release of [`xnn`](https://pypi.org/project/xnns/) and into the `dev`
branch. Older releases are not patched. Upgrade with:

```bash
pip install -U xnns
```

A minor release does not break the public API or the saved model format, so upgrading within a
major version is safe.

## What xnn trusts

Some xnn inputs are designed to be safe to load from anyone. Others can run code by design.

**Safe to load from untrusted sources.** A way to run code or write outside the target directory
through these inputs is a vulnerability. Please report it.

- Portable model directories, written by `save_pretrained()` and read by `from_pretrained()` or
  `load_checkpoint()`. The weights load with `torch.load(weights_only=True)`, and the
  configuration and card are plain YAML and JSON.
- Configuration files and command-line options. YAML is parsed with `yaml.safe_load`, and option
  values with `ast.literal_eval`.
- Files fetched by the dataset hub, the model hub and the semiempirical parameter download. Each
  file is checked against an MD5 checksum stored in xnn or in the record it comes from. Archive
  members cannot land outside the target directory (for tar archives, on Python 3.11.4 and later).

**Trusted inputs only.** These run code or open the machine by design. Load them only from
sources you trust.

- Trainer checkpoints (`.pt` files written by `Trainer.save()`). They are pickles and load with
  `weights_only=False`. To share a model, convert it with `save_pretrained()` and share the
  directory instead.
- Checkpoints of other packages read through a converter (`format="mace-torch"`,
  `MACE.from_foundation()` with a path or URL). The upstream package unpickles them.
- TorchScript exports for LAMMPS and OpenMM. A TorchScript file is a program.
- The MDI engine (`xnn mdi`, `MDIEngine`). Its TCP connection has no authentication or
  encryption. Bind it to `localhost` or to a private cluster network.
- The kernel caches. Triton and the optional C++ kernels (`XNN_CPP_KERNELS=1`) compile code into
  cache directories under `~/.cache` (or `XDG_CACHE_HOME`, `XNN_CPU_KERNELS_DIR`). Anyone who can
  write there can run code as you, so keep them private to your account.

## Out of scope

- **Vulnerabilities in dependencies** (PyTorch, NumPy, ASE, e3nn, mace-torch, Triton and others).
  Report them to those projects. Do tell us when xnn makes one reachable, or when xnn's
  requirements prevent you from installing a fixed version.
- **Wrong numbers.** A result that differs from the reference code or the paper is a bug. Use the
  **Numerical discrepancy** issue form described in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Resource use that grows with the input.** A very large structure, cutoff or batch can exhaust
  memory or time. That is expected for a simulation code.
- **Anything under "Trusted inputs only"** above, when it behaves as described there.

## Dependencies and releases

- Dependabot opens a pull request as soon as an advisory affects a locked dependency or a
  workflow action. Routine updates arrive against `dev` as one grouped pull request a week.
- Security updates are checked with `pip-audit` on the exported lockfile.
- Some packages are held on purpose in the development lockfile: `torch` (the CUDA build the GPU
  environments use), `e3nn` (the version mace-torch requires) and `nequip` and `nequip-allegro`
  (the versions the comparison tests use). A PyTorch advisory fixed in a later release can
  therefore stay on the development lockfile until the next manual bump. Installs from PyPI are not affected: xnn
  requires `torch>=2.0` and resolves a current release.
- xnn vendors no third-party code.
- Releases are built, tested and uploaded only by the release workflow, through PyPI Trusted
  Publishing. No upload token exists. Each release is rehearsed on TestPyPI, its tag must match
  the package version, and its wheel is installed and tested in a clean environment before upload.
- The workflows run with read-only repository permissions. Only the release and documentation jobs
  receive the extra permissions they need to publish.
