.. _releasing:

*******************
Releasing to PyPI
*******************

The distribution on PyPI is called ``xnns``; the import package, the command
line and the GitHub repository are ``xnn`` (the PyPI name ``xnn`` belongs to
someone else). Releases are built and uploaded by the GitHub Actions workflow
``.github/workflows/release.yml`` using `PyPI Trusted Publishing
<https://docs.pypi.org/trusted-publishers/>`_, so no API token is stored
anywhere.

What the workflow does
======================
* On every **published GitHub Release** whose tag is ``vX.Y.Z``: build the
  sdist and wheel with ``python -m build``, run ``twine check --strict``,
  verify that the tag equals ``v`` + ``xnn.__version__``, verify that the
  ``.frc`` force-field files and the D3 tables are inside the wheel, install
  the wheel into a clean virtual environment and import it, then upload to
  PyPI and attach the files to the GitHub Release.
* On a manual **Run workflow** with ``target = testpypi``: the same build,
  uploaded to `TestPyPI <https://test.pypi.org/p/xnns>`_ instead. Use this to
  rehearse.

The version is defined once, in ``src/xnn/__init__.py`` (``__version__``);
``pyproject.toml`` reads it (``dynamic = ["version"]``).

One-time setup
==============
1. **GitHub environments.** In the repository go to *Settings > Environments*
   and create ``pypi`` and ``testpypi``. Optionally add yourself as a required
   reviewer of ``pypi`` so every upload needs an explicit approval.
2. **PyPI.** Log in at https://pypi.org, open *Your account > Publishing* and
   add a *pending* trusted publisher (the project does not exist yet, it is
   created by the first upload):

   * PyPI project name: ``xnns``
   * Owner: ``molssi-ai``
   * Repository name: ``xnn`` (the name the repository has *when the workflow
     runs*)
   * Workflow name: ``release.yml``
   * Environment name: ``pypi``
3. **TestPyPI.** The same at https://test.pypi.org with environment
   ``testpypi``.

Cutting a release
=================
1. Bump ``__version__`` in ``src/xnn/__init__.py``, make sure ``pytest tests``
   and the strict docs build pass, commit and push to ``main``.
2. Rehearse: *Actions > Release > Run workflow* with ``target = testpypi``.
   Check https://test.pypi.org/p/xnns and, in a scratch environment,
   ``pip install --index-url https://test.pypi.org/simple/ --extra-index-url
   https://pypi.org/simple/ xnns``.
3. Create the release: *Releases > Draft a new release*, new tag ``vX.Y.Z``
   on ``main`` (the tag must equal ``v`` + ``__version__`` or the workflow
   fails before uploading), title ``xnn X.Y.Z``, generate release notes,
   *Publish release*.
4. Watch *Actions > Release*. When it is green the package is at
   https://pypi.org/p/xnns and ``pip install xnns`` works.

A version can be uploaded to PyPI only once. If a release has to be redone,
bump the version (for example ``0.1.1``) and publish a new release.

Metadata rules that PyPI enforces
=================================
* No direct URL requirements in the metadata: the Allegro reference
  implementation (``mir-group/allegro``, a git dependency) is therefore not
  part of any extra and is installed by hand for its fidelity notebook.
* ``twine check --strict`` must pass, so the README has to be valid Markdown
  with absolute links only.
* The ``license`` field is an SPDX expression (``MIT``), which needs
  ``setuptools >= 77`` at build time. The BSD-3 notice of the vendored SEAMM
  force-field files travels inside the package itself
  (``xnn/ffnn/data/LICENSE-SEAMM``), so it is in every wheel and sdist.
