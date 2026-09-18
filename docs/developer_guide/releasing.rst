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
* On every pushed tag ``vX.Y.Z``: build the sdist and wheel with
  ``python -m build``, run ``twine check --strict``, verify that the tag
  equals ``v`` + ``xnn.__version__``, verify that the ``.frc`` force-field
  files and the D3 tables are inside the wheel, install the wheel into a
  clean virtual environment and import it, upload to PyPI, then create the
  GitHub Release for the tag (auto-generated notes) with the files attached.
  Plain ``git`` is all that is needed on your side.
* On a manual **Run workflow**: ``target = testpypi`` uploads the same build
  to `TestPyPI <https://test.pypi.org/p/xnns>`_ as a rehearsal, from any
  branch or tag; ``target = pypi`` publishes a tag that was pushed earlier
  (select the tag under *Use workflow from*; it must equal ``v`` +
  ``__version__``); ``target = none`` only builds and checks.

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
3. Tag and push (the tag must equal ``v`` + ``__version__`` or the workflow
   fails before uploading):

   .. code-block:: bash

      git tag -a v0.1.0 -m "xnn 0.1.0"
      git push origin v0.1.0

4. Watch *Actions > Release*. When it is green the package is at
   https://pypi.org/p/xnns, ``pip install xnns`` works, and the GitHub
   Release for the tag exists with the sdist and wheel attached.

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
