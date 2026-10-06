.. _releasing:

*******************
Releasing to PyPI
*******************

The distribution on PyPI is ``xnns``; the import package, the command line
and the repository are ``xnn``. Releases are built and uploaded by
``.github/workflows/release.yml`` through `PyPI Trusted Publishing
<https://docs.pypi.org/trusted-publishers/>`_, so no token is stored
anywhere. The version is defined once, in ``src/xnn/__init__.py``.

Cutting a release
=================
1. Bump ``__version__``, make sure ``pytest tests`` and the strict docs build
   pass, and push to ``main``.
2. Rehearse on TestPyPI: *Actions > Release > Run workflow* with
   ``target = testpypi``, then install from https://test.pypi.org/p/xnns in
   a scratch environment.
3. Tag and push; the tag must equal ``v`` + ``__version__``:

   .. code-block:: bash

      git tag -a v0.1.0 -m "xnn 0.1.0"
      git push origin v0.1.0

The workflow builds the sdist and wheel, runs ``twine check --strict``,
checks that the ``.frc`` files and the dispersion tables are in the wheel,
installs it in a clean environment, uploads to PyPI and creates the GitHub
Release. A version can be uploaded only once; to redo a release, bump the
version.

One-time setup
==============
Create the GitHub environments ``pypi`` and ``testpypi`` (*Settings >
Environments*), then add a pending trusted publisher on PyPI and TestPyPI
with project name ``xnns``, owner ``molssi-ai``, repository ``xnn``, workflow
``release.yml`` and the matching environment name.

Metadata rules
==============
PyPI refuses direct URL requirements, so the Allegro reference
implementation (a git dependency) is not part of any extra. The README must
be valid Markdown with absolute links, and the ``license`` field is an SPDX
expression (``MIT``); the BSD-3 notice of the SEAMM force-field files
travels inside the package.
