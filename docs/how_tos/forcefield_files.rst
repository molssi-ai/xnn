.. _howto-forcefield-files:

*******************************************
Use and write ``.frc`` force-field files
*******************************************

The classical force fields of the ``ffnn`` family -- ReaxFF, OPLS, and the
ones to come -- read their parameters from one format: the MolSSI/SEAMM
``.frc`` force-field file (:mod:`xnn.ffnn.common.frc`), the format of the
`SEAMM force-field distribution
<https://github.com/molssi-seamm/forcefield_step/tree/main/forcefield_step/data>`_.
Besides the parameters, a ``.frc`` file carries the **SMARTS templates**
that assign its own atom types to any structure, so a fixed-topology force
field such as OPLS can be applied to a molecule without knowing its type
names.

What ships with xnn
====================
.. code-block:: python

   from xnn.ffnn.common import list_forcefields
   sorted(list_forcefields())
   # ['CL&P', 'lopls', 'oplsaa', 'oplsaa+', 'oplsaa-1996',
   #  'reaxff/CHLiOFSi_Yun_2017', ..., 'reaxff/CHO_cho_2008', ...]

``oplsaa.frc`` is the OPLS-AA distribution (variants ``oplsaa``, the
CL&P ionic-liquid extension ``CL&P``, and their union ``oplsaa+``), and a
dozen published ReaxFF fields live under ``reaxff/``; all are copied verbatim
from SEAMM (BSD-3-Clause, see ``src/xnn/ffnn/data/README.md`` for the
commit and per-file provenance). xnn adds ``lopls`` (Siu *et al.* 2012) and
``oplsaa-1996`` (the alkane and alcohol torsions of the original paper),
both small files that ``#include`` ``oplsaa.frc`` and override a few rows.

Every model and reader accepts the same *spec*: a variant name
(``"oplsaa"``, ``"CHO_cho_2008"``), a path to a ``.frc`` file, or
``"<path>.frc:<variant>"`` to pick one of several variants in a file.

Apply a force field to a molecule
=================================
.. code-block:: python

   from ase.build import molecule
   from xnn.ffnn.models import OPLS, ReaxFF

   ethanol = molecule("CH3CH2OH")
   model = OPLS.from_atoms(ethanol, "oplsaa", cutoff=12.0)   # types + topology
   model.topology.types            # ['opls_80', 'opls_99', 'opls_96', ...]

   reax = ReaxFF("CHO_cho_2008")   # ReaxFF needs no typing: species only

``OPLS.from_atoms`` perceives the bonding with RDKit, assigns the library's
atom types from its templates (:func:`~xnn.ffnn.common.typing.assign_atom_types`),
derives the topology, and places impropers at trigonal centers the library
has a pattern for. RDKit is an optional dependency::

   pip install "xnn[ffnn]"

The lower-level pieces are available separately when you need them:

.. code-block:: python

   from xnn.ffnn.common import read_forcefield, assign_atom_types
   from xnn.ffnn.models import MolecularTopology, read_opls

   ff = read_forcefield("oplsaa")                  # resolved variant
   types = assign_atom_types(ethanol, ff)          # SMARTS typing only
   top = MolecularTopology.from_ase(ethanol, forcefield="oplsaa")
   lib = read_opls("CL&P", strict=False)           # skip forms OPLS lacks

Query a force field directly
============================
:class:`~xnn.ffnn.common.frc.ForceField` exposes the lookups force-field
codes need, with the equivalence and wildcard rules of the format:

.. code-block:: python

   ff.charge("opls_80")                     # -0.18 (through the NonB equivalence)
   ff.nonbond("opls_80")                    # (3.5, 0.066): sigma / A, eps / kcal/mol
   ff.bond("opls_80", "opls_81")            # -> ('quadratic_bond', key, Row)
   ff.torsion("opls_85", "opls_18", "opls_18", "opls_85")[2].values
   ff.improper("opls_89", "opls_89", "opls_88", "opls_88")   # center third
   ff.templates["opls_80"]["smarts"]        # ['[CD4H3:1][#6]' ...]

Units are those of the format's defaults (kcal/mol, Angstrom, degrees);
``@units`` and ``@type`` modifiers in the file are converted on reading, so a
section written in kJ/nm or as ``rmin``/``epsilon`` comes out the same way.

Write a force field
===================
Trained parameters go back out in the same format:

.. code-block:: python

   trained = model.ff.export_library()      # OPLSLibrary, kcal/mol units
   trained.save_frc("my_opls.frc", name="my-opls")
   OPLS.from_atoms(ethanol, "my_opls.frc")  # ... and straight back in

   reax.export_library().save("ffield.json")   # ReaxFF-nn weights: JSON only

The written file carries the templates, so it types structures like the
original. Two caveats: a dihedral ``V0`` constant has no column in
``torsion_opls`` and is dropped (it affects neither forces nor energy
differences), and ReaxFF-nn network weights have no place in the format, so
neural libraries keep using the JSON format of
:meth:`~xnn.ffnn.models.ffield.FFieldLibrary.save`.

The format in brief
===================
A file opens with ``!MolSSI forcefield 1``: the word after ``!`` is the
dialect (``MolSSI``, or ``BIOSYM`` in legacy files) and the trailing number
is the *format version*, the version of the file grammar. Version 1 is the
current and only published one; it is what xnn implements and writes
(:data:`~xnn.ffnn.common.frc.FRC_FORMAT_VERSION`), every shipped file
declares it, :class:`~xnn.ffnn.common.frc.FrcFile` exposes it as
``format_version``, and a newer number is parsed with a warning. It is not a parameter version: those sit in the ``Version``
column of every row (see below), and the newest wins. Everything else is
organised in sections that start at a ``#`` line and run to the next one::

   #quadratic_bond oplsaa          <- kind, label

   > E = K2 * (R - R0)^2           <- annotation
   @units K2 kJ/mol/nm^2           <- modifier (optional)

   !Version    Ref  I        J        R0      K2     <- column header
   2023.01.29  1    opls_18  opls_18  1.5290  268.00 <- data row

``#define <name>`` lists, per functional form, the labelled sections that
make up the variant ``<name>``; later labels override earlier ones for the
same key, which is how ``CL&P`` and ``lopls`` extend ``oplsaa``, and within a
section the newest version of each key wins. ``#include <file>`` splices
another file in (``local:`` resolves against the xnn data directory and any
``include_dirs`` you pass); ``#templates`` and ``#fragments`` hold JSON;
``#reference <n>`` holds provenance.

Adding a new force field
========================
Write its parameters as a ``.frc`` file (any unknown section kind is read
from its column header; register known ones with
:func:`~xnn.ffnn.common.frc.register_section_schema` to get key symmetry
and unit conversion), give it a ``#templates`` section, and write the small
bridge that turns a :class:`~xnn.ffnn.common.frc.ForceField` into the
model's parameter tables -- :func:`xnn.ffnn.models.oplslib.from_forcefield`
and :func:`xnn.ffnn.models.ffield.from_forcefield` are the two existing
examples. Files placed in ``src/xnn/ffnn/data/`` are found by name.
