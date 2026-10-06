.. _howto-forcefield-files:

*******************************************
Use and write ``.frc`` force-field files
*******************************************

The classical force fields (ReaxFF, OPLS, DREIDING) read their parameters
from one format: the MolSSI/SEAMM ``.frc`` file (:mod:`xnn.ffnn.common.frc`),
the format of the `SEAMM force-field distribution
<https://github.com/molssi-seamm/forcefield_step/tree/main/forcefield_step/data>`_.
A ``.frc`` file also carries SMARTS templates that assign its atom types to
any molecule, so a fixed-topology field applies without knowing its type
names.

What ships with xnn
===================

.. code-block:: python

   from xnn.ffnn.common import list_forcefields

   sorted(list_forcefields())
   # ['CL&P', 'dreiding', 'dreiding/X6', 'lopls', 'oplsaa', 'oplsaa+', 'oplsaa-1996',
   #  'reaxff/CHLiOFSi_Yun_2017', ..., 'reaxff/CHO_cho_2008', ...]

``oplsaa`` is the OPLS-AA distribution (``CL&P`` adds the ionic-liquid
extension, ``oplsaa+`` is their union), ``dreiding`` and ``dreiding/X6`` are
the two DREIDING nonbond forms, and a dozen published ReaxFF fields live
under ``reaxff/``. All are copied verbatim from SEAMM (BSD-3-Clause). xnn
adds ``lopls`` (Siu *et al.* 2012) and ``oplsaa-1996`` (the torsions of the
original paper), which ``#include`` ``oplsaa.frc`` and override a few rows.

Every model and reader takes the same *spec*: a variant name, a path to a
``.frc`` file, or ``"<path>.frc:<variant>"``.

Apply a force field to a molecule
=================================
RDKit (``ffnn`` extra) perceives the bonds and the templates assign the
types:

.. code-block:: python

   from ase.build import molecule
   from xnn.ffnn.models import OPLS, ReaxFF, Dreiding

   ethanol = molecule("CH3CH2OH")
   opls = OPLS.from_atoms(ethanol, "oplsaa", cutoff=12.0)
   opls.topology.types                # ['opls_80', 'opls_99', 'opls_96', ...]

   dre = Dreiding.from_atoms(ethanol, "dreiding", cutoff=12.0)
   dre.topology.bond_orders           # DREIDING's rules read these

   reax = ReaxFF("CHO_cho_2008")      # ReaxFF needs no typing

The pieces are available on their own:

.. code-block:: python

   from xnn.ffnn.common import read_forcefield, assign_atom_types
   from xnn.ffnn.models import MolecularTopology

   ff = read_forcefield("oplsaa")
   types = assign_atom_types(ethanol, ff)
   top = MolecularTopology.from_ase(ethanol, forcefield="oplsaa")

   ff.charge("opls_80")                      # -0.18
   ff.nonbond("opls_80")                     # (3.5, 0.066): sigma / A, eps / kcal/mol
   ff.bond("opls_80", "opls_81")             # ('quadratic_bond', key, Row)
   ff.templates["opls_80"]["smarts"]         # ['[CD4H3:1][#6]', ...]

Units are the format's defaults (kcal/mol, Å, degrees); ``@units`` and
``@type`` modifiers in a file are converted on reading.

Write a force field
===================
Trained parameters go back out in the same format:

.. code-block:: python

   lib = opls.ff.export_library()
   lib.save_frc("my_opls.frc", name="my-opls")
   OPLS.from_atoms(ethanol, "my_opls.frc")        # and straight back in

   reax.export_library().save("ffield.json")      # ReaxFF-nn weights: JSON only

The written file carries the templates. A dihedral ``V0`` constant has no
column in the format and is dropped (it changes neither forces nor energy
differences); ReaxFF-nn network weights have no place in it and stay in JSON.

The format in brief
===================
A file opens with ``!MolSSI forcefield 1`` (dialect and format version).
Sections start at a ``#`` line and run to the next one::

   #quadratic_bond oplsaa          <- kind, label

   > E = K2 * (R - R0)^2           <- annotation
   @units K2 kJ/mol/nm^2           <- modifier (optional)

   !Version    Ref  I        J        R0      K2     <- column header
   2023.01.29  1    opls_18  opls_18  1.5290  268.00 <- data row

``#define <name>`` lists, per functional form, the labelled sections that
make up a variant; later labels override earlier ones, which is how ``CL&P``
and ``lopls`` extend ``oplsaa``. ``#include <file>`` splices another file in,
``#templates`` holds the SMARTS JSON, ``#reference`` the provenance. Within a
section the newest ``Version`` of each key wins.

To add a force field, write its parameters as a ``.frc`` file with a
``#templates`` section (unknown section kinds are read from their column
header; :func:`~xnn.ffnn.common.frc.register_section_schema` adds key
symmetry and unit rules) and a small bridge from
:class:`~xnn.ffnn.common.frc.ForceField` to the model's tables, as
:func:`xnn.ffnn.models.oplslib.from_forcefield` does. Files under
``src/xnn/ffnn/data/`` are found by name.
