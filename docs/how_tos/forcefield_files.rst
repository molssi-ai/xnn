.. _howto-forcefield-files:

******************************
Working with Force-field Files
******************************

The classical force-fields such as ReaxFF, OPLS, DREIDING read their parameters
from the MolSSI/SEAMM ``.frc`` file format (see :mod:`xnn.ffnn.common.frc`),
developed by MolSSI and used in `SEAMM force-field distribution
<https://github.com/molssi-seamm/forcefield_step/tree/main/forcefield_step/data>`_.
A ``.frc`` file also carries SMARTS templates that assign its atom types to the
molecules. Thus, any fixed-topology field can be adopted without knowing its type
names.

Force-field files shipped with xnn
==================================

You can list the force-field files that are available in your xnn installation:

.. code-block:: python

   from xnn.ffnn.common import list_forcefields

   sorted(list_forcefields())
   # ['CL&P', 'dreiding', 'dreiding/X6', 'ff94', 'lopls', 'oplsaa', 'oplsaa+',
   #  'oplsaa-1996', 'reaxff/CHLiOFSi_Yun_2017', ..., 'reaxff/CHO_cho_2008', ...]

Among the force-field files shipped with xnn, ``oplsaa`` is the OPLS-AA
distribution (``CL&P`` adds the ionic-liquid extension, ``oplsaa+`` is their
union), ``dreiding`` and ``dreiding/X6`` provide the two DREIDING nonbond forms,
and a dozen published ReaxFF force-field variants live under ``reaxff/`` folder.
All force-field files are copied verbatim from `SEAMM
<https://github.com/molssi-seamm/forcefield_step/tree/main/forcefield_step/data>`_
(BSD-3-Clause). xnn adds ``lopls`` (`Siu et al. (2012)
<https://doi.org/10.1021/ct200908r>`_) and ``oplsaa-1996`` (the torsions of `the
original paper <https://doi.org/10.1021/ja9621760>`_), which ``#include``
``oplsaa.frc`` and override a few rows, and the electrolyte-water ReaxFF field
of `Fedkin et al. (2019) <https://doi.org/10.1021/acs.jpca.8b10453>`_ as,
``reaxff/CHOLiNaKCsFClI_Fedkin_2019``. ``ff94`` (``amber.frc``) is the AMBER
force field of `Cornell et al. (1995) <https://doi.org/10.1021/ja00124a002>`_,
converted from the public-domain AMBER parameter and residue files: its
torsions use the multi-term ``torsion_fourier`` section, its impropers the
``improper_fourier`` section (the key order carries meaning, the central atom
third), and its residues are ``#fragments`` whose SMARTS carry atom maps in
template order together with ``atom names``, ``atom types`` and ``charges``.

Every model and reader takes the same **spec**: a variant name, a path to a
``.frc`` file, or ``"<path>.frc:<variant>"``.

Apply a force field to a molecule
=================================
In order to be able to apply a force field to molecules, xnn uses RDKit
(requires the ``ffnn`` extra dependency installed) to perceive the bonds and the
templates to assign the types. Here, we show how to apply the OPLS-AA, DREIDING,
and ReaxFF force fields to an ethanol molecule to get the atom types and the
parameters.

.. code-block:: python

   from ase.build import molecule
   from xnn.ffnn.models import OPLS, ReaxFF, Dreiding, Amber

   ethanol = molecule("CH3CH2OH")
   opls = OPLS.from_atoms(ethanol, "oplsaa", cutoff=12.0)
   opls.topology.types                # ['opls_80', 'opls_99', 'opls_96', ...]

   dre = Dreiding.from_atoms(ethanol, "dreiding", cutoff=12.0)
   dre.topology.bond_orders           # DREIDING's rules read these

   amb = Amber.from_atoms(dipeptide, "ff94", cutoff=12.0)
   amb.topology.residue_names         # ['ACE', 'ALA', 'NME']: residue templates typed it
   amb.topology.charges               # and gave every atom its RESP charge

   reax = ReaxFF("CHO_cho_2008")      # ReaxFF needs no typing

Note that xnn allows accessing the force-field parameters directly:

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

Units are the MolSSI's format's defaults (kcal/mol, Å, degrees). The ``@units``
and ``@type`` modifiers in a force-field file are converted to their
corresponding values whenever xnn reads them.

Write a force field file
========================
After training a force-field, you can write its parameters to a force-field
file in the standard format.

.. code-block:: python

   # Write the OPLS-AA parameters of ethanol to a file
   lib = opls.ff.export_library()
   lib.save_frc("my_opls.frc", name="my-opls")

   # Read the force field file back in and apply it to ethanol
   OPLS.from_atoms(ethanol, "my_opls.frc")

   from xnn.ffnn.models.ffield import write_ffield

   # Write the ReaxFF parameters to a JSON file
   reax.export_library().save("ffield.json")

   # Write the ReaxFF parameters to a LAMMPS ``ffield`` file
   write_ffield(reax.export_library(), "ffield.refit")

The written ``.frc`` file carries the templates. A dihedral ``V0`` constant has
no column in the format and is dropped (it changes neither forces nor energy
differences). The ReaxFF-nn's network weights have no place in a force-field
file and stay in JSON. The ``write_ffield()`` writes in the ``ffield`` text
format of LAMMPS's ``pair_style reaxff``. So, a ReaxFF field refit with
``conventions="lammps"`` (the default for classical fields) can run in LAMMPS
without any changes.

The format in brief
===================
A standard force-field file opens with ``!MolSSI forcefield 1`` (dialect and
format version). Sections start at a ``#`` line and run to the next one

.. code-block:: text

   #quadratic_bond oplsaa          <- kind, label

   > E = K2 * (R - R0)^2           <- annotation
   @units K2 kJ/mol/nm^2           <- modifier (optional)

   !Version    Ref  I        J        R0      K2     <- column header
   2023.01.29  1    opls_18  opls_18  1.5290  268.00 <- data row

``#define <name>`` lists, per functional form, the labelled sections that make
up a variant. Later labels override earlier ones, which is how ``CL&P`` and
``lopls`` extend ``oplsaa``. The ``#include <file>`` entry includes another file
in, ``#templates`` holds the SMARTS JSON, ``#reference`` provides a way for
managing the provenance. Within a section, the newest ``Version`` of each key
wins.

To add a force-field file to xnn, write its parameters as a ``.frc`` file with a
``#templates`` section (unknown section kinds are read from their column header)
The :func:`~xnn.ffnn.common.frc.register_section_schema` function adds key
symmetry and unit rules. Then, create a small bridge from
:class:`~xnn.ffnn.common.frc.ForceField` to the model's tables, as
:func:`xnn.ffnn.models.oplslib.from_forcefield` does. Then, put the force-field
file under ``src/xnn/ffnn/data/``. The force-field files are automatically found
by xnn using their names.
