"""Atom typing from ``.frc`` SMARTS templates (:mod:`xnn.ffnn.common.typing`).

Needs RDKit (the ``ffnn`` extra); skipped without it. The shipped OPLS-AA
templates are applied to small molecules built from SMILES and from bare
coordinates (bond perception), and a synthetic force field checks the
precedence rules: fragments first and final, later templates override
earlier ones, every mapped atom of a pattern is typed, untyped atoms are an
error.
"""
import numpy as np
import pytest

pytest.importorskip("rdkit")

from xnn.ffnn.common import (assign_atom_types, to_rdkit, perceive_bonds,
                              AtomTypingError, read_frc, read_forcefield)
from xnn.ffnn.models import OPLS, MolecularTopology, read_opls


def _embedded(smiles, seed=1):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return (mol.GetConformer().GetPositions(),
            [a.GetAtomicNum() for a in mol.GetAtoms()])


@pytest.mark.parametrize("smiles, expected", [
    ("CCCC", {"opls_80", "opls_81", "opls_85"}),
    ("CCO", {"opls_80", "opls_99", "opls_96", "opls_85", "opls_97"}),
    ("C=C", {"opls_88", "opls_89"}),
    ("c1ccccc1", {"opls_90", "opls_91"}),
    ("CO", {"opls_99", "opls_96", "opls_98", "opls_97"}),
])
def test_oplsaa_types_from_smiles_and_from_coordinates(smiles, expected):
    ff = read_forcefield("oplsaa")
    types = assign_atom_types(smiles, ff)
    assert set(types) == expected
    # the same molecule from bare coordinates: bonds are perceived
    pos, z = _embedded(smiles)
    types2, mol = assign_atom_types((pos, z), ff, return_mol=True)
    assert types2 == types
    assert len(perceive_bonds(mol)) == mol.GetNumBonds()


def test_butane_typing_matches_hand_assignment():
    ff = read_forcefield("oplsaa")
    types = assign_atom_types("CCCC", ff)
    assert types[:4] == ["opls_80", "opls_81", "opls_81", "opls_80"]
    assert types[4:] == ["opls_85"] * 10


def test_library_object_and_spec_are_accepted():
    lib = read_opls("oplsaa")
    assert assign_atom_types("C", lib) == ["opls_83"] + ["opls_85"] * 4
    assert assign_atom_types("C", "oplsaa") == ["opls_83"] + ["opls_85"] * 4


def test_given_bonds_are_kept():
    pos, z = _embedded("C=C")
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    types, mol = assign_atom_types((pos, z), "oplsaa", bonds=bonds,
                                   return_mol=True)
    assert perceive_bonds(mol) == bonds
    assert mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2.0


def test_lopls_overrides_alkane_types_only():
    types = assign_atom_types("CCCCCC", "lopls")
    assert set(types) == {"lopls_CT_CH3", "lopls_CT_CH2", "lopls_HC_CH3",
                          "lopls_HC_CH2"}
    # ethanol's CH2 sits next to an oxygen and keeps its OPLS-AA type
    types = assign_atom_types("CCO", "lopls")
    assert "opls_99" in types and "lopls_CT_CH3" in types


SYNTH = """!MolSSI forcefield 1
#define s
!V R F L
1.0 1 atom_types s
1.0 1 templates s
1.0 1 fragments s
#end
#atom_types s
!Version Ref Type Mass El connections Comment
1.0 1 c   12.011 C 4 carbon
1.0 1 cm  12.011 C 4 methyl carbon
1.0 1 h    1.008 H 1 hydrogen
1.0 1 hw   1.008 H 1 water hydrogen
1.0 1 ow  15.999 O 2 water oxygen
#templates s
{
 "c":  {"1.0": {"smarts": ["[#6:1]"], "description": "any C", "overrides": []}},
 "h":  {"1.0": {"smarts": ["[H:1][#6]"], "description": "H on C", "overrides": []}},
 "cm": {"1.0": {"smarts": ["[CH3:1]"], "description": "methyl, listed later so it wins", "overrides": []}},
 "hw": {"1.0": {"smarts": ["[H:1][O][H:2]"], "description": "both water H at once", "overrides": []}}
}
#fragments s
{
 "water": {"1.0": {"name": "water", "SMILES": "O", "SMARTS": "[OX2H2]([H])[H]",
           "atom types": ["ow", "hw", "hw"]}}
}
#end
"""


def test_precedence_rules(tmp_path):
    path = tmp_path / "s.frc"
    path.write_text(SYNTH)
    ff = read_frc(path).forcefield()
    # later template wins over the earlier generic one; multi-map patterns
    # type every mapped atom
    assert assign_atom_types("CC", ff) == ["cm", "cm"] + ["h"] * 6
    assert assign_atom_types("CCC", ff) == ["cm", "c", "cm"] + ["h"] * 8
    # the fragment types water and is final (the "hw" template would agree
    # anyway; the oxygen has no template at all and still gets typed)
    assert assign_atom_types("O", ff) == ["ow", "hw", "hw"]
    # an atom no template covers is an error naming it
    with pytest.raises(AtomTypingError, match="no atom type"):
        assign_atom_types("N", ff)
    # a force field without templates cannot type
    ff.templates.clear()
    ff.fragments.clear()
    with pytest.raises(ValueError, match="no #templates"):
        assign_atom_types("C", ff)


def test_to_rdkit_inputs():
    from rdkit import Chem
    pos, z = _embedded("CO")
    m1 = to_rdkit((pos, z))
    m2 = to_rdkit({"pos": pos, "atomic_numbers": z})
    m3 = to_rdkit(positions=pos, atomic_numbers=z)
    assert m1.GetNumBonds() == m2.GetNumBonds() == m3.GetNumBonds() == 5
    assert to_rdkit("CO").GetNumAtoms() == 6
    assert to_rdkit(Chem.MolFromSmiles("CO")).GetNumAtoms() == 6   # Hs added
    with pytest.raises(TypeError):
        to_rdkit(42)
    with pytest.raises(ValueError):
        to_rdkit("not a smiles ((")


def test_from_atoms_and_topology_from_ase_type_the_structure():
    ase = pytest.importorskip("ase")
    pos, z = _embedded("C=C")
    atoms = ase.Atoms(numbers=z, positions=pos)
    top = MolecularTopology.from_ase(atoms, forcefield="oplsaa")
    assert set(top.types) == {"opls_88", "opls_89"} and len(top.bonds) == 5
    model = OPLS.from_atoms(atoms, "oplsaa", cutoff=20.0)
    assert model.topology.types == top.types
    # two trigonal centers, both matching the X-X-opls_86-X improper pattern
    assert len(model.impropers) == 2
    with pytest.raises(ValueError):
        MolecularTopology.from_ase(atoms)          # neither types nor forcefield
