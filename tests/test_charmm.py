"""CHARMM force-field tests: equations, library, typing, invariances, training.

Every energy term is recomputed by hand from the published functional form
(MacKerell et al., J. Phys. Chem. B 102, 3586, 1998, eq 1) and compared
against the model's intermediates, the shipped ``charmm22`` library is
spot-checked against the CHARMM22 parameter and topology files, the 1,4
bookkeeping is checked on small rings, the impropers come from the residue
templates, and float32 is held against float64. The LAMMPS and NAMD
cross-checks live in
``notebooks/fidelity_checks/ffnn/charmm/charmm_verification.ipynb``.
"""

import math

import numpy as np
import pytest
import torch

from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models
from xnn.common.train import weighted_loss
from xnn.ffnn.common import list_forcefields, find_forcefield, read_forcefield
from xnn.ffnn.models import (
    Charmm,
    CharmmForceField,
    CharmmLibrary,
    MolecularTopology,
    builtin_charmm,
    read_charmm,
    read_topology,
    place_charmm_impropers,
)
from xnn.ffnn.models.charmmlib import (
    COULOMB_CONSTANT,
    KCAL_TO_EV,
    resolve_improper_type,
)

EV_TO_KCAL = 1.0 / KCAL_TO_EV
DEG = math.pi / 180.0


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# small builders
def _graph(pos, z, cutoff, cell=None):
    """Build a single-structure graph from raw arrays."""
    s = {
        "pos": torch.as_tensor(pos, dtype=torch.get_default_dtype()),
        "atomic_numbers": torch.as_tensor(z, dtype=torch.long),
    }
    if cell is not None:
        s["cell"] = torch.as_tensor(cell, dtype=torch.get_default_dtype())
        s["pbc"] = torch.tensor([True, True, True])
    return structure_to_graph(s, cutoff=cutoff)


def _ethane(phi_deg=47.0, r_cc=1.53, r_ch=1.111, ang_deg=110.0):
    """Ethane with a prescribed H-C-C-H dihedral, hand-typed CT3 / HA."""
    ang = math.radians(ang_deg)
    pos = [[0.0, 0.0, 0.0], [r_cc, 0.0, 0.0]]
    s, c = r_ch * math.sin(math.pi - ang), r_ch * math.cos(math.pi - ang)
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        pos.append([-c, s * math.cos(a), s * math.sin(a)])
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0 + math.radians(phi_deg)
        pos.append([r_cc + c, s * math.cos(a), s * math.sin(a)])
    types = ["CT3", "CT3"] + ["HA"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    return pos, [6, 6] + [1] * 6, types, bonds


def _ethane_model(**kwargs):
    pos, z, types, bonds = _ethane()
    top = MolecularTopology.from_bonds(types, bonds)
    return Charmm("charmm22", top, cutoff=kwargs.pop("cutoff", 20.0), **kwargs), pos, z


def _nma():
    """N-methylacetamide, hand-typed as the ACE and CT3 groups with their charges
    and the two impropers of the templates (central atom first)."""
    pos = [
        [0.0, 0.0, 0.0],  # CAY
        [1.51, 0.0, 0.0],  # CY
        [2.13, 1.07, 0.0],  # OY
        [2.14, -1.19, 0.0],  # NT
        [3.59, -1.28, 0.0],  # CAT
        [1.60, -2.05, 0.0],  # HNT
        [-0.38, 1.02, 0.0],
        [-0.38, -0.51, 0.89],
        [-0.38, -0.51, -0.89],
        [3.95, -0.76, 0.89],
        [3.95, -0.76, -0.89],
        [3.95, -2.31, 0.0],
    ]
    z = [6, 6, 8, 7, 6, 1, 1, 1, 1, 1, 1, 1]
    types = ["CT3", "C", "O", "NH1", "CT3", "H", "HA", "HA", "HA", "HA", "HA", "HA"]
    bonds = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (0, 6), (0, 7), (0, 8),
             (4, 9), (4, 10), (4, 11)]
    charges = [-0.27, 0.51, -0.51, -0.47, -0.11, 0.31, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]
    impropers = [(1, 0, 3, 2), (3, 1, 4, 5)]
    return pos, z, types, bonds, charges, impropers


def _nma_topology():
    pos, z, types, bonds, charges, impropers = _nma()
    return MolecularTopology.from_bonds(types, bonds, charges=charges, impropers=impropers)


def _chain4(phi_deg, r=1.5, ang_deg=109.5):
    """Four atoms with a prescribed signed dihedral (IUPAC sign)."""
    ang = math.radians(ang_deg)
    phi = math.radians(phi_deg)
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([r, 0.0, 0.0])
    p2 = p1 + r * torch.tensor([-math.cos(ang), math.sin(ang), 0.0])
    e2 = (p2 - p1) / (p2 - p1).norm()
    n = torch.linalg.cross(p1 - p0, p2 - p1)
    n = n / n.norm()
    m = torch.linalg.cross(n, e2)
    p3 = (p2 - r * math.cos(ang) * e2
          + r * math.sin(ang) * (math.cos(phi) * m - math.sin(phi) * n))
    return torch.stack([p0, p1, p2, p3])


def _signed_dihedral(p):
    """The dihedral angle in degrees by the atan2 formula (IUPAC sign)."""
    p = np.asarray(p, dtype=float)
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.degrees(math.atan2(np.dot(m1, n2), np.dot(n1, n2)))


def _ring(n_c, r=1.53):
    """A planar cycloalkane ring of CT2 / HA atoms with two hydrogens per carbon."""
    pos, types, bonds = [], [], []
    for i in range(n_c):
        a = 2 * math.pi * i / n_c
        rad = r / (2 * math.sin(math.pi / n_c))
        c = [rad * math.cos(a), rad * math.sin(a), 0.0]
        pos.append(c)
        types.append("CT2")
    for i in range(n_c):
        bonds.append((i, (i + 1) % n_c))
    for i in range(n_c):
        a = 2 * math.pi * i / n_c
        rad = r / (2 * math.sin(math.pi / n_c))
        for s in (1.0, -1.0):
            pos.append([(rad + 0.6) * math.cos(a), (rad + 0.6) * math.sin(a), s * 0.9])
            types.append("HA")
            bonds.append((i, len(pos) - 1))
    return pos, [6] * n_c + [1] * (2 * n_c), types, bonds


def _embedded(smiles, seed=1):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


# registry and library
def test_registration():
    assert "charmm" in available_models()
    assert "charmm22" in list_forcefields()
    assert find_forcefield("charmm22")[1] == "charmm22"


def test_library_matches_the_charmm22_files():
    lib = builtin_charmm("charmm22")
    assert lib.coulomb_constant == COULOMB_CONSTANT and lib.scale_14_coulomb == 1.0
    # bonds (Kb, b0)
    assert lib.bond("CT3-CT3") == {"k": 222.5, "r0": 1.53}
    assert lib.bond("CT3-HA") == {"k": 322.0, "r0": 1.111}
    assert lib.bond("C-NH1") == {"k": 370.0, "r0": 1.345}
    assert lib.bond("H-NH1") == {"k": 440.0, "r0": 0.997}
    assert lib.bond("OT-HT") == {"k": 450.0, "r0": 0.9572}
    assert lib.bond("HT-HT") is None  # the SHAKE helper "bond" of TIP3P is not a bond
    # angles (Ktheta, theta0) and their Urey-Bradley terms (Kub, S0)
    assert lib.angle("HA-CT3-HA") == {"k": 35.5, "theta0": 108.4}
    assert lib.urey_bradley("HA-CT3-HA") == {"k": 5.4, "s0": 1.802}
    assert lib.angle("CT2-CT2-CT2") == {"k": 58.35, "theta0": 113.6}
    assert lib.urey_bradley("CT2-CT2-CT2") == {"k": 11.16, "s0": 2.561}
    assert lib.angle("O-C-NH1") == {"k": 80.0, "theta0": 122.5}
    assert lib.urey_bradley("O-C-NH1") is None
    assert lib.angle("HT-OT-HT") == {"k": 55.0, "theta0": 104.52}
    # dihedrals: K per dihedral, n, delta; several terms per quadruple
    assert lib.torsion("CT3-C-NH1-CT3")["terms"] == [[1.6, 1, 0.0], [2.5, 2, 180.0]]
    assert lib.torsion("X-CT3-CT3-X")["terms"] == [[0.155, 3, 0.0]]
    assert lib.torsion("CT2-CT2-CT2-CT2")["terms"] == [[0.15, 1, 0.0]]
    assert lib.torsion("O-C-NH1-H")["terms"] == [[2.5, 2, 180.0]]
    assert lib.torsion("H-NH1-C-O") is lib.torsion("O-C-NH1-H")
    # impropers keep the written order and match in either direction
    assert lib.improper("NH1-X-X-H") == {"k": 20.0, "psi0": 0.0}
    assert lib.improper("O-X-X-C") == {"k": 120.0, "psi0": 0.0}
    assert lib.improper("CPH2-NR2-NR1-HR1") == {"k": 0.5, "psi0": 0.0}
    assert "O-X-X-C" in lib.improper_types
    # Lennard-Jones Rmin/2, epsilon and the 1,4 values of the types that have them
    for t, rs, eps, rs14, eps14 in (
        ("C", 2.0, 0.11, None, None),
        ("CT3", 2.06, 0.08, 1.9, 0.01),
        ("CT1", 2.275, 0.02, 1.9, 0.01),
        ("NH1", 1.85, 0.2, 1.55, 0.2),
        ("O", 1.7, 0.12, 1.4, 0.12),
        ("H", 0.2245, 0.046, None, None),
        ("HA", 1.32, 0.022, None, None),
        ("OT", 1.7682, 0.1521, None, None),
        ("HT", 0.2245, 0.046, None, None),
        ("SOD", 1.41075, 0.0469, None, None),
        ("CLA", 2.27, 0.15, None, None),
    ):
        a = lib.atom_types[t]
        assert abs(a["rstar"] - rs) < 1e-12 and abs(a["epsilon"] - eps) < 1e-12, t
        if rs14 is None:
            assert a["rstar14"] is None and a["epsilon14"] is None, t
        else:
            assert abs(a["rstar14"] - rs14) < 1e-12 and abs(a["epsilon14"] - eps14) < 1e-12, t
    # the pair overrides (NBFIX) of the water / ion stream
    assert lib.nbfix("SOD-CLA") == {"rmin": 3.731, "eps": 0.0839, "rmin14": None, "eps14": None}
    assert lib.nbfix("SOD-OC")["rmin"] == 3.23 and lib.nbfix("POT-OC")["eps"] == 0.1022
    assert lib.atom_types["NH1"]["mass"] == 14.007 and lib.atom_types["NH1"]["element"] == 7
    assert lib.atom_types["SOD"]["element"] == 11 and lib.atom_types["CLA"]["element"] == 17
    assert lib.atom_types["CT3"]["mass"] == 12.011 and lib.atom_types["HA"]["mass"] == 1.008


def test_residue_templates_carry_charges_and_impropers():
    lib = builtin_charmm("charmm22")
    ala = lib.residue_charges("ALA")
    assert ala == {"N": -0.47, "HN": 0.31, "CA": 0.07, "HA": 0.09, "CB": -0.27,
                   "HB1": 0.09, "HB2": 0.09, "HB3": 0.09, "C": 0.51, "O": -0.51}
    assert lib.fragments["ALA"]["atom types"] == ["NH1", "H", "CT1", "HB", "CT3",
                                                   "HA", "HA", "HA", "C", "O"]
    assert lib.fragments["ALA"]["impropers"] == [["N", "-C", "CA", "HN"], ["C", "CA", "+N", "O"]]
    assert lib.residue_charges("ACE") == {"CAY": -0.27, "HY1": 0.09, "HY2": 0.09, "HY3": 0.09,
                                          "CY": 0.51, "OY": -0.51}
    assert lib.fragments["ACE"]["impropers"] == [["CY", "CAY", "+N", "OY"]]
    assert lib.residue_charges("CT3") == {"NT": -0.47, "HNT": 0.31, "CAT": -0.11, "HT1": 0.09,
                                          "HT2": 0.09, "HT3": 0.09}
    assert lib.fragments["CT3"]["impropers"] == [["NT", "-C", "CAT", "HNT"]]
    assert lib.residue_charges("TIP3") == {"OH2": -0.834, "H1": 0.417, "H2": 0.417}
    assert lib.residue_charges("SOD") == {"SOD": 1.0}
    # terminal and patched forms
    nter = lib.residue_charges("ALA:NTER")
    assert nter["N"] == -0.3 and nter["HT1"] == 0.33 and nter["CA"] == 0.21 and "HN" not in nter
    cter = lib.residue_charges("ALA:CTER")
    assert cter["C"] == 0.34 and cter["OT1"] == -0.67 and "O" not in cter
    assert lib.fragments["ALA:CTER"]["impropers"] == [["N", "-C", "CA", "HN"], ["C", "CA", "OT2", "OT1"]]
    assert lib.fragments["GLY:GLYP"]["atom types"][0] == "NH3"
    assert lib.fragments["PRO:PROP"]["atom types"][0] == "NP"
    assert "HG1" not in lib.residue_charges("CYS:DISU")
    # every template carries an integer total charge and consistent lists
    for name, frag in lib.fragments.items():
        total = sum(frag["charges"])
        assert abs(total - round(total)) < 1e-6, name
        assert len(frag["charges"]) == len(frag["atom types"]) == len(frag["atom names"])
        for quad in frag.get("impropers", []):
            assert len(quad) == 4 and quad[0] in frag["atom names"], (name, quad)
    assert {"ALA", "ARG", "HSD", "HSE", "HSP", "PRO", "GLY", "ACE", "CT3", "TIP3", "SOD", "CLA",
            "ALA:NTER", "ALA:CTER", "GLY:GLYP", "PRO:PROP", "CYS:DISU", "ASP:ASPP",
            "GLU:GLUP", "LYS:LSN"} <= set(lib.fragments)


# energy expressions (eq 1)
def test_bond_angle_urey_bradley_and_14_equations():
    model, pos, z = _ethane_model(keep_intermediates=True)
    out = model(_graph(pos, z, 20.0))
    lib = builtin_charmm("charmm22")
    P = np.asarray(pos)
    e_bond = 0.0
    for i, j in model.topology.bonds:
        r = np.linalg.norm(P[i] - P[j])
        p = lib.bond("CT3-CT3" if z[i] == z[j] == 6 else "CT3-HA")
        e_bond += p["k"] * (r - p["r0"]) ** 2
    assert abs(float(out["e_bond"]) * EV_TO_KCAL - e_bond) < 1e-10
    e_angle, e_ub = 0.0, 0.0
    for i, j, k in model.topology.angles:
        u, v = P[i] - P[j], P[k] - P[j]
        th = math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))
        key = "CT3-CT3-HA" if 6 in (z[i], z[k]) else "HA-CT3-HA"
        p = lib.angle(key)
        e_angle += p["k"] * (th - p["theta0"] * DEG) ** 2
        ub = lib.urey_bradley(key)
        if ub is not None:
            s = np.linalg.norm(P[i] - P[k])
            e_ub += ub["k"] * (s - ub["s0"]) ** 2
    assert abs(float(out["e_angle"]) * EV_TO_KCAL - e_angle) < 1e-10
    assert e_ub > 1e-6 and abs(float(out["e_urey_bradley"]) * EV_TO_KCAL - e_ub) < 1e-10
    # 1,2 and 1,3 pairs are excluded: no direct nonbonded energy at all
    assert abs(float(out["e_vdw"]) + float(out["e_coulomb"])) < 1e-14
    # the nine H...H 1,4 pairs use the regular HA parameters (HA has no 1,4 values)
    e14 = 0.0
    for i, j in model.topology.pairs14:
        r = np.linalg.norm(P[i] - P[j])
        x = (2 * 1.32 / r) ** 6
        e14 += 0.022 * (x * x - 2 * x)
    assert len(model.topology.pairs14) == 9
    assert abs(float(out["e_vdw14"]) * EV_TO_KCAL - e14) < 1e-12
    assert abs(float(out["e_coulomb14"])) < 1e-14  # no charges


def test_own_14_parameters_and_torsion_terms():
    lib = builtin_charmm("charmm22")
    # a CT3-C-NH1-CT3 chain: the two-term dihedral and the 1,4 pair CT3...CT3,
    # which uses the 1,4 values Rmin/2 = 1.9, eps = 0.01 instead of 2.06 / 0.08
    top = MolecularTopology.from_bonds(["CT3", "C", "NH1", "CT3"], [(0, 1), (1, 2), (2, 3)])
    model = Charmm(lib, top, cutoff=20.0)
    for phi in (0.0, 60.0, 100.0, 180.0, -70.0):
        pos = _chain4(phi)
        out = model(_graph(pos, [6, 6, 7, 6], 20.0))
        p = math.radians(phi)
        want = 1.6 * (1 + math.cos(p)) + 2.5 * (1 + math.cos(2 * p - math.pi))
        assert abs(float(out["e_torsion"]) * EV_TO_KCAL - want) < 1e-10
        r = float((pos[0] - pos[3]).norm())
        x = (2 * 1.9 / r) ** 6
        assert abs(float(out["e_vdw14"]) * EV_TO_KCAL - 0.01 * (x * x - 2 * x)) < 1e-10
    # a phase of 90 degrees makes the term odd in phi: the signed dihedral
    # enters through sin(n phi), checked against the atan2 definition
    lib.torsion("CT3-C-NH1-CT3")["terms"] = [[1.0, 1, 90.0], [0.5, 3, 30.0]]
    model = Charmm(lib, top, cutoff=20.0)
    for phi in (35.0, -35.0, 120.0, -170.0):
        pos = _chain4(phi)
        assert abs(_signed_dihedral(pos) - phi) < 1e-9
        out = model(_graph(pos, [6, 6, 7, 6], 20.0))
        p = math.radians(phi)
        want = 1.0 * (1 + math.cos(p - math.pi / 2)) + 0.5 * (1 + math.cos(3 * p - math.pi / 6))
        assert abs(float(out["e_torsion"]) * EV_TO_KCAL - want) < 1e-10


def test_improper_equation_and_wrapping():
    lib = builtin_charmm("charmm22")
    # a carbonyl center: C bonded to CT3, NH1 and O; the template order
    # C-CA-N-O matches the row O-X-X-C reversed
    pos = [[0.0, 0.0, 0.0], [-1.5, 0.3, 0.0], [0.6, 1.1, 0.25], [0.7, -1.2, -0.2]]
    z = [6, 6, 8, 7]
    top = MolecularTopology.from_bonds(["C", "CT3", "O", "NH1"], [(0, 1), (0, 2), (0, 3)],
                                       impropers=[(0, 1, 3, 2)])
    model = Charmm(lib, top, cutoff=20.0, keep_intermediates=True)
    assert model.impropers == [(0, 1, 3, 2)]
    out = model(_graph(pos, z, 20.0))
    phi = math.radians(_signed_dihedral([pos[0], pos[1], pos[3], pos[2]]))
    assert abs(float(out["e_improper"]) * EV_TO_KCAL - 120.0 * phi * phi) < 1e-10
    assert float(out["e_improper"]) > 1e-4
    assert resolve_improper_type(lib.improper_types, "C", "CT3", "NH1", "O") == "O-X-X-C"
    assert resolve_improper_type(lib.improper_types, "NH1", "C", "CT1", "H") == "NH1-X-X-H"
    assert resolve_improper_type(lib.improper_types, "CPH2", "ND1", "NE2", "HR1") is None
    assert resolve_improper_type(lib.improper_types, "CPH2", "NR2", "NR1", "HR1") == "HR1-NR1-NR2-CPH2"
    # a non-zero reference angle: the difference is wrapped into (-180, 180]
    lib.improper_types["O-X-X-C"] = {"k": 10.0, "psi0": 170.0}
    model = Charmm(lib, top, cutoff=20.0)
    for phi_deg in (175.0, -175.0, 10.0, -100.0):
        c = _chain4(phi_deg)                      # the chain runs over atoms 0, 1, 3, 2
        p = torch.stack([c[0], c[1], c[3], c[2]])
        out = model(_graph(p, z, 20.0))
        d = (phi_deg - 170.0 + 180.0) % 360.0 - 180.0
        assert abs(float(out["e_improper"]) * EV_TO_KCAL - 10.0 * math.radians(d) ** 2) < 1e-10


def test_nonbond_tables_combination_and_overrides():
    lib = builtin_charmm("charmm22")
    top = MolecularTopology.from_bonds(["C", "NH1"], [], charges=[0.5, -0.5])
    model = Charmm(lib, top, cutoff=20.0)
    for r in (2.8, 3.5, 5.0):
        out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 7], 20.0))
        rmin = 2.0 + 1.85
        eps = math.sqrt(0.11 * 0.2)
        x = (rmin / r) ** 6
        assert abs(float(out["e_vdw"]) * EV_TO_KCAL - eps * (x * x - 2 * x)) < 1e-10
        assert abs(float(out["e_coulomb"]) * EV_TO_KCAL + COULOMB_CONSTANT * 0.25 / r) < 1e-10
    # the pair override replaces the combination rule for Na+ / Cl-
    top = MolecularTopology.from_bonds(["SOD", "CLA"], [], charges=[1.0, -1.0])
    model = Charmm(lib, top, cutoff=20.0, coulomb_constant=332.0636)
    out = model(_graph([[0, 0, 0], [3.0, 0, 0]], [11, 17], 20.0))
    x = (3.731 / 3.0) ** 6
    assert abs(float(out["e_vdw"]) * EV_TO_KCAL - 0.0839 * (x * x - 2 * x)) < 1e-10
    assert abs(float(out["e_coulomb"]) * EV_TO_KCAL + 332.0636 / 3.0) < 1e-10
    # the override tables are symmetric and the well sits at rmin
    rmin_tab, eps_tab, rmin14_tab, eps14_tab = model.ff.pair_tables()
    i, j = model.ff.type_index("SOD"), model.ff.type_index("CLA")
    assert float(rmin_tab[i, j]) == float(rmin_tab[j, i]) == 3.731
    assert abs(float(eps14_tab[i, j]) * EV_TO_KCAL - 0.0839) < 1e-12
    k = model.ff.type_index("NH1")
    assert abs(float(rmin14_tab[k, k]) - 3.1) < 1e-12 and abs(float(rmin_tab[k, k]) - 3.7) < 1e-12


def test_14_bookkeeping_in_small_rings():
    lib = builtin_charmm("charmm22")
    for n_c, para_pairs in ((6, 3), (5, 0), (4, 0)):
        pos, z, types, bonds = _ring(n_c)
        top = MolecularTopology.from_bonds(types, bonds)
        carbons = set(range(n_c))
        cc14 = [p for p in top.pairs14 if set(p) <= carbons]
        assert len(cc14) == para_pairs, n_c
        # every 1,4 pair is listed once, never also as a 1,2 or 1,3 pair
        assert len(set(top.pairs14)) == len(top.pairs14)
        assert not set(top.pairs14) & set(top.exclusions)
        # per-dihedral bookkeeping would count the pairs reached by two paths
        # twice and add pairs that are really 1,2 or 1,3
        per_dihedral = [(min(i, l), max(i, l)) for i, j, k, l in top.dihedrals]
        ring_dihedrals = [p for p in per_dihedral if set(p) <= carbons]
        assert len(ring_dihedrals) == n_c
        model = Charmm(lib, top, cutoff=20.0)
        out = model(_graph(pos, z, 20.0))
        P = np.asarray(pos)
        e14 = 0.0
        for i, j in top.pairs14:
            r = np.linalg.norm(P[i] - P[j])
            a, b = lib.atom_types[types[i]], lib.atom_types[types[j]]
            rs = (a["rstar14"] or a["rstar"]) + (b["rstar14"] or b["rstar"])
            eps = math.sqrt((a["epsilon14"] or a["epsilon"]) * (b["epsilon14"] or b["epsilon"]))
            x = (rs / r) ** 6
            e14 += eps * (x * x - 2 * x)
        assert abs(float(out["e_vdw14"]) * EV_TO_KCAL - e14) < 1e-10


def test_impropers_come_from_the_residue_templates():
    pytest.importorskip("rdkit")
    lib = builtin_charmm("charmm22")
    model = Charmm.from_atoms(_embedded("CC(=O)N[C@@H](C)C(=O)NC"), lib, cutoff=20.0)
    top = model.topology
    assert top.residue_names == ["ACE", "ALA", "CT3"]
    names = {a: top.residue_names[r[0]] + ":" + lib.fragments[top.residue_names[r[0]]]["atom names"][r[1]]
             for a, r in enumerate(top.residues)}
    got = sorted(tuple(names[a] for a in im) for im in model.impropers)
    assert got == sorted([
        ("ACE:CY", "ACE:CAY", "ALA:N", "ACE:OY"),
        ("ALA:N", "ACE:CY", "ALA:CA", "ALA:HN"),
        ("ALA:C", "ALA:CA", "CT3:NT", "ALA:O"),
        ("CT3:NT", "ALA:C", "CT3:CAT", "CT3:HNT"),
    ])
    # the same quadruples by hand through place_impropers
    elements = [lib.atom_types[t]["element"] for t in top.types]
    from xnn.common.data.elements import CHEMICAL_SYMBOLS
    syms = [CHEMICAL_SYMBOLS[z] for z in elements]
    assert sorted(place_charmm_impropers(top, lib.fragments, syms)) == sorted(model.impropers)
    # a chain end drops the improper that needs the missing neighbour: a
    # single ALA with both termini has no "-C" and no "+N"
    zw = Charmm.from_atoms(_embedded("[NH3+][C@@H](C)C(=O)[O-]"), lib, cutoff=20.0, charge=0)
    assert zw.topology.residue_names == ["ALA:NTER:CTER"]
    assert len(zw.impropers) == 1  # C CA OT2 OT1
    # explicit impropers are used as given
    pos, z, types, bonds, charges, impropers = _nma()
    m = Charmm(lib, MolecularTopology.from_bonds(types, bonds, impropers=impropers), cutoff=20.0)
    assert m.impropers == impropers
    assert Charmm(lib, MolecularTopology.from_bonds(types, bonds), cutoff=20.0).impropers == []


# typing
def test_from_atoms_types_residues_and_charges():
    pytest.importorskip("rdkit")
    lib = builtin_charmm("charmm22")
    model = Charmm.from_atoms(_embedded("CC(=O)N[C@@H](C)C(=O)NC"), lib, cutoff=20.0)
    top = model.topology
    assert len(top.types) == 22
    assert sorted(set(top.types)) == ["C", "CT1", "CT3", "H", "HA", "HB", "NH1", "O"]
    assert abs(sum(top.charges)) < 1e-6
    for a, r in enumerate(top.residues):
        frag = lib.fragments[top.residue_names[r[0]]]
        assert top.types[a] == frag["atom types"][r[1]]
        assert top.charges[a] == frag["charges"][r[1]]
    out = ForceStressOutput(model)(
        _graph(_embedded("CC(=O)N[C@@H](C)C(=O)NC").GetConformer().GetPositions(),
               [a.GetAtomicNum() for a in _embedded("CC(=O)N[C@@H](C)C(=O)NC").GetAtoms()], 20.0))
    assert bool(torch.isfinite(out["energy"]).all()) and bool(torch.isfinite(out["forces"]).all())
    m2 = Charmm.from_atoms(_embedded("CC(=O)N[C@@H](C)C(=O)NC"), lib, cutoff=20.0, charges=[0.0] * 22)
    assert float(m2.charges.abs().sum()) == 0.0
    # the histidine tautomers and a charged pair
    for smiles, residue, charge in (
        ("CC(=O)N[C@@H](Cc1c[nH]cn1)C(=O)NC", "HSE", 0),
        ("CC(=O)N[C@@H](Cc1cnc[nH]1)C(=O)NC", "HSD", 0),
        ("CC(=O)N[C@@H](Cc1c[nH]c[nH+]1)C(=O)NC", "HSP", 1),
        ("CC(=O)N[C@@H](CCCNC(=[NH2+])N)C(=O)N[C@@H](CC(=O)[O-])C(=O)NC", "ARG", 0),
    ):
        m = Charmm.from_atoms(_embedded(smiles), lib, cutoff=20.0, charge=charge)
        assert residue in m.topology.residue_names, smiles
        assert abs(sum(m.topology.charges) - charge) < 1e-6
        assert len(m.impropers) >= 4


def test_from_atoms_small_molecules_use_the_type_templates():
    pytest.importorskip("rdkit")
    cases = {
        "CC": {"CT3", "HA"},
        "CCC": {"CT3", "CT2", "HA"},
        "CC(C)C": {"CT3", "CT1", "HA"},
        "c1ccccc1": {"CA", "HP"},
        "Cc1ccccc1": {"CA", "HP", "CT3", "HA"},
        "C=C": {"CE2", "HE2"},
        "CC(=O)NC": {"CT3", "C", "O", "NH1", "H", "HA"},
        "O": {"OT", "HT"},
        "[Na+]": {"SOD"},
        "[Cl-]": {"CLA"},
    }
    for smiles, expect in cases.items():
        model = Charmm.from_atoms(_embedded(smiles), "charmm22", cutoff=20.0,
                                  charge=-1 if "-]" in smiles else (1 if "+]" in smiles else 0))
        assert set(model.topology.types) == expect, smiles
        if smiles == "CC(=O)NC":
            assert model.topology.residue_names == ["ACE", "CT3"]
            assert abs(sum(model.topology.charges)) < 1e-6
            assert len(model.impropers) == 2
        elif smiles not in ("O", "[Na+]", "[Cl-]"):
            assert not model.topology.residues
            assert float(model.charges.abs().sum()) == 0.0
    water = Charmm.from_atoms(_embedded("O"), "charmm22", cutoff=20.0)
    assert water.topology.residue_names == ["TIP3"]
    assert sorted(water.topology.charges) == [-0.834, 0.417, 0.417]
    assert water.topology.bonds == [(0, 1), (0, 2)]


# precision: float32 against float64
def test_float32_matches_float64():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model64 = Charmm("charmm22", top, cutoff=20.0)
    out64 = ForceStressOutput(model64)(_graph(pos, z, 20.0))
    torch.set_default_dtype(torch.float32)
    try:
        model32 = Charmm("charmm22", top, cutoff=20.0)
        out32 = ForceStressOutput(model32)(_graph(pos, z, 20.0))
    finally:
        torch.set_default_dtype(torch.float64)
    assert out32["node_energy"].dtype == torch.float32
    for key in ("e_bond", "e_angle", "e_urey_bradley", "e_torsion", "e_improper", "e_vdw",
                "e_coulomb", "e_vdw14", "e_coulomb14", "energy"):
        a, b = float(out64[key]), float(out32[key])
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), key
    assert float((out64["forces"] - out32["forces"].to(torch.float64)).abs().max()) < 2e-4
    # the fixed phases and reference angles are exact in both precisions
    assert float(model32.ff.dihedral_sin.abs().max()) == 0.0
    assert set(model32.ff.dihedral_cos.unique().tolist()) <= {-1.0, 1.0}
    assert float(model32.ff.improper_sin.abs().max()) == 0.0


# invariances, forces, batching, periodicity
def test_rotation_translation_invariance_and_force_equivariance():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    fmodel = ForceStressOutput(Charmm("charmm22", top, cutoff=20.0))
    out = fmodel(_graph(pos, z, 20.0))
    e0, f0 = out["energy"], out["forces"]
    ang = 0.6
    R = torch.tensor([[math.cos(ang), -math.sin(ang), 0.0],
                      [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    pos2 = torch.as_tensor(pos) @ R.T + torch.tensor([3.0, -2.0, 1.0])
    out2 = fmodel(_graph(pos2, z, 20.0))
    assert float((out2["energy"] - e0).abs()) < 1e-9
    assert float((out2["forces"] - f0 @ R.T).abs().max()) < 1e-8
    # mirror image: every term is even in the dihedral and improper angles
    out3 = fmodel(_graph(torch.as_tensor(pos) * torch.tensor([1.0, 1.0, -1.0]), z, 20.0))
    assert float((out3["energy"] - e0).abs()) < 1e-9


def test_forces_match_finite_differences():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", top, cutoff=20.0)
    forces = ForceStressOutput(model)(_graph(pos, z, 20.0))["forces"]
    d = 1e-5
    for atom, comp in ((0, 0), (2, 1), (3, 2), (5, 0), (9, 1)):
        pp = [list(p) for p in pos]
        pp[atom][comp] += d
        ep = float(model(_graph(pp, z, 20.0))["energy"])
        pp[atom][comp] -= 2 * d
        em = float(model(_graph(pp, z, 20.0))["energy"])
        assert abs(-(ep - em) / (2 * d) - float(forces[atom, comp])) < 1e-6


def test_batching_matches_individual_evaluation():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", top, cutoff=20.0)
    torch.manual_seed(0)
    graphs, singles = [], []
    for _ in range(3):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        g = _graph(p, z, 20.0)
        graphs.append(g)
        singles.append(model(g))
    batched = model(collate(graphs))
    for key in ("energy", "e_bond", "e_angle", "e_urey_bradley", "e_torsion", "e_improper",
                "e_vdw", "e_coulomb", "e_vdw14", "e_coulomb14"):
        stacked = torch.cat([s[key] for s in singles])
        assert float((batched[key] - stacked).abs().max()) < 1e-10
    assert float((batched["node_energy"].sum(0) - batched["energy"].sum()).abs()) < 1e-10


def test_size_extensivity_with_replicated_topology():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    single = Charmm("charmm22", top, cutoff=12.0)
    double = Charmm("charmm22", top.replicate(2), cutoff=12.0)
    e1 = float(single(_graph(pos, z, 12.0))["energy"])
    pos2 = torch.cat([torch.as_tensor(pos), torch.as_tensor(pos) + torch.tensor([50.0, 0.0, 0.0])])
    e2 = float(double(_graph(pos2, z + z, 12.0))["energy"])
    assert abs(e2 - 2 * e1) < 1e-9
    assert double.impropers == single.impropers + [tuple(x + len(z) for x in im) for im in single.impropers]


def test_topology_mismatch_raises():
    model, pos, z = _ethane_model()
    with pytest.raises(ValueError):
        model(_graph(pos[:-1], z[:-1], 20.0))
    z2 = list(z)
    z2[0] = 8
    with pytest.raises(ValueError):
        model(_graph(pos, z2, 20.0))


def test_periodic_minimum_image_and_stress():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", top, cutoff=8.0)
    cell = 20.0 * torch.eye(3)
    e0 = float(model(_graph(pos, z, 8.0, cell))["energy"])
    pos2 = torch.as_tensor(pos).clone()
    pos2[4] += torch.tensor([20.0, 0.0, 0.0])
    assert abs(float(model(_graph(pos2, z, 8.0, cell))["energy"]) - e0) < 1e-9
    pos3 = torch.as_tensor(pos) + torch.tensor([19.0, 0.0, 0.0])
    assert abs(float(model(_graph(pos3, z, 8.0, cell))["energy"]) - e0) < 1e-9
    out = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, 8.0, cell))
    assert bool(torch.isfinite(out["stress"]).all())


def test_switching_function():
    lib = builtin_charmm("charmm22")
    top = MolecularTopology.from_bonds(["CT3", "CT3"], [])
    plain = Charmm(lib, top, cutoff=10.0)
    switched = Charmm(lib, top, cutoff=10.0, switch_width=2.0)
    r = 9.0
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], 10.0)
    x = (r - 8.0) / 2.0
    s = 1 - x**3 * (10 - 15 * x + 6 * x * x)
    assert abs(float(switched(g)["energy"]) - s * float(plain(g)["energy"])) < 1e-12


def test_missing_parameters_are_reported_together():
    lib = builtin_charmm("charmm22")
    with pytest.raises(KeyError) as err:
        Charmm(lib, MolecularTopology.from_bonds(["H", "H"], [(0, 1)]), cutoff=10.0)
    assert "H-H" in str(err.value)
    with pytest.raises(KeyError) as err:
        Charmm(lib, MolecularTopology.from_bonds(["nope"], []), cutoff=10.0)
    assert "nope" in str(err.value)


# training, shared parameters, export, config
def test_trainable_selection_and_gradients():
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", top, cutoff=20.0, trainable=("dihedral_k", "ub_k", "charge"))
    out = model(_graph(pos, z, 20.0))
    out["energy"].pow(2).sum().backward()
    P = model.ff.params
    assert P["dihedral_k"].requires_grad and P["dihedral_k"].grad is not None
    assert P["ub_k"].requires_grad and float(P["ub_k"].grad.abs().sum()) > 0.0
    assert model.charge.requires_grad and model.charge.grad is not None
    # unused Fourier slots of a torsion row get no gradient
    row = model.ff.resolve_dihedral("CT3", "C", "NH1", "CT3")  # two terms
    assert float(P["dihedral_k"].grad[row, :2].abs().min()) > 0.0
    assert float(P["dihedral_k"].grad[row, 2:].abs().sum()) == 0.0
    assert model.ff.dihedral_mask[row].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert not P["bond_k"].requires_grad and not P["rstar"].requires_grad
    frozen = Charmm("charmm22", top, cutoff=20.0)
    assert not any(p.requires_grad for p in frozen.parameters())
    every = Charmm("charmm22", top, cutoff=20.0, trainable="all")
    assert all(p.requires_grad for p in every.ff.params.values())
    assert every.charge.requires_grad and every.atom_ref.weight.requires_grad
    with pytest.raises(ValueError):
        Charmm("charmm22", top, cutoff=20.0, trainable=("not_a_group",))


def test_charge_constraint_keeps_total_charge():
    top = _nma_topology()
    pos, z, charges = _nma()[0], _nma()[1], _nma()[4]
    for constrain, total in ((True, sum(charges)), (0.5, 0.5)):
        model = Charmm("charmm22", top, cutoff=20.0, trainable=("charge",), constrain_charge=constrain)
        g = _graph(pos, z, 20.0)
        assert abs(float(model(g)["charges"].sum()) - total) < 1e-12
        opt = torch.optim.Adam([model.charge], lr=1e-2)
        for _ in range(20):
            opt.zero_grad()
            (model(g)["e_coulomb14"] + 1.0).pow(2).sum().backward()
            opt.step()
        assert abs(float(model(g)["charges"].sum()) - total) < 1e-12
    free = Charmm("charmm22", top, cutoff=20.0, trainable=("charge",), constrain_charge=False)
    assert torch.equal(free.charges, free.charge)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    target_lib = builtin_charmm("charmm22")
    target_lib.torsion("CT3-C-NH1-CT3")["terms"] = [[3.0, 1, 0.0], [5.0, 2, 180.0]]
    target_lib.torsion("O-C-NH1-H")["terms"] = [[4.0, 2, 180.0]]
    target_model = Charmm(target_lib, top, cutoff=20.0)
    model = Charmm("charmm22", top, cutoff=20.0, trainable=("dihedral_k",))
    graphs = [_graph(torch.as_tensor(pos) + 0.08 * torch.randn(len(pos), 3), z, 20.0)
              for _ in range(8)]
    batch = collate(graphs)
    target = target_model(batch)["energy"].detach()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (model(batch)["energy"] - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_shared_forcefield_accumulates_gradients():
    ff = CharmmForceField(builtin_charmm("charmm22"), trainable=("dihedral_k",))
    nma = Charmm(ff, _nma_topology(), cutoff=20.0)
    pos, z = _nma()[0], _nma()[1]
    pos_e, z_e, types_e, bonds_e = _ethane()
    ethane = Charmm(ff, MolecularTopology.from_bonds(types_e, bonds_e), cutoff=20.0)
    assert nma.ff is ethane.ff
    nma(_graph(pos, z, 20.0))["energy"].pow(2).sum().backward()
    g1 = ff.params["dihedral_k"].grad.clone()
    ethane(_graph(pos_e, z_e, 20.0))["energy"].pow(2).sum().backward()
    assert float((ff.params["dihedral_k"].grad - g1).abs().max()) > 0.0


def test_export_library_roundtrip(tmp_path):
    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", top, cutoff=20.0)
    lib = model.export_library()
    assert abs(lib.bond_types["CT3-CT3"]["k"] - 222.5) < 1e-9
    assert abs(lib.angle("HA-CT3-HA")["theta0"] - 108.4) < 1e-9
    assert abs(lib.urey_bradley("HA-CT3-HA")["k"] - 5.4) < 1e-9
    for got, want in zip(lib.torsion("CT3-C-NH1-CT3")["terms"], [[1.6, 1, 0.0], [2.5, 2, 180.0]]):
        assert abs(got[0] - want[0]) < 1e-12 and got[1:] == want[1:]
    assert abs(lib.improper("O-X-X-C")["k"] - 120.0) < 1e-12
    assert abs(lib.atom_types["CT3"]["rstar14"] - 1.9) < 1e-12 and lib.atom_types["HA"]["rstar14"] is None
    assert abs(lib.nbfix("SOD-CLA")["eps"] - 0.0839) < 1e-12
    assert lib.fragments == builtin_charmm("charmm22").fragments
    e1 = float(model(_graph(pos, z, 20.0))["energy"])
    e2 = float(Charmm(lib, top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e2) < 1e-12
    path = lib.save_frc(tmp_path / "exported.frc")
    e3 = float(Charmm(str(path), top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e3) < 1e-12
    lib.save(tmp_path / "lib.json")
    e4 = float(Charmm(str(tmp_path / "lib.json"), top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e4) < 1e-12
    # trained charges land in the residue templates when residues are known
    types, bonds, charges, impropers = _nma()[2], _nma()[3], _nma()[4], _nma()[5]
    top_r = MolecularTopology.from_bonds(
        types, bonds, charges=charges, impropers=impropers,
        residues=[(0, 0), (0, 4), (0, 5), (1, 0), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3),
                  (1, 3), (1, 4), (1, 5)],
        residue_names=["ACE", "CT3"])
    m = Charmm("charmm22", top_r, cutoff=20.0, trainable=("charge",), constrain_charge=False)
    with torch.no_grad():
        m.charge[2] = -0.25
    frag = m.export_library().fragments["ACE"]
    assert frag["charges"][5] == -0.25
    assert builtin_charmm("charmm22").fragments["ACE"]["charges"][5] == -0.51


def test_frc_sections_round_trip(tmp_path):
    lib = CharmmLibrary(name="mini")
    lib.atom_types["A"] = {"element": 6, "mass": 12.0, "rstar": 1.9, "epsilon": 0.1,
                           "rstar14": 1.7, "epsilon14": 0.05}
    lib.atom_types["B"] = {"element": 7, "mass": 14.0, "rstar": 1.8, "epsilon": 0.2,
                           "rstar14": None, "epsilon14": None}
    lib.bond_types["A-A"] = {"k": 300.0, "r0": 1.5}
    lib.angle_types["A-A-A"] = {"k": 40.0, "theta0": 110.0}
    lib.urey_bradley_types["A-A-A"] = {"k": 10.0, "s0": 2.5}
    lib.dihedral_types["X-A-A-X"] = {"terms": [[0.2, 3, 0.0], [1.1, 2, 180.0], [0.3, 1, 0.0],
                                               [0.1, 4, 0.0], [0.05, 5, 180.0], [0.02, 6, 0.0]]}
    lib.improper_types["B-X-X-A"] = {"k": 50.0, "psi0": 0.0}
    lib.nbfix_types["A-B"] = {"rmin": 3.3, "eps": 0.07, "rmin14": None, "eps14": None}
    path = lib.save_frc(tmp_path / "mini.frc")
    ff = read_forcefield(str(path))
    for kind in ("urey_bradley", "improper_harmonic", "nonbond_1-4(12-6)", "nonbond_pair(12-6)",
                 "torsion_fourier"):
        assert kind in ff.sections, kind
    back = read_charmm(str(path))
    assert back.dihedral_types["X-A-A-X"]["terms"] == lib.dihedral_types["X-A-A-X"]["terms"]
    assert back.improper_types["B-X-X-A"] == {"k": 50.0, "psi0": 0.0}
    assert back.urey_bradley_types["A-A-A"] == {"k": 10.0, "s0": 2.5}
    assert abs(back.atom_types["A"]["rstar14"] - 1.7) < 1e-12
    assert back.atom_types["B"]["rstar14"] is None
    assert back.nbfix("A-B") == {"rmin": 3.3, "eps": 0.07, "rmin14": None, "eps14": None}
    assert ff.combination() == "arithmetic"


def test_masses_property():
    model, pos, z = _ethane_model()
    assert torch.allclose(model.masses, torch.tensor([12.011, 12.011] + [1.008] * 6))


def test_from_config_and_key_translation(tmp_path):
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    top = _nma_topology()
    pos, z = _nma()[0], _nma()[1]
    top.save(tmp_path / "top.json")
    cfg = from_dict({"model": {"name": "charmm", "cutoff": 12.0, "prm": "charmm22",
                               "topology": str(tmp_path / "top.json"), "e14fac": 1.0,
                               "ccelec": 332.0636, "trainable": ["dihedral_k"]}})
    model = build_model(cfg.model)
    assert isinstance(model, Charmm)
    assert model.cutoff == 12.0 and model.ff.params["dihedral_k"].requires_grad
    assert model.fudge_qq == 1.0 and model.coulomb_constant == 332.0636
    direct = float(Charmm("charmm22", top, cutoff=12.0, coulomb_constant=332.0636)(
        _graph(pos, z, 12.0))["energy"])
    assert abs(float(model(_graph(pos, z, 12.0))["energy"]) - direct) < 1e-12
    types, bonds, charges, impropers = _nma()[2], _nma()[3], _nma()[4], _nma()[5]
    cfg2 = from_dict({"model": {"name": "charmm", "cutoff": 12.0, "frc": "charmm22",
                                "types": types, "bonds": [list(b) for b in bonds],
                                "impropers": [list(im) for im in impropers], "charges": charges,
                                "coulomb_constant": 332.0636}})
    assert abs(float(build_model(cfg2.model)(_graph(pos, z, 12.0))["energy"]) - direct) < 1e-12
    with pytest.raises(ValueError):
        build_model(from_dict({"model": {"name": "charmm"}}).model)


def test_dataset_training_smoke():
    torch.manual_seed(0)
    pos, z = _nma()[0], _nma()[1]
    model = Charmm("charmm22", _nma_topology(), cutoff=12.0, trainable=("dihedral_k", "charge"))
    fmodel = ForceStressOutput(model)
    fmodel.train()
    structures = []
    for _ in range(4):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        structures.append({"pos": p, "atomic_numbers": torch.tensor(z),
                           "energy": torch.randn(()), "forces": torch.randn(len(pos), 3)})
    ds = AtomicDataset(structures, cutoff=12.0)
    batch = collate([ds[i] for i in range(len(ds))])
    out = fmodel(batch)
    loss, logs = weighted_loss(out, batch, energy_weight=1.0, force_weight=0.1, stress_weight=0.0)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert model.ff.params["dihedral_k"].grad is not None and model.charge.grad is not None


def test_reference_energies_absorb_a_label_offset():
    pos, z = _nma()[0], _nma()[1]
    top = _nma_topology()
    base = Charmm("charmm22", top, cutoff=20.0)
    shifted = Charmm("charmm22", top, cutoff=20.0,
                     atomic_energies={"C": -1.0, "H": -0.5, 6: -1.0, "N": -2.0, "O": -3.0})
    g = _graph(pos, z, 20.0)
    offset = sum({6: -1.0, 1: -0.5, 7: -2.0, 8: -3.0}[zz] for zz in z)
    assert abs(float(shifted(g)["energy"]) - float(base(g)["energy"]) - offset) < 1e-12
    assert abs(float(shifted(g)["e_ref"]) - offset) < 1e-12
