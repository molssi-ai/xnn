"""AMBER force-field tests: equations, library, typing, invariances, training.

Verification is self-contained, in the clean-room convention of this code
base: every energy term is recomputed by hand from the published functional
form (Cornell et al., JACS 117, 5179, 1995, eq 1) and compared against the
model's intermediates, the shipped ``ff94`` library is spot-checked against
Table 14 and the residue charges of Figure 4 of the paper, the improper
placement follows the AMBER builder's ordering rules, and float32 is held
against float64. The LAMMPS cross-check lives in
``notebooks/fidelity_checks/ffnn/amber/amber_verification.ipynb``.
"""

import json
import math

import numpy as np
import pytest
import torch

from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models
from xnn.common.train import weighted_loss
from xnn.ffnn.common import list_forcefields, find_forcefield, read_forcefield
from xnn.ffnn.models import (
    Amber,
    AmberForceField,
    AmberLibrary,
    MolecularTopology,
    builtin_amber,
    read_amber,
    read_topology,
    match_improper,
    place_impropers,
)
from xnn.ffnn.models.amber import KE, KE_KCAL, multiple_angles
from xnn.ffnn.models.amberlib import (
    KCAL_TO_EV,
    SCALE_14_COULOMB,
    SCALE_14_VDW,
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


def _ethane(phi_deg=47.0, r_cc=1.53, r_ch=1.09, ang_deg=110.0):
    """Ethane with a prescribed H-C-C-H dihedral, hand-typed CT / HC."""
    ang = math.radians(ang_deg)
    pos = [[0.0, 0.0, 0.0], [r_cc, 0.0, 0.0]]
    s, c = r_ch * math.sin(math.pi - ang), r_ch * math.cos(math.pi - ang)
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        pos.append([-c, s * math.cos(a), s * math.sin(a)])
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0 + math.radians(phi_deg)
        pos.append([r_cc + c, s * math.cos(a), s * math.sin(a)])
    types = ["CT", "CT"] + ["HC"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    return pos, [6, 6] + [1] * 6, types, bonds


def _ethane_model(**kwargs):
    pos, z, types, bonds = _ethane()
    top = MolecularTopology.from_bonds(types, bonds)
    return Amber("ff94", top, cutoff=kwargs.pop("cutoff", 20.0), **kwargs), pos, z


def _nma():
    """N-methylacetamide, hand-typed with the ACE / NME charges of Figure 4."""
    pos = [
        [0.0, 0.0, 0.0],  # CH3 (acetyl)
        [1.51, 0.0, 0.0],  # C
        [2.13, 1.07, 0.0],  # O
        [2.14, -1.19, 0.0],  # N
        [3.59, -1.28, 0.0],  # CH3 (N-methyl)
        [1.60, -2.05, 0.0],  # H on N
        [-0.38, 1.02, 0.0],
        [-0.38, -0.51, 0.89],
        [-0.38, -0.51, -0.89],
        [3.95, -0.76, 0.89],
        [3.95, -0.76, -0.89],
        [3.95, -2.31, 0.0],
    ]
    z = [6, 6, 8, 7, 6, 1, 1, 1, 1, 1, 1, 1]
    types = ["CT", "C", "O", "N", "CT", "H", "HC", "HC", "HC", "H1", "H1", "H1"]
    bonds = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (3, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (4, 9),
        (4, 10),
        (4, 11),
    ]
    charges = [
        -0.3662,
        0.5972,
        -0.5679,
        -0.4157,
        -0.1490,
        0.2719,
        0.1123,
        0.1123,
        0.1123,
        0.0976,
        0.0976,
        0.0976,
    ]
    return pos, z, types, bonds, charges


def _chain4(phi_deg, r=1.526, ang_deg=109.5):
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
    p3 = (
        p2
        - r * math.cos(ang) * e2
        + r * math.sin(ang) * (math.cos(phi) * m - math.sin(phi) * n)
    )
    return torch.stack([p0, p1, p2, p3])


def _signed_dihedral(p):
    """The dihedral angle in degrees by the atan2 formula (IUPAC sign)."""
    p = np.asarray(p, dtype=float)
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.degrees(math.atan2(np.dot(m1, n2), np.dot(n1, n2)))


def _embedded(smiles, seed=1):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


# registry and library
def test_registration():
    assert "amber" in available_models()
    assert "ff94" in list_forcefields()
    assert find_forcefield("ff94")[1] == "ff94"
    assert find_forcefield("amber")[1] == "ff94"  # bare stem of amber.frc


def test_library_matches_paper_table_14():
    lib = builtin_amber("ff94")
    assert lib.fudge_lj == SCALE_14_VDW and abs(lib.fudge_qq - 1.0 / 1.2) < 1e-15
    # bond parameters (K_r, r_eq)
    assert lib.bond("C-CA") == {"k": 469.0, "r0": 1.409}
    assert lib.bond("CT-CT") == {"k": 310.0, "r0": 1.526}
    assert lib.bond("OW-HW") == {"k": 553.0, "r0": 0.9572}
    assert "HW-OW" in lib.bond_types  # keys are stored canonically
    # angles (K_theta, theta_eq)
    assert lib.angle("CT-CT-CT") == {"k": 40.0, "theta0": 109.5}
    assert lib.angle("H-N-H") == {"k": 35.0, "theta0": 120.0}
    assert lib.angle("HW-OW-HW") == {"k": 100.0, "theta0": 104.52}
    # torsions: V_n/2 over the number of paths, phase, periodicity
    t = lib.torsion("X-CT-CT-X")["terms"]
    assert len(t) == 1 and abs(t[0][0] - 1.40 / 9) < 1e-12 and t[0][1:] == [3, 0.0]
    t = lib.torsion("X-C-CA-X")["terms"]
    assert abs(t[0][0] - 14.50 / 4) < 1e-12 and t[0][1:] == [2, 180.0]
    assert lib.torsion("N-CT-C-N")["terms"] == [
        [0.40, 4, 180.0],
        [1.35, 2, 180.0],
        [0.75, 1, 180.0],
    ]
    assert lib.torsion("CT-CT-OS-CT")["terms"] == [[0.383, 3, 0.0], [0.1, 2, 180.0]]
    assert lib.torsion("H-N-C-O")["terms"] == [[2.5, 2, 180.0], [2.0, 1, 0.0]]
    assert lib.torsion("O-C-N-H") is lib.torsion("H-N-C-O")
    # impropers keep the written order (center third, distinguished atom last)
    assert lib.improper("X-X-C-O")["terms"] == [[10.5, 2, 180.0]]
    assert lib.improper("X-O2-C-O2")["terms"] == [[10.5, 2, 180.0]]
    assert lib.improper("X-X-CA-HA")["terms"] == [[1.1, 2, 180.0]]
    assert lib.improper("CA-CA-C-OH")["terms"] == [[1.1, 2, 180.0]]
    assert "X-X-C-O" in lib.improper_types
    # van der Waals R* and epsilon
    for t, rs, eps in (
        ("C", 1.9080, 0.0860),
        ("CT", 1.9080, 0.1094),
        ("HC", 1.4870, 0.0157),
        ("H", 0.6000, 0.0157),
        ("HO", 0.0, 0.0),
        ("HW", 0.0, 0.0),
        ("OW", 1.7682, 0.1521),  # parm94.dat; Table 14 prints 1.7683 / 0.1520
        ("O", 1.6612, 0.2100),
        ("N", 1.8240, 0.1700),
        ("IP", 1.8680, 0.00277),
        ("IM", 2.47, 0.1),
        ("P", 2.1, 0.2),
    ):
        assert abs(lib.atom_types[t]["rstar"] - rs) < 1e-12, t
        assert abs(lib.atom_types[t]["epsilon"] - eps) < 1e-12, t
    # the nonbond equivalences of parm94: NA, N2, N*, NC, NB take N's values,
    # the aromatic carbons C's
    for t in ("NA", "N2", "N*", "NC", "NB"):
        assert lib.cls(t, "nonbond") == "N"
        assert lib.atom_types[t]["rstar"] == lib.atom_types["N"]["rstar"]
    for t in ("CA", "CB", "CC", "CK", "CM", "CN", "CQ", "CR", "CV", "CW", "C*"):
        assert lib.cls(t, "nonbond") == "C"
        assert lib.atom_types[t]["epsilon"] == 0.0860
    assert lib.atom_types["N"]["mass"] == 14.01 and lib.atom_types["N"]["element"] == 7
    assert (
        lib.atom_types["IP"]["element"] == 11 and lib.atom_types["IM"]["element"] == 17
    )


def test_residue_charges_match_figure_4():
    lib = builtin_amber("ff94")
    ala = lib.residue_charges("ALA")
    assert ala == {
        "N": -0.4157,
        "H": 0.2719,
        "CA": 0.0337,
        "HA": 0.0823,
        "CB": -0.1825,
        "HB1": 0.0603,
        "HB2": 0.0603,
        "HB3": 0.0603,
        "C": 0.5973,
        "O": -0.5679,
    }
    gly = lib.residue_charges("GLY")
    assert gly["CA"] == -0.0252 and gly["HA2"] == 0.0698 and gly["C"] == 0.5973
    ace = lib.residue_charges("ACE")
    assert ace["CH3"] == -0.3662 and ace["C"] == 0.5972 and ace["O"] == -0.5679
    nme = lib.residue_charges("NME")
    assert nme["N"] == -0.4157 and nme["CH3"] == -0.1490 and nme["HH31"] == 0.0976
    assert lib.residue_charges("TP3") == {"O": -0.834, "H1": 0.417, "H2": 0.417}
    assert lib.residue_charges("Na+") == {"Na+": 1.0}
    # every residue carries an integer total charge, except the 5'- and
    # 3'-terminal nucleotides whose fragment charges are integer as a pair
    # (Figure 5: A + C and B + D)
    termini = {
        n for n in lib.fragments if len(n) == 3 and n[0] in "DR" and n[-1] in "53"
    }
    for name, frag in lib.fragments.items():
        total = sum(frag["charges"])
        if name not in termini:
            assert abs(total - round(total)) < 1e-3, name
        assert (
            len(frag["charges"]) == len(frag["atom types"]) == len(frag["atom names"])
        )
    for name in termini:
        if name.endswith("5"):
            pair = sum(lib.fragments[name]["charges"]) + sum(
                lib.fragments[name[:-1] + "3"]["charges"]
            )
            assert abs(pair - round(pair)) < 1e-3, name
    assert sum(lib.fragments["DA5"]["charges"]) == pytest.approx(-0.3079, abs=1e-4)
    assert sum(lib.fragments["DA3"]["charges"]) == pytest.approx(-0.6921, abs=1e-4)
    assert sum(lib.fragments["LYS"]["charges"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(lib.fragments["ASP"]["charges"]) == pytest.approx(-1.0, abs=1e-3)
    assert len(lib.fragments) > 100 and {
        "NALA",
        "CALA",
        "DA",
        "RA5",
        "HIP",
        "CYX",
    } <= set(lib.fragments)


# energy expressions (eq 1)
def test_bond_angle_and_14_equations():
    model, pos, z = _ethane_model(keep_intermediates=True)
    out = model(_graph(pos, z, 20.0))
    lib = builtin_amber("ff94")
    P = np.asarray(pos)
    e_bond = 0.0
    for i, j in model.topology.bonds:
        r = np.linalg.norm(P[i] - P[j])
        p = lib.bond_types["CT-CT" if z[i] == z[j] == 6 else "CT-HC"]
        e_bond += p["k"] * (r - p["r0"]) ** 2
    assert abs(float(out["e_bond"]) * EV_TO_KCAL - e_bond) < 1e-10
    e_angle = 0.0
    for i, j, k in model.topology.angles:
        u, v = P[i] - P[j], P[k] - P[j]
        th = math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))
        key = "CT-CT-HC" if 6 in (z[i], z[k]) else "HC-CT-HC"
        p = lib.angle_types[key]
        e_angle += p["k"] * (th - p["theta0"] * DEG) ** 2
    assert abs(float(out["e_angle"]) * EV_TO_KCAL - e_angle) < 1e-10
    # 1,2 and 1,3 pairs are excluded: no direct nonbonded energy at all
    assert abs(float(out["e_vdw"]) + float(out["e_coulomb"])) < 1e-14
    # the nine H...H 1,4 pairs: 1/2 eps [(R*/r)^12 - 2 (R*/r)^6], R* = 2 x 1.487
    e14 = 0.0
    for i, j in model.topology.pairs14:
        r = np.linalg.norm(P[i] - P[j])
        x = (2 * 1.487 / r) ** 6
        e14 += 0.5 * 0.0157 * (x * x - 2 * x)
    assert len(model.topology.pairs14) == 9
    assert abs(float(out["e_vdw14"]) * EV_TO_KCAL - e14) < 1e-12
    assert abs(float(out["e_coulomb14"])) < 1e-14  # no charges


def test_torsion_fourier_terms_and_phase():
    lib = builtin_amber("ff94")
    # X-CT-CT-X: one term, K = 1.4/9, n = 3, gamma = 0
    top = MolecularTopology.from_bonds(["CT"] * 4, [(0, 1), (1, 2), (2, 3)])
    model = Amber(lib, top, cutoff=20.0)
    for phi in (0.0, 60.0, 100.0, 180.0, -70.0):
        out = model(_graph(_chain4(phi), [6] * 4, 20.0))
        want = 1.4 / 9 * (1 + math.cos(3 * math.radians(phi)))
        assert abs(float(out["e_torsion"]) * EV_TO_KCAL - want) < 1e-12
    # N-CT-C-N: three terms with gamma = 180 (even in phi)
    top = MolecularTopology.from_bonds(["N", "CT", "C", "N"], [(0, 1), (1, 2), (2, 3)])
    model = Amber(lib, top, cutoff=20.0)
    for phi in (0.0, 45.0, 100.0, 180.0):
        out = model(_graph(_chain4(phi), [7, 6, 6, 7], 20.0))
        p = math.radians(phi)
        want = sum(
            k * (1 + math.cos(n * p - math.radians(g)))
            for k, n, g in lib.torsion("N-CT-C-N")["terms"]
        )
        assert abs(float(out["e_torsion"]) * EV_TO_KCAL - want) < 1e-12
    # a phase of 90 degrees makes the term odd in phi: the signed dihedral
    # enters through sin(n phi), checked against the atan2 definition
    lib.torsion("N-CT-C-N")["terms"] = [[1.0, 1, 90.0], [0.5, 3, 30.0]]
    model = Amber(lib, top, cutoff=20.0)
    for phi in (35.0, -35.0, 120.0, -170.0):
        pos = _chain4(phi)
        assert abs(_signed_dihedral(pos) - phi) < 1e-9
        out = model(_graph(pos, [7, 6, 6, 7], 20.0))
        p = math.radians(phi)
        want = 1.0 * (1 + math.cos(p - math.pi / 2)) + 0.5 * (
            1 + math.cos(3 * p - math.pi / 6)
        )
        assert abs(float(out["e_torsion"]) * EV_TO_KCAL - want) < 1e-12


def test_multiple_angle_recurrence():
    phi = torch.tensor([0.3, -1.2, 2.9, 3.14159])
    c, s = multiple_angles(torch.cos(phi), torch.sin(phi), 6)
    for n in range(1, 7):
        assert torch.allclose(c[:, n - 1], torch.cos(n * phi), atol=1e-14)
        assert torch.allclose(s[:, n - 1], torch.sin(n * phi), atol=1e-14)


def test_vdw_combination_and_coulomb_constant():
    lib = builtin_amber("ff94")
    top = MolecularTopology.from_bonds(["C", "N"], [], charges=[0.5, -0.5])
    model = Amber(lib, top, cutoff=20.0)
    for r in (2.8, 3.5, 5.0):
        out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 7], 20.0))
        rstar = 1.908 + 1.824
        eps = math.sqrt(0.086 * 0.17)
        x = (rstar / r) ** 6
        assert abs(float(out["e_vdw"]) * EV_TO_KCAL - eps * (x * x - 2 * x)) < 1e-10
        assert abs(float(out["e_coulomb"]) * EV_TO_KCAL + KE_KCAL * 0.25 / r) < 1e-10
    assert abs(KE_KCAL - 332.0522173) < 1e-7
    assert abs(KE - KE_KCAL * KCAL_TO_EV) < 1e-15
    # the well sits at R*_ij with depth -eps_ij
    out = model(_graph([[0, 0, 0], [1.908 + 1.824, 0, 0]], [6, 7], 20.0))
    assert abs(float(out["e_vdw"]) * EV_TO_KCAL + math.sqrt(0.086 * 0.17)) < 1e-10


def test_improper_equation_and_ordering():
    lib = builtin_amber("ff94")
    # a carbonyl center: C bonded to CT, N and O; pattern X-X-C-O puts O
    # last and orders the two wildcard atoms by index (no residues here)
    pos = [[0.0, 0.0, 0.0], [-1.5, 0.3, 0.0], [0.6, 1.1, 0.25], [0.7, -1.2, -0.2]]
    z = [6, 6, 8, 7]
    top = MolecularTopology.from_bonds(["C", "CT", "O", "N"], [(0, 1), (0, 2), (0, 3)])
    model = Amber(lib, top, cutoff=20.0, keep_intermediates=True)
    assert model.impropers == [(1, 3, 0, 2)]
    out = model(_graph(pos, z, 20.0))
    phi = math.radians(_signed_dihedral([pos[1], pos[3], pos[0], pos[2]]))
    want = 10.5 * (1 + math.cos(2 * phi - math.pi))
    assert abs(float(out["e_improper"]) * EV_TO_KCAL - want) < 1e-10
    assert float(out["e_improper"]) > 1e-4
    # with residue positions the template order decides: give N the lower
    # template position and it leads
    top2 = MolecularTopology.from_bonds(
        ["C", "CT", "O", "N"],
        [(0, 1), (0, 2), (0, 3)],
        residues=[(0, 5), (0, 3), (0, 6), (1, 0)],
    )
    assert Amber(lib, top2, cutoff=20.0).impropers == [(1, 3, 0, 2)]
    top3 = MolecularTopology.from_bonds(
        ["C", "CT", "O", "N"],
        [(0, 1), (0, 2), (0, 3)],
        residues=[(0, 5), (1, 3), (0, 6), (0, 0)],
    )
    assert Amber(lib, top3, cutoff=20.0).impropers == [(3, 1, 0, 2)]
    # X-O2-C-O2 at a carboxylate: the carbon neighbour goes first, the two
    # oxygens follow template order
    outer = [(7, "O2", "O", (0, 8)), (5, "CT", "C", (0, 5)), (6, "O2", "O", (0, 7))]
    assert match_improper(lib.improper_types, "C", outer) == ("X-O2-C-O2", (5, 6, 7))
    # a wildcard-free pattern wins over wildcard ones, whatever the order
    table = {
        "X-X-C-O": {"terms": [[10.5, 2, 180.0]]},
        "CT-O-C-OH": {"terms": [[10.5, 2, 180.0]]},
    }
    outer = [(1, "OH", "O", (0, 9)), (2, "O", "O", (0, 8)), (3, "CT", "C", (0, 2))]
    assert match_improper(table, "C", outer) == ("CT-O-C-OH", (3, 2, 1))
    assert match_improper(table, "CT", outer) is None
    assert resolve_improper_type(lib.improper_types, "CT", "N", "C", "O") == "X-X-C-O"
    assert resolve_improper_type(lib.improper_types, "CT", "CT", "CT", "CT") is None
    # a tetrahedral center gets no improper; an explicit list is used as given
    pos_e, z_e, types, bonds = _ethane()
    assert (
        Amber(lib, MolecularTopology.from_bonds(types, bonds), cutoff=20.0).impropers
        == []
    )
    explicit = MolecularTopology.from_bonds(
        ["C", "CT", "O", "N"], [(0, 1), (0, 2), (0, 3)], impropers=[(3, 1, 0, 2)]
    )
    m = Amber(lib, explicit, cutoff=20.0)
    assert m.impropers == [(3, 1, 0, 2)]
    assert (
        abs(float(m(_graph(pos, z, 20.0))["e_improper"]) - float(out["e_improper"]))
        > 1e-6
    )


def test_improper_placement_matches_amber_for_a_dipeptide():
    pytest.importorskip("rdkit")
    model = Amber.from_atoms(_embedded("CC(=O)N[C@@H](C)C(=O)NC"), "ff94", cutoff=20.0)
    top = model.topology
    assert top.residue_names == ["ACE", "ALA", "NME"]
    assert len(model.impropers) == 4
    by_center = {im[2]: im for im in model.impropers}
    names = {
        a: top.residue_names[r[0]]
        + ":"
        + builtin_amber("ff94").fragments[top.residue_names[r[0]]]["atom names"][r[1]]
        for a, r in enumerate(top.residues)
    }
    # the ACE carbonyl: (CH3, N of ALA, C, O); the ALA carbonyl: (CA, N of NME, C, O)
    ace_c = next(a for a, n in names.items() if n == "ACE:C")
    assert [names[a] for a in by_center[ace_c]] == [
        "ACE:CH3",
        "ALA:N",
        "ACE:C",
        "ACE:O",
    ]
    ala_c = next(a for a, n in names.items() if n == "ALA:C")
    assert [names[a] for a in by_center[ala_c]] == ["ALA:CA", "NME:N", "ALA:C", "ALA:O"]
    # the amide nitrogens: (C of the previous residue, CA, N, H)
    ala_n = next(a for a, n in names.items() if n == "ALA:N")
    assert [names[a] for a in by_center[ala_n]] == ["ACE:C", "ALA:CA", "ALA:N", "ALA:H"]
    nme_n = next(a for a, n in names.items() if n == "NME:N")
    assert [names[a] for a in by_center[nme_n]] == [
        "ALA:C",
        "NME:CH3",
        "NME:N",
        "NME:H",
    ]


# typing
def test_from_atoms_types_residues_and_charges():
    pytest.importorskip("rdkit")
    lib = builtin_amber("ff94")
    model = Amber.from_atoms(_embedded("CC(=O)N[C@@H](C)C(=O)NC"), lib, cutoff=20.0)
    top = model.topology
    assert len(top.types) == 22
    assert sorted(set(top.types)) == ["C", "CT", "H", "H1", "HC", "N", "O"]
    assert abs(sum(top.charges)) < 1e-6
    # every atom got a residue position and the template charge of that atom
    for a, r in enumerate(top.residues):
        frag = lib.fragments[top.residue_names[r[0]]]
        assert top.types[a] == frag["atom types"][r[1]]
        assert top.charges[a] == frag["charges"][r[1]]
    ala_atoms = [
        a for a, r in enumerate(top.residues) if top.residue_names[r[0]] == "ALA"
    ]
    assert len(ala_atoms) == 10
    assert sorted(top.charges[a] for a in ala_atoms) == sorted(
        lib.fragments["ALA"]["charges"]
    )
    out = ForceStressOutput(model)(
        _graph(
            _embedded("CC(=O)N[C@@H](C)C(=O)NC").GetConformer().GetPositions(),
            [a.GetAtomicNum() for a in _embedded("CC(=O)N[C@@H](C)C(=O)NC").GetAtoms()],
            20.0,
        )
    )
    assert bool(torch.isfinite(out["energy"]).all()) and bool(
        torch.isfinite(out["forces"]).all()
    )
    # explicit charges override the templates
    m2 = Amber.from_atoms(
        _embedded("CC(=O)N[C@@H](C)C(=O)NC"), lib, cutoff=20.0, charges=[0.0] * 22
    )
    assert float(m2.charges.abs().sum()) == 0.0


def test_from_atoms_small_molecules_use_the_type_templates():
    pytest.importorskip("rdkit")
    cases = {
        "CC": {"CT", "HC"},
        "CO": {"CT", "H1", "OH", "HO"},
        "COC": {"CT", "H1", "OS"},
        "c1ccccc1": {"CA", "HA"},
        "CS": {"CT", "H1", "SH", "HS"},
        "CSSC": {"CT", "H1", "S"},
        "CC(=O)NC": {"CT", "C", "O", "N", "H", "HC", "H1"},
        "COP(=O)([O-])OC": {"CT", "H1", "OS", "P", "O2"},
        "O": {"OW", "HW"},
        "[Na+]": {"IP"},
    }
    for smiles, expect in cases.items():
        model = Amber.from_atoms(
            _embedded(smiles),
            "ff94",
            cutoff=20.0,
            charge=-1 if "[O-]" in smiles else (1 if "+" in smiles else 0),
        )
        assert set(model.topology.types) == expect, smiles
        if smiles == "CC(=O)NC":
            # N-methylacetamide is the ACE + NME capping groups: the residue
            # templates type it and supply the paper's charges
            assert model.topology.residue_names == ["ACE", "NME"]
            assert abs(sum(model.topology.charges)) < 1e-6
            assert float(model.charges.abs().sum()) > 1.0
        elif smiles != "O" and smiles != "[Na+]":
            assert not model.topology.residues  # no residue template matched
            assert float(model.charges.abs().sum()) == 0.0
    water = Amber.from_atoms(_embedded("O"), "ff94", cutoff=20.0)
    assert water.topology.residue_names == ["TP3"]
    assert sorted(water.topology.charges) == [-0.834, 0.417, 0.417]
    assert water.topology.bonds == [(0, 1), (0, 2)]


def test_topology_carries_charges_and_residues(tmp_path):
    top = MolecularTopology.from_bonds(
        ["C", "CT", "O", "N"],
        [(0, 1), (0, 2), (0, 3)],
        charges=[0.5, -0.1, -0.5, 0.1],
        residues=[(0, 5), (0, 3), (0, 6), (1, 0)],
        residue_names=["ACE", "ALA"],
    )
    top.save(tmp_path / "top.json")
    back = read_topology(tmp_path / "top.json")
    assert back.charges == top.charges and back.residues == top.residues
    assert back.residue_names == ["ACE", "ALA"]
    two = top.replicate(2)
    assert two.charges == top.charges * 2
    assert two.residues[4:] == [(2, 5), (2, 3), (2, 6), (3, 0)]
    assert two.residue_names == ["ACE", "ALA", "ACE", "ALA"]
    with pytest.raises(ValueError):
        MolecularTopology.from_bonds(["C", "CT"], [(0, 1)], charges=[0.0])


# precision: float32 against float64
def test_float32_matches_float64():
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    model64 = Amber("ff94", top, cutoff=20.0)
    g64 = _graph(pos, z, 20.0)
    out64 = ForceStressOutput(model64)(g64)
    torch.set_default_dtype(torch.float32)
    try:
        model32 = Amber("ff94", top, cutoff=20.0)
        out32 = ForceStressOutput(model32)(_graph(pos, z, 20.0))
    finally:
        torch.set_default_dtype(torch.float64)
    assert out32["node_energy"].dtype == torch.float32  # totals are kept in float64
    for key in (
        "e_bond",
        "e_angle",
        "e_torsion",
        "e_improper",
        "e_vdw",
        "e_coulomb",
        "e_vdw14",
        "e_coulomb14",
        "energy",
    ):
        a, b = float(out64[key]), float(out32[key])
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), key
    assert (
        float((out64["forces"] - out32["forces"].to(torch.float64)).abs().max()) < 2e-4
    )
    # the fixed phases are exact in both precisions (sin 180 = 0 exactly)
    assert float(model32.ff.dihedral_sin.abs().max()) == 0.0
    assert set(model32.ff.dihedral_cos.unique().tolist()) <= {-1.0, 1.0}


# invariances, forces, batching, periodicity
def test_rotation_translation_invariance_and_force_equivariance():
    pos, z, types, bonds, charges = _nma()
    model = Amber(
        "ff94", MolecularTopology.from_bonds(types, bonds, charges=charges), cutoff=20.0
    )
    fmodel = ForceStressOutput(model)
    out = fmodel(_graph(pos, z, 20.0))
    e0, f0 = out["energy"], out["forces"]
    ang = 0.6
    R = torch.tensor(
        [
            [math.cos(ang), -math.sin(ang), 0.0],
            [math.sin(ang), math.cos(ang), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pos2 = torch.as_tensor(pos) @ R.T + torch.tensor([3.0, -2.0, 1.0])
    out2 = fmodel(_graph(pos2, z, 20.0))
    assert float((out2["energy"] - e0).abs()) < 1e-9
    assert float((out2["forces"] - f0 @ R.T).abs().max()) < 1e-8
    # mirror image: the energy is even in every dihedral and improper
    out3 = fmodel(
        _graph(torch.as_tensor(pos) * torch.tensor([1.0, 1.0, -1.0]), z, 20.0)
    )
    assert float((out3["energy"] - e0).abs()) < 1e-9


def test_forces_match_finite_differences():
    pos, z, types, bonds, charges = _nma()
    model = Amber(
        "ff94", MolecularTopology.from_bonds(types, bonds, charges=charges), cutoff=20.0
    )
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
    pos, z, types, bonds, charges = _nma()
    model = Amber(
        "ff94", MolecularTopology.from_bonds(types, bonds, charges=charges), cutoff=20.0
    )
    torch.manual_seed(0)
    graphs, singles = [], []
    for _ in range(3):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        g = _graph(p, z, 20.0)
        graphs.append(g)
        singles.append(model(g))
    batched = model(collate(graphs))
    for key in (
        "energy",
        "e_bond",
        "e_angle",
        "e_torsion",
        "e_improper",
        "e_vdw",
        "e_coulomb",
        "e_vdw14",
        "e_coulomb14",
    ):
        stacked = torch.cat([s[key] for s in singles])
        assert float((batched[key] - stacked).abs().max()) < 1e-10
    assert (
        float((batched["node_energy"].sum(0) - batched["energy"].sum()).abs()) < 1e-10
    )


def test_size_extensivity_with_replicated_topology():
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    single = Amber("ff94", top, cutoff=12.0)
    double = Amber("ff94", top.replicate(2), cutoff=12.0)
    e1 = float(single(_graph(pos, z, 12.0))["energy"])
    pos2 = torch.cat(
        [torch.as_tensor(pos), torch.as_tensor(pos) + torch.tensor([50.0, 0.0, 0.0])]
    )
    e2 = float(double(_graph(pos2, z + z, 12.0))["energy"])
    assert abs(e2 - 2 * e1) < 1e-9
    assert double.impropers == single.impropers + [
        tuple(x + len(z) for x in im) for im in single.impropers
    ]


def test_topology_mismatch_raises():
    model, pos, z = _ethane_model()
    with pytest.raises(ValueError):
        model(_graph(pos[:-1], z[:-1], 20.0))
    z2 = list(z)
    z2[0] = 8
    with pytest.raises(ValueError):
        model(_graph(pos, z2, 20.0))


def test_periodic_minimum_image_and_stress():
    pos, z, types, bonds, charges = _nma()
    model = Amber(
        "ff94", MolecularTopology.from_bonds(types, bonds, charges=charges), cutoff=8.0
    )
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
    lib = builtin_amber("ff94")
    top = MolecularTopology.from_bonds(["CT", "CT"], [])
    plain = Amber(lib, top, cutoff=10.0)
    switched = Amber(lib, top, cutoff=10.0, switch_width=2.0)
    r = 9.0
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], 10.0)
    x = (r - 8.0) / 2.0
    s = 1 - x**3 * (10 - 15 * x + 6 * x * x)
    assert abs(float(switched(g)["energy"]) - s * float(plain(g)["energy"])) < 1e-12


def test_missing_parameters_are_reported_together():
    lib = builtin_amber("ff94")
    with pytest.raises(KeyError) as err:
        Amber(lib, MolecularTopology.from_bonds(["HO", "HO"], [(0, 1)]), cutoff=10.0)
    assert "HO-HO" in str(err.value)
    with pytest.raises(KeyError) as err:
        Amber(lib, MolecularTopology.from_bonds(["nope"], []), cutoff=10.0)
    assert "nope" in str(err.value)


# training, shared parameters, export, config
def test_trainable_selection_and_gradients():
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    model = Amber("ff94", top, cutoff=20.0, trainable=("dihedral_k", "charge"))
    out = model(_graph(pos, z, 20.0))
    out["energy"].pow(2).sum().backward()
    P = model.ff.params
    assert P["dihedral_k"].requires_grad and P["dihedral_k"].grad is not None
    assert model.charge.requires_grad and model.charge.grad is not None
    # unused Fourier slots of a torsion row get no gradient
    row = model.ff.resolve_dihedral("CT", "C", "N", "CT")  # X-C-N-X, one term
    assert float(P["dihedral_k"].grad[row, 0].abs()) > 0.0
    assert float(P["dihedral_k"].grad[row, 1:].abs().sum()) == 0.0
    assert model.ff.dihedral_mask[row].tolist() == [1.0, 0.0, 0.0, 0.0]
    row = model.ff.resolve_dihedral("H", "N", "C", "O")  # two terms
    assert float(P["dihedral_k"].grad[row, :2].abs().min()) > 0.0
    assert float(P["dihedral_k"].grad[row, 2:].abs().sum()) == 0.0
    assert not P["bond_k"].requires_grad and not P["rstar"].requires_grad
    frozen = Amber("ff94", top, cutoff=20.0)
    assert not any(p.requires_grad for p in frozen.parameters())
    every = Amber("ff94", top, cutoff=20.0, trainable="all")
    assert all(p.requires_grad for p in every.ff.params.values())
    assert every.charge.requires_grad and every.atom_ref.weight.requires_grad
    with pytest.raises(ValueError):
        Amber("ff94", top, cutoff=20.0, trainable=("not_a_group",))


def test_charge_constraint_keeps_total_charge():
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    for constrain, total in ((True, sum(charges)), (0.5, 0.5)):
        model = Amber(
            "ff94", top, cutoff=20.0, trainable=("charge",), constrain_charge=constrain
        )
        g = _graph(pos, z, 20.0)
        assert abs(float(model(g)["charges"].sum()) - total) < 1e-12
        opt = torch.optim.Adam([model.charge], lr=1e-2)
        for _ in range(20):
            opt.zero_grad()
            (model(g)["e_coulomb"] + 1.0).pow(2).sum().backward()
            opt.step()
        assert abs(float(model(g)["charges"].sum()) - total) < 1e-12
    free = Amber(
        "ff94", top, cutoff=20.0, trainable=("charge",), constrain_charge=False
    )
    assert torch.equal(free.charges, free.charge)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    target_lib = builtin_amber("ff94")
    target_lib.torsion("H-N-C-O")["terms"] = [[6.0, 2, 180.0], [4.0, 1, 0.0]]
    target_lib.torsion("X-C-N-X")["terms"] = [[5.0, 2, 180.0]]
    target_model = Amber(target_lib, top, cutoff=20.0)
    model = Amber("ff94", top, cutoff=20.0, trainable=("dihedral_k",))
    graphs = [
        _graph(torch.as_tensor(pos) + 0.08 * torch.randn(len(pos), 3), z, 20.0)
        for _ in range(8)
    ]
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
    ff = AmberForceField(builtin_amber("ff94"), trainable=("dihedral_k",))
    pos, z, types, bonds, charges = _nma()
    nma = Amber(
        ff, MolecularTopology.from_bonds(types, bonds, charges=charges), cutoff=20.0
    )
    pos_e, z_e, types_e, bonds_e = _ethane()
    ethane = Amber(ff, MolecularTopology.from_bonds(types_e, bonds_e), cutoff=20.0)
    assert nma.ff is ethane.ff
    nma(_graph(pos, z, 20.0))["energy"].pow(2).sum().backward()
    g1 = ff.params["dihedral_k"].grad.clone()
    ethane(_graph(pos_e, z_e, 20.0))["energy"].pow(2).sum().backward()
    assert float((ff.params["dihedral_k"].grad - g1).abs().max()) > 0.0


def test_export_library_roundtrip(tmp_path):
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    model = Amber("ff94", top, cutoff=20.0)
    lib = model.export_library()
    assert abs(lib.bond_types["CT-CT"]["k"] - 310.0) < 1e-9
    assert abs(lib.angle_types["CT-CT-CT"]["theta0"] - 109.5) < 1e-9
    for got, want in zip(
        lib.torsion("N-CT-C-N")["terms"],
        [[0.40, 4, 180.0], [1.35, 2, 180.0], [0.75, 1, 180.0]],
    ):
        assert abs(got[0] - want[0]) < 1e-12 and got[1:] == want[1:]
    got = lib.improper("X-X-C-O")["terms"]
    assert len(got) == 1 and abs(got[0][0] - 10.5) < 1e-12 and got[0][1:] == [2, 180.0]
    assert abs(lib.atom_types["CT"]["rstar"] - 1.908) < 1e-12
    assert lib.fragments == builtin_amber("ff94").fragments
    e1 = float(model(_graph(pos, z, 20.0))["energy"])
    e2 = float(Amber(lib, top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e2) < 1e-12
    path = lib.save_frc(tmp_path / "exported.frc")
    e3 = float(Amber(str(path), top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e3) < 1e-12
    lib.save(tmp_path / "lib.json")
    e4 = float(
        Amber(str(tmp_path / "lib.json"), top, cutoff=20.0)(_graph(pos, z, 20.0))[
            "energy"
        ]
    )
    assert abs(e1 - e4) < 1e-12
    # trained charges land in the residue templates when residues are known
    top_r = MolecularTopology.from_bonds(
        types,
        bonds,
        charges=charges,
        residues=[
            (0, 1),
            (0, 4),
            (0, 5),
            (1, 0),
            (1, 2),
            (1, 1),
            (0, 0),
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 4),
            (1, 5),
        ],
        residue_names=["ACE", "NME"],
    )
    m = Amber("ff94", top_r, cutoff=20.0, trainable=("charge",), constrain_charge=False)
    with torch.no_grad():
        m.charge[2] = -0.25
    frag = m.export_library().fragments["ACE"]
    assert frag["charges"][5] == -0.25
    assert builtin_amber("ff94").fragments["ACE"]["charges"][5] == -0.5679


def test_frc_torsion_fourier_round_trip(tmp_path):
    lib = AmberLibrary(name="mini")
    lib.atom_types["A"] = {"element": 6, "mass": 12.0, "rstar": 1.9, "epsilon": 0.1}
    lib.bond_types["A-A"] = {"k": 300.0, "r0": 1.5}
    lib.angle_types["A-A-A"] = {"k": 40.0, "theta0": 110.0}
    lib.dihedral_types["X-A-A-X"] = {
        "terms": [[0.2, 3, 0.0], [1.1, 2, 180.0], [0.3, 1, 0.0]]
    }
    lib.improper_types["X-X-A-A"] = {"terms": [[1.0, 2, 180.0]]}
    path = lib.save_frc(tmp_path / "mini.frc")
    ff = read_forcefield(str(path))
    assert "torsion_fourier" in ff.sections and "improper_fourier" in ff.sections
    back = read_amber(str(path))
    assert (
        back.dihedral_types["X-A-A-X"]["terms"]
        == lib.dihedral_types["X-A-A-X"]["terms"]
    )
    assert back.improper_types["X-X-A-A"]["terms"] == [[1.0, 2, 180.0]]
    assert abs(back.atom_types["A"]["rstar"] - 1.9) < 1e-12
    assert ff.combination() == "arithmetic"


def test_masses_property():
    model, pos, z = _ethane_model()
    assert torch.allclose(model.masses, torch.tensor([12.01, 12.01] + [1.008] * 6))


def test_from_config_and_key_translation(tmp_path):
    from xnn.common.config import from_dict
    from xnn.common.models import build_model

    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    top.save(tmp_path / "top.json")
    cfg = from_dict(
        {
            "model": {
                "name": "amber",
                "cutoff": 12.0,
                "parameter_file": "ff94",  # translated key
                "topology": str(tmp_path / "top.json"),
                "scee": 1.2,
                "scnb": 2.0,  # AMBER's 1,4 divisors
                "trainable": ["dihedral_k"],
            }
        }
    )
    model = build_model(cfg.model)
    assert isinstance(model, Amber)
    assert model.cutoff == 12.0 and model.ff.params["dihedral_k"].requires_grad
    assert abs(model.fudge_qq - 1 / 1.2) < 1e-15 and model.fudge_lj == 0.5
    direct = float(Amber("ff94", top, cutoff=12.0)(_graph(pos, z, 12.0))["energy"])
    assert abs(float(model(_graph(pos, z, 12.0))["energy"]) - direct) < 1e-12
    cfg2 = from_dict(
        {
            "model": {
                "name": "amber",
                "cutoff": 12.0,
                "frc": "ff94",
                "types": types,
                "bonds": [list(b) for b in bonds],
                "charges": charges,
            }
        }
    )
    assert (
        abs(float(build_model(cfg2.model)(_graph(pos, z, 12.0))["energy"]) - direct)
        < 1e-12
    )
    with pytest.raises(ValueError):
        build_model(from_dict({"model": {"name": "amber"}}).model)


def test_dataset_training_smoke():
    torch.manual_seed(0)
    pos, z, types, bonds, charges = _nma()
    model = Amber(
        "ff94",
        MolecularTopology.from_bonds(types, bonds, charges=charges),
        cutoff=12.0,
        trainable=("dihedral_k", "charge"),
    )
    fmodel = ForceStressOutput(model)
    fmodel.train()
    structures = []
    for _ in range(4):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        structures.append(
            {
                "pos": p,
                "atomic_numbers": torch.tensor(z),
                "energy": torch.randn(()),
                "forces": torch.randn(len(pos), 3),
            }
        )
    ds = AtomicDataset(structures, cutoff=12.0)
    batch = collate([ds[i] for i in range(len(ds))])
    out = fmodel(batch)
    loss, logs = weighted_loss(
        out, batch, energy_weight=1.0, force_weight=0.1, stress_weight=0.0
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert (
        model.ff.params["dihedral_k"].grad is not None and model.charge.grad is not None
    )


def test_reference_energies_absorb_a_label_offset():
    pos, z, types, bonds, charges = _nma()
    top = MolecularTopology.from_bonds(types, bonds, charges=charges)
    base = Amber("ff94", top, cutoff=20.0)
    shifted = Amber(
        "ff94",
        top,
        cutoff=20.0,
        atomic_energies={"C": -1.0, "H": -0.5, 6: -1.0, "N": -2.0, "O": -3.0},
    )
    g = _graph(pos, z, 20.0)
    offset = sum({6: -1.0, 1: -0.5, 7: -2.0, 8: -3.0}[zz] for zz in z)
    assert abs(float(shifted(g)["energy"]) - float(base(g)["energy"]) - offset) < 1e-12
    assert abs(float(shifted(g)["e_ref"]) - offset) < 1e-12
