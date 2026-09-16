"""OPLS force-field tests: equations, invariances, libraries, training.

Verification is self-contained, in the clean-room convention this code base
uses: every energy term is recomputed by hand from the published OPLS
functional form (Jorgensen, Maxwell & Tirado-Rives, JACS 118, 11225, 1996,
eqs 1-4) and compared against the model's intermediates; the Fourier /
Ryckaert-Bellemans conversion is checked against the L-OPLS paper's own
table (Siu et al., JCTC 8, 1459, 2012, Table 2), which lists both forms of
the same torsions. The relaxed ethane rotational barrier is checked against
Table 1 of the 1996 paper (3.01 kcal/mol) when ASE is available.

Parameters come from the OPLS-AA distribution shipped as a SEAMM ``.frc``
file, whose atom-type names are used throughout (``opls_80`` alkane CH3
carbon, ``opls_81`` CH2, ``opls_85`` H on carbon, ``opls_88``/``opls_89``
ethylene, ``opls_96``/``opls_97``/``opls_99`` alcohol O / H / C); bonded
parameters are keyed by the equivalent types (``opls_18`` alkane carbon,
``opls_86`` alkene carbon, ``opls_5``/``opls_7`` alcohol O / H).
"""
import json
import math

import pytest
import torch

from xnns.common.data import AtomicDataset, collate, structure_to_graph
from xnns.common.models import ForceStressOutput, available_models
from xnns.common.train import weighted_loss
from xnns.ffnn.models import (OPLS, OPLSForceField, MolecularTopology,
                              builtin_library, guess_bonds, read_opls,
                              read_topology, fourier_to_rb, rb_to_fourier)
from xnns.ffnn.models.oplslib import (KCAL_TO_EV, resolve_improper_type,
                                      improper_key)
from xnns.ffnn.models.opls import KE

EV_TO_KCAL = 1.0 / KCAL_TO_EV
CT, HC, CM, OH, HO = "opls_18", "opls_85", "opls_86", "opls_5", "opls_7"


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ----------------------------------------------------------------------
# small builders
# ----------------------------------------------------------------------
def _graph(pos, z, cutoff, cell=None):
    """Build a single-structure graph from raw arrays."""
    s = {"pos": torch.as_tensor(pos, dtype=torch.get_default_dtype()),
         "atomic_numbers": torch.as_tensor(z, dtype=torch.long)}
    if cell is not None:
        s["cell"] = torch.as_tensor(cell, dtype=torch.get_default_dtype())
        s["pbc"] = torch.tensor([True, True, True])
    return structure_to_graph(s, cutoff=cutoff)


def _butane():
    """A rough (staggered-ish) butane geometry, types and bonds."""
    pos = [[0.0, 0.0, 0.0], [1.53, 0.0, 0.0], [2.05, 1.44, 0.0],
           [3.58, 1.44, 0.0],
           [-0.4, -0.5, 0.9], [-0.4, -0.5, -0.9], [-0.4, 1.0, 0.0],
           [1.93, -0.52, 0.88], [1.93, -0.52, -0.88],
           [1.65, 1.96, -0.88], [1.65, 1.96, 0.88],
           [3.98, 0.44, 0.0], [3.98, 1.96, 0.88], [3.98, 1.96, -0.88]]
    z = [6, 6, 6, 6] + [1] * 10
    types = ["opls_80", "opls_81", "opls_81", "opls_80"] + ["opls_85"] * 10
    bonds = [(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (0, 6), (1, 7), (1, 8),
             (2, 9), (2, 10), (3, 11), (3, 12), (3, 13)]
    return pos, z, types, bonds


def _butane_model(**kwargs):
    """An OPLS-AA butane model plus its graph builder."""
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    model = OPLS("oplsaa", top, cutoff=kwargs.pop("cutoff", 20.0), **kwargs)
    return model, pos, z


def _chain4(phi_deg, r=1.529, ang_deg=112.7):
    """Four collinear-free atoms with a prescribed dihedral angle."""
    ang = math.radians(ang_deg)
    phi = math.radians(phi_deg)
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([r, 0.0, 0.0])
    p2 = p1 + r * torch.tensor([-math.cos(ang), math.sin(ang), 0.0])
    e2 = (p2 - p1) / (p2 - p1).norm()
    n = torch.linalg.cross(p1 - p0, p2 - p1)
    n = n / n.norm()
    m = torch.linalg.cross(n, e2)
    p3 = p2 - r * math.cos(ang) * e2 \
        + r * math.sin(ang) * (math.cos(phi) * m + math.sin(phi) * n)
    return torch.stack([p0, p1, p2, p3])


def _ethylene(pyramid_deg=0.0, explicit_impropers=True):
    """Ethylene with one H rotated out of plane by ``pyramid_deg``."""
    d, dh = 1.34, 1.08
    ang = math.radians(120.0)
    phi = math.radians(pyramid_deg)
    pos = [[0.0, 0.0, 0.0], [d, 0.0, 0.0]]
    pos += [[-dh * math.cos(math.pi - ang), dh * math.sin(math.pi - ang), 0],
            [-dh * math.cos(math.pi - ang), -dh * math.sin(math.pi - ang), 0]]
    y = dh * math.sin(math.pi - ang)
    pos += [[d + dh * math.cos(math.pi - ang),
             y * math.cos(phi), y * math.sin(phi)],
            [d + dh * math.cos(math.pi - ang), -y, 0.0]]
    types = ["opls_88", "opls_88"] + ["opls_89"] * 4
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    # impropers resolve by the classes of their atoms (center third)
    impropers = [(2, 3, 0, 1), (4, 5, 1, 0)] if explicit_impropers else ()
    top = MolecularTopology.from_bonds(types, bonds, impropers=impropers)
    return pos, [6, 6, 1, 1, 1, 1], top


# ----------------------------------------------------------------------
# registration, topology derivation, libraries
# ----------------------------------------------------------------------
def test_registration():
    assert "opls" in available_models()


def test_topology_derivation_counts():
    # butane bookkeeping matches the 1996 paper's own counting: 27 dihedrals
    # (1 CCCC + 10 HCCC + 16 HCCH)
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    assert len(top.bonds) == 13
    assert len(top.angles) == 24
    assert len(top.dihedrals) == 27
    assert len(top.pairs14) == 27
    assert len(top.exclusions) == 13 + 24
    # 2-methyl-2-propanol has 30 dihedrals (paper, "Results"): 9 H-C-C-O,
    # 18 H-C-C-C and 3 H-O-C-C
    types2 = (["opls_101"] + ["opls_80"] * 3 + ["opls_96"]
              + ["opls_85"] * 9 + ["opls_97"])
    bonds2 = [(0, 1), (0, 2), (0, 3), (0, 4),
              (1, 5), (1, 6), (1, 7), (2, 8), (2, 9), (2, 10),
              (3, 11), (3, 12), (3, 13), (4, 14)]
    top2 = MolecularTopology.from_bonds(types2, bonds2)
    assert len(top2.dihedrals) == 30
    # guessing bonds from the geometry gives the hand-written list
    assert guess_bonds(pos, z) == sorted(bonds)


def test_topology_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        MolecularTopology.from_bonds(["a", "b"], [(0, 0)])
    with pytest.raises(ValueError):
        MolecularTopology.from_bonds(["a", "b"], [(0, 1), (1, 0)])
    with pytest.raises(ValueError):
        MolecularTopology.from_bonds(["a", "b"], [(0, 2)])
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    top.save(tmp_path / "butane.json")
    back = read_topology(tmp_path / "butane.json")
    assert back.types == top.types
    assert back.bonds == top.bonds
    assert back.dihedrals == top.dihedrals
    assert back.pairs14 == top.pairs14


def test_shipped_libraries_are_neutral_molecules():
    lib = builtin_library("oplsaa")
    q = {n: t["charge"] for n, t in lib.atom_types.items()}
    # butane: 2 CH3 + 2 CH2
    assert abs(2 * (q["opls_80"] + 3 * q["opls_85"])
               + 2 * (q["opls_81"] + 2 * q["opls_85"])) < 1e-12
    # ethanol: CH3 + CH2(O) + OH
    assert abs(q["opls_80"] + 3 * q["opls_85"] + q["opls_99"]
               + 2 * q["opls_85"] + q["opls_96"] + q["opls_97"]) < 1e-12
    # L-OPLS pentadecane-style CH2/CH3 groups are neutral too
    lq = {n: t["charge"]
          for n, t in builtin_library("lopls").atom_types.items()}
    assert abs(lq["lopls_CT_CH3"] + 3 * lq["lopls_HC_CH3"]) < 1e-12
    assert abs(lq["lopls_CT_CH2"] + 2 * lq["lopls_HC_CH2"]) < 1e-12
    # the classes behind the alkane types
    assert lib.cls("opls_80") == CT and lib.cls("opls_80", "oop") == CT
    assert lib.cls("opls_85", "torsion") == HC
    assert lib.atom_types["opls_80"]["element"] == 6
    assert lib.atom_types["opls_80"]["cls_nonbond"] == "opls_80"
    assert len(lib.templates) == 572 and lib.metadata["ff_form"] == "oplsaa"


def test_rb_fourier_conversion_against_lopls_table():
    # Siu et al. 2012, Table 2 lists the same hexane torsion in both forms
    rb = [0.518787, -0.230192, 0.896807, -1.49134, 0.0, 0.0]
    fourier = [-0.305938, 2.697394, -0.896807, 0.74567, 0.0]
    got = rb_to_fourier(rb)
    assert max(abs(a - b) for a, b in zip(got, fourier)) < 1e-6
    back = fourier_to_rb(fourier)
    assert max(abs(a - b) for a, b in zip(back, rb)) < 1e-6
    with pytest.raises(ValueError):
        rb_to_fourier([0, 0, 0, 0, 0, 1.0])
    # the shipped lopls.frc carries the same torsion (kJ -> kcal, V0 dropped)
    v = builtin_library("lopls").dihedral_types[f"{CT}-{CT}-{CT}-{CT}"]["v"]
    assert v[0] == 0.0
    assert max(abs(a - b / 4.184) for a, b in zip(v[1:4], fourier[1:4])) < 1e-9


def test_library_json_roundtrip(tmp_path):
    lib = builtin_library("lopls")
    lib.save(tmp_path / "lopls.json")
    back = read_opls(tmp_path / "lopls.json")
    assert back.atom_types == lib.atom_types
    assert back.dihedral_types == lib.dihedral_types
    assert back.templates == lib.templates
    assert back.fudge_lj == lib.fudge_lj
    # energies are identical through a round trip
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    g = _graph(pos, z, 20.0)
    e1 = OPLS(lib, top, cutoff=20.0)(g)["energy"]
    e2 = OPLS(back, top, cutoff=20.0)(g)["energy"]
    assert float((e1 - e2).abs()) < 1e-12


def test_frc_library_roundtrip_and_spec_forms(tmp_path):
    lib = builtin_library("oplsaa")
    path = lib.save_frc(tmp_path / "mine.frc", name="mine")
    back = read_opls(str(path))
    assert back.name == "mine"
    assert back.bond_types == lib.bond_types
    assert back.angle_types == lib.angle_types
    assert back.dihedral_types == lib.dihedral_types
    assert back.improper_types == lib.improper_types
    assert back.templates == lib.templates
    for n, a in lib.atom_types.items():
        b = back.atom_types[n]
        for key in ("charge", "sigma", "epsilon", "mass"):
            assert b[key] == pytest.approx(a[key], abs=1e-9), (n, key)
        for term in ("nonbond", "bond", "angle", "torsion", "oop"):
            assert back.cls(n, term) == lib.cls(n, term)
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    g = _graph(pos, z, 20.0)
    assert float((OPLS(lib, top, cutoff=20.0)(g)["energy"]
                  - OPLS(str(path), top, cutoff=20.0)(g)["energy"]).abs()) \
        < 1e-12
    # "<path>.frc:<variant>" and shipped names are the same thing
    from xnns.ffnn.common import builtin_data_dir
    spec = f"{builtin_data_dir() / 'oplsaa.frc'}:oplsaa"
    assert read_opls(spec).bond_types == lib.bond_types
    with pytest.raises(FileNotFoundError):
        read_opls("no-such-library")


def test_strict_refuses_unimplemented_forms():
    # CL&P carries a tabulated PF6- angle the OPLS model has no term for
    with pytest.raises(ValueError, match="tabulated_angle"):
        builtin_library("CL&P")
    clp = builtin_library("CL&P", strict=False)
    assert any("tabulated_angle" in n for n in clp.notes)
    # per-term equivalences differ for the CL&P types
    assert clp.cls("CE", "bond") == CT and clp.cls("CE", "oop") == "C2"
    assert clp.cls("FB", "nonbond") == "FB" and clp.cls("FB", "bond") == "F"
    plus = builtin_library("oplsaa+", strict=False)
    assert len(plus.atom_types) > len(builtin_library("oplsaa").atom_types)
    assert plus.fragments


# ----------------------------------------------------------------------
# equation-level references
# ----------------------------------------------------------------------
def test_bond_energy_equation():
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(["opls_80", "opls_80"], [(0, 1)])
    model = OPLS(lib, top, cutoff=10.0, keep_intermediates=True)
    r = 1.6
    out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 6], 10.0))
    expect = 268.0 * KCAL_TO_EV * (r - 1.529) ** 2
    assert abs(float(out["e_bond"]) - expect) < 1e-10
    # the bonded pair is excluded from every nonbonded term
    for key in ("e_lj", "e_coulomb", "e_lj14", "e_coulomb14", "e_angle",
                "e_torsion", "e_improper"):
        assert abs(float(out[key])) < 1e-14
    assert abs(float(out["energy"]) - expect) < 1e-10


def test_lj_coulomb_dimer_analytic():
    # two atoms with no bond: the full eq 1 with geometric combining rules
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(["opls_80", "opls_85"], [])
    model = OPLS(lib, top, cutoff=10.0)
    r = 3.2
    out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 1], 10.0))
    sig = math.sqrt(3.5 * 2.5)
    eps = math.sqrt(0.066 * 0.03) * KCAL_TO_EV
    lj = 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
    coul = KE * (-0.18) * 0.06 / r
    assert abs(float(out["e_lj"]) - lj) < 1e-10
    assert abs(float(out["e_coulomb"]) - coul) < 1e-10
    assert abs(float(out["energy"]) - (lj + coul)) < 1e-10


def test_angle_energy_equation():
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(
        ["opls_80", "opls_81", "opls_80"], [(0, 1), (1, 2)])
    model = OPLS(lib, top, cutoff=10.0, keep_intermediates=True)
    theta = math.radians(100.0)
    r = 1.529
    pos = [[r, 0, 0], [0, 0, 0],
           [r * math.cos(theta), r * math.sin(theta), 0]]
    out = model(_graph(pos, [6, 6, 6], 10.0))
    expect = 58.35 * KCAL_TO_EV * (theta - math.radians(112.7)) ** 2
    assert abs(float(out["e_angle"]) - expect) < 1e-10
    assert abs(float(model.intermediates["angle_theta"][0]) - theta) < 1e-10
    # 1,2 and 1,3 pairs are all excluded: no nonbonded energy at all
    assert abs(float(out["e_lj"]) + float(out["e_coulomb"])) < 1e-14


def test_torsion_fourier_and_14_scaling():
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(["opls_80"] * 4,
                                       [(0, 1), (1, 2), (2, 3)])
    model = OPLS(lib, top, cutoff=20.0, keep_intermediates=True)
    for phi in (0.0, 60.0, 100.0, 180.0):
        pos = _chain4(phi)
        out = model(_graph(pos, [6] * 4, 20.0))
        p = math.radians(phi)
        v = (0.0, 1.3, -0.05, 0.2, 0.0)
        e_t = KCAL_TO_EV * (v[0] + 0.5 * (
            v[1] * (1 + math.cos(p)) + v[2] * (1 - math.cos(2 * p))
            + v[3] * (1 + math.cos(3 * p)) + v[4] * (1 - math.cos(4 * p))))
        assert abs(float(out["e_torsion"]) - e_t) < 1e-10
        # the 1,4 pair is scaled by 1/2 and evaluated exactly
        r14 = float((pos[3] - pos[0]).norm())
        lj = 4.0 * 0.066 * KCAL_TO_EV * ((3.5 / r14) ** 12 - (3.5 / r14) ** 6)
        coul = KE * (-0.18) ** 2 / r14
        assert abs(float(out["e_lj14"]) - 0.5 * lj) < 1e-10
        assert abs(float(out["e_coulomb14"]) - 0.5 * coul) < 1e-10
        # and does not appear in the plain nonbonded terms
        assert abs(float(out["e_lj"]) + float(out["e_coulomb"])) < 1e-14


def test_improper_dihedral_equation():
    lib = builtin_library("oplsaa")
    pos, z, top = _ethylene(0.0)
    model = OPLS(lib, top, cutoff=20.0, keep_intermediates=True)
    assert model.impropers == [(2, 3, 0, 1), (4, 5, 1, 0)]
    out = model(_graph(pos, z, 20.0))
    assert abs(float(out["e_improper"])) < 1e-10   # planar: both minima
    # planar ethylene also has zero X-CM-CM-X torsional energy
    assert abs(float(out["e_torsion"])) < 1e-10
    pos2, z, top = _ethylene(25.0)
    out2 = model(_graph(pos2, z, 20.0))
    # e_improper = sum V2/2 (1 - cos 2 phi) over the distorted center's
    # improper, recomputed from the model's own dihedral cosines; V2 = 30
    # is the library's X-X-opls_86-X pattern (alkene carbon center)
    cos = model.intermediates["improper_cos"]
    expect = sum(0.5 * 30.0 * KCAL_TO_EV * (1 - (2 * float(c) ** 2 - 1))
                 for c in cos)
    assert abs(float(out2["e_improper"]) - expect) < 1e-12
    assert float(out2["e_improper"]) > 1e-4


def test_auto_impropers_at_trigonal_centers():
    # no impropers listed: one is placed at every three-connected atom whose
    # classes match a library pattern (both ethylene carbons here), with the
    # outer atoms in index order and the center third
    pos, z, top = _ethylene(25.0, explicit_impropers=False)
    auto = OPLS("oplsaa", top, cutoff=20.0)
    assert auto.impropers == [(1, 2, 0, 3), (0, 4, 1, 5)]
    assert float(auto(_graph(pos, z, 20.0))["e_improper"]) > 1e-4
    off = OPLS("oplsaa", top, cutoff=20.0, auto_impropers=False)
    assert off.impropers == []
    assert abs(float(off(_graph(pos, z, 20.0))["e_improper"])) < 1e-14
    # a center without any pattern (sp3 carbons have none) gets nothing
    pos_b, z_b, types, bonds = _butane()
    assert OPLS("oplsaa", MolecularTopology.from_bonds(types, bonds),
                cutoff=20.0).impropers == []
    # legacy opaque keys still resolve exactly
    lib = builtin_library("oplsaa")
    lib.improper_types["my-key"] = {"v2": 1.0}
    top_k = MolecularTopology.from_bonds(
        top.types, top.bonds, impropers=[(2, 3, 0, 1)], improper_keys=["my-key"])
    m = OPLS(lib, top_k, cutoff=20.0)
    assert m.impropers == [(2, 3, 0, 1)]
    assert m.ff.improper_keys[m.ff.resolve_improper("my-key")] == "my-key"


def test_improper_pattern_precedence():
    table = {improper_key("X", "X", CM, "X"): {"v2": 30.0},
             improper_key(HC, "X", CM, "X"): {"v2": 5.0},
             improper_key(HC, HC, CM, CT): {"v2": 1.0}}
    assert resolve_improper_type(table, HC, HC, CM, CT) \
        == improper_key(HC, HC, CM, CT)
    assert resolve_improper_type(table, CT, HC, CM, HC) \
        == improper_key(HC, HC, CM, CT)            # outer order is free
    assert resolve_improper_type(table, HC, CT, CM, CT) \
        == improper_key(HC, "X", CM, "X")
    assert resolve_improper_type(table, CT, CT, CM, CT) \
        == improper_key("X", "X", CM, "X")
    assert resolve_improper_type(table, CT, CT, CT, CT) is None


def test_dihedral_wildcard_resolution():
    from xnns.ffnn.models.oplslib import resolve_dihedral_type
    table = {"X-CM-CM-X": {}, "CT-CM-CM-CT": {}}
    assert resolve_dihedral_type(table, "CT", "CM", "CM", "CT") \
        == "CT-CM-CM-CT"
    assert resolve_dihedral_type(table, "HC", "CM", "CM", "CT") \
        == "X-CM-CM-X"
    assert resolve_dihedral_type(table, "CT", "CT", "CT", "CT") is None


def test_missing_parameters_are_reported_together():
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(
        ["opls_97", "opls_97"], [(0, 1)])   # H(O)-H(O) bond: no such type
    with pytest.raises(KeyError) as err:
        OPLS(lib, top, cutoff=10.0)
    assert f"{HO}-{HO}" in str(err.value)
    with pytest.raises(KeyError) as err:
        OPLS(lib, MolecularTopology.from_bonds(["nope"], []), cutoff=10.0)
    assert "nope" in str(err.value)


# ----------------------------------------------------------------------
# invariances, forces, batching, periodicity
# ----------------------------------------------------------------------
def test_rotation_translation_invariance_and_force_equivariance():
    model, pos, z = _butane_model()
    fmodel = ForceStressOutput(model)
    g = _graph(pos, z, 20.0)
    out = fmodel(g)
    e0, f0 = out["energy"], out["forces"]
    ang = 0.6
    R = torch.tensor([[math.cos(ang), -math.sin(ang), 0.0],
                      [math.sin(ang), math.cos(ang), 0.0],
                      [0.0, 0.0, 1.0]])
    pos2 = torch.as_tensor(pos) @ R.T + torch.tensor([3.0, -2.0, 1.0])
    out2 = fmodel(_graph(pos2, z, 20.0))
    assert float((out2["energy"] - e0).abs()) < 1e-9
    assert float((out2["forces"] - f0 @ R.T).abs().max()) < 1e-8


def test_forces_match_finite_differences():
    model, pos, z = _butane_model()
    fmodel = ForceStressOutput(model)
    forces = fmodel(_graph(pos, z, 20.0))["forces"]
    d = 1e-5
    for atom, comp in ((0, 0), (2, 1), (7, 2), (13, 0)):
        pp = [list(p) for p in pos]
        pp[atom][comp] += d
        ep = float(model(_graph(pp, z, 20.0))["energy"])
        pp[atom][comp] -= 2 * d
        em = float(model(_graph(pp, z, 20.0))["energy"])
        assert abs(-(ep - em) / (2 * d) - float(forces[atom, comp])) < 1e-6


def test_batching_matches_individual_evaluation():
    model, pos, z = _butane_model()
    torch.manual_seed(0)
    graphs, singles = [], []
    for _ in range(3):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        g = _graph(p, z, 20.0)
        graphs.append(g)
        singles.append(model(g))
    batched = model(collate(graphs))
    for key in ("energy", "e_bond", "e_angle", "e_torsion", "e_lj",
                "e_coulomb", "e_lj14", "e_coulomb14"):
        stacked = torch.cat([s[key] for s in singles])
        assert float((batched[key] - stacked).abs().max()) < 1e-10
    assert float((batched["node_energy"].sum(0)
                  - batched["energy"].sum()).abs()) < 1e-10


def test_size_extensivity_with_replicated_topology():
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    single = OPLS("oplsaa", top, cutoff=12.0)
    double = OPLS("oplsaa", top.replicate(2), cutoff=12.0)
    e1 = float(single(_graph(pos, z, 12.0))["energy"])
    pos2 = torch.cat([torch.as_tensor(pos),
                      torch.as_tensor(pos) + torch.tensor([50.0, 0.0, 0.0])])
    e2 = float(double(_graph(pos2, z + z, 12.0))["energy"])
    assert abs(e2 - 2 * e1) < 1e-9


def test_topology_mismatch_raises():
    model, pos, z = _butane_model()
    with pytest.raises(ValueError):
        model(_graph(pos[:-1], z[:-1], 20.0))
    z2 = list(z)
    z2[0] = 8
    with pytest.raises(ValueError):
        model(_graph(pos, z2, 20.0))


def test_periodic_minimum_image_and_stress():
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    model = OPLS("oplsaa", top, cutoff=8.0)
    cell = 20.0 * torch.eye(3)
    e0 = float(model(_graph(pos, z, 8.0, cell))["energy"])
    # translating one atom by a lattice vector changes nothing: the bonded
    # terms use minimum-image displacements
    pos2 = torch.as_tensor(pos).clone()
    pos2[3] += torch.tensor([20.0, 0.0, 0.0])
    e1 = float(model(_graph(pos2, z, 8.0, cell))["energy"])
    assert abs(e1 - e0) < 1e-9
    # and the whole molecule sliding across the boundary changes nothing
    pos3 = torch.as_tensor(pos) + torch.tensor([19.0, 0.0, 0.0])
    e2 = float(model(_graph(pos3, z, 8.0, cell))["energy"])
    assert abs(e2 - e0) < 1e-9
    out = ForceStressOutput(model, compute_stress=True)(
        _graph(pos, z, 8.0, cell))
    assert bool(torch.isfinite(out["stress"]).all())


def test_switching_function():
    lib = builtin_library("oplsaa")
    top = MolecularTopology.from_bonds(["opls_80", "opls_80"], [])
    plain = OPLS(lib, top, cutoff=10.0)
    switched = OPLS(lib, top, cutoff=10.0, switch_width=2.0)
    r = 9.0
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], 10.0)
    e_p = float(plain(g)["energy"])
    e_s = float(switched(g)["energy"])
    x = (r - 8.0) / 2.0
    s = 1 - x ** 3 * (10 - 15 * x + 6 * x * x)
    assert abs(e_s - s * e_p) < 1e-12
    # inside the switching onset both models agree
    g2 = _graph([[0, 0, 0], [4.0, 0, 0]], [6, 6], 10.0)
    assert abs(float(plain(g2)["energy"])
               - float(switched(g2)["energy"])) < 1e-14
    with pytest.raises(ValueError):
        OPLS(lib, top, cutoff=10.0, switch_width=10.0)


# ----------------------------------------------------------------------
# fidelity anchor: the 1996 paper's ethane barrier
# ----------------------------------------------------------------------
def test_relaxed_ethane_barrier_matches_paper():
    ase = pytest.importorskip("ase")
    from ase.constraints import FixInternals
    from ase.optimize import BFGS
    from xnns.common.deploy import XNNSCalculator

    d, dh = 1.529, 1.09
    ang = math.radians(110.7)
    pos = [[0.0, 0.0, 0.0], [d, 0.0, 0.0]]
    for base, sign, off in ((0.0, -1.0, 60.0), (d, 1.0, 0.0)):
        for k in range(3):
            phi = math.radians(off + 120.0 * k)
            pos.append([base - sign * dh * math.cos(math.pi - ang),
                        dh * math.sin(math.pi - ang) * math.cos(phi),
                        dh * math.sin(math.pi - ang) * math.sin(phi)])
    atoms = ase.Atoms(numbers=[6, 6] + [1] * 6, positions=pos)
    types = ["opls_80"] * 2 + ["opls_85"] * 6
    top = MolecularTopology.from_bonds(types, guess_bonds(pos, [6, 6] + [1] * 6))
    model = OPLS("oplsaa-1996", top, cutoff=30.0)
    energies = {}
    for target in (60.0, 0.0):
        at = atoms.copy()
        at.calc = XNNSCalculator(ForceStressOutput(model),
                                 cutoff=model.cutoff)
        at.set_dihedral(2, 0, 1, 5, target, indices=[5, 6, 7])
        at.set_constraint(
            FixInternals(dihedrals_deg=[[target, [2, 0, 1, 5]]]))
        BFGS(at, logfile=None).run(fmax=1e-5, steps=500)
        energies[target] = at.get_potential_energy() * EV_TO_KCAL
    barrier = energies[0.0] - energies[60.0]
    assert abs(barrier - 3.01) < 0.02   # Table 1: 3.01 kcal/mol
    # oplsaa-1996 restores the paper's alcohol torsion too (H-C-O-H V3 = 0.45)
    l96 = builtin_library("oplsaa-1996")
    assert l96.dihedral_types[f"{HC}-{CT}-{OH}-{HO}"]["v"][3] == 0.45
    assert builtin_library("oplsaa").dihedral_types[
        f"{HC}-{CT}-{OH}-{HO}"]["v"][3] == 0.352


def test_openmm_parity():
    # independent verification: the same library + topology evaluated
    # through OpenMM (Coulomb via NonbondedForce, LJ via a
    # CustomNonbondedForce with OPLS geometric mixing, exact 1,4
    # exceptions, Fourier torsions as phased periodic torsions)
    mm = pytest.importorskip("openmm")
    import numpy as np
    import openmm.unit as u
    from xnns.ffnn.models.oplslib import (resolve_angle_type,
                                          resolve_bond_type,
                                          resolve_dihedral_type)

    lib = builtin_library("oplsaa")
    at = lib.atom_types
    pos0, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    cls = [at[n]["cls"] for n in types]

    system = mm.System()
    for name in types:
        system.addParticle(at[name]["mass"])
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    lj = mm.CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6); "
        "sig=sqrt(sig1*sig2); eps=sqrt(eps1*eps2)")
    lj.addPerParticleParameter("sig")
    lj.addPerParticleParameter("eps")
    for name in types:
        nb.addParticle(at[name]["charge"], 0.1, 0.0)
        lj.addParticle([max(at[name]["sigma"], 1e-6) * 0.1,
                        at[name]["epsilon"] * 4.184])
    for i, j in top.exclusions + top.pairs14:
        nb.addException(i, j, 0.0, 0.1, 0.0)
        lj.addExclusion(i, j)
    for i, j in top.pairs14:
        ti, tj = at[types[i]], at[types[j]]
        nb.addException(i, j, 0.5 * ti["charge"] * tj["charge"],
                        math.sqrt(ti["sigma"] * tj["sigma"]) * 0.1,
                        0.5 * math.sqrt(ti["epsilon"] * tj["epsilon"])
                        * 4.184, replace=True)
    system.addForce(nb)
    system.addForce(lj)
    bond = mm.HarmonicBondForce()
    for i, j in top.bonds:
        bt = lib.bond_types[resolve_bond_type(lib.bond_types,
                                              cls[i], cls[j])]
        bond.addBond(i, j, bt["r0"] * 0.1, 2 * bt["k"] * 4.184 * 100)
    system.addForce(bond)
    ang = mm.HarmonicAngleForce()
    for i, j, k in top.angles:
        a = lib.angle_types[resolve_angle_type(lib.angle_types, cls[i],
                                               cls[j], cls[k])]
        ang.addAngle(i, j, k, math.radians(a["theta0"]), 2 * a["k"] * 4.184)
    system.addForce(ang)
    tors = mm.PeriodicTorsionForce()
    for i, j, k, l in top.dihedrals:
        key = resolve_dihedral_type(lib.dihedral_types, cls[i], cls[j],
                                    cls[k], cls[l])
        v = lib.dihedral_types[key]["v"]
        for n_, (vn, ph) in enumerate(zip(v[1:],
                                          (0.0, math.pi, 0.0, math.pi)), 1):
            if vn:
                tors.addTorsion(i, j, k, l, n_, ph, 0.5 * vn * 4.184)
    system.addForce(tors)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3),
                     mm.Platform.getPlatformByName("Reference"))

    model = ForceStressOutput(OPLS(lib, top, cutoff=100.0))
    ev_to_kj = 96.48533212331
    rng = np.random.default_rng(1)
    for _ in range(3):
        pos = np.asarray(pos0) + 0.1 * rng.standard_normal((len(z), 3))
        out = model(_graph(pos, z, 100.0))
        ctx.setPositions(pos * 0.1)
        st = ctx.getState(getEnergy=True, getForces=True)
        e_omm = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        f_omm = st.getForces(asNumpy=True).value_in_unit(
            u.kilojoule_per_mole / u.nanometer)
        assert abs(float(out["energy"]) * ev_to_kj - e_omm) < 1e-6
        f_x = out["forces"].detach().numpy() * ev_to_kj * 10.0
        assert np.abs(f_x - f_omm).max() < 1e-5


# ----------------------------------------------------------------------
# SMARTS typing entry points (RDKit)
# ----------------------------------------------------------------------
def test_from_atoms_reproduces_hand_typed_model():
    pytest.importorskip("rdkit")
    pos, z, types, bonds = _butane()
    hand = OPLS("oplsaa", MolecularTopology.from_bonds(types, bonds), cutoff=20.0)
    auto = OPLS.from_atoms((pos, z), "oplsaa", cutoff=20.0)
    assert auto.topology.types == types
    assert auto.topology.bonds == hand.topology.bonds
    g = _graph(pos, z, 20.0)
    assert float((auto(g)["energy"] - hand(g)["energy"]).abs()) < 1e-12
    # L-OPLS retypes the hydrocarbon: charges and the C-C-C-C torsion change,
    # bonds do not. (The fixture is an exact anti conformer, where every OPLS
    # torsion is zero, so twist the last carbon out of plane first.)
    lo = OPLS.from_atoms((pos, z), "lopls", cutoff=20.0)
    assert set(lo.topology.types) == {"lopls_CT_CH3", "lopls_CT_CH2",
                                      "lopls_HC_CH3", "lopls_HC_CH2"}
    twisted = [list(p) for p in pos]
    twisted[3][2] += 0.8
    gt = _graph(twisted, z, 20.0)
    assert abs(float(lo(gt)["e_torsion"]) - float(hand(gt)["e_torsion"])) > 1e-4
    assert abs(float(lo(gt)["e_coulomb"]) - float(hand(gt)["e_coulomb"])) > 1e-4
    assert abs(float(lo(gt)["e_bond"]) - float(hand(gt)["e_bond"])) < 1e-12


# ----------------------------------------------------------------------
# training, shared parameters, config
# ----------------------------------------------------------------------
def test_trainable_selection_and_gradients():
    model, pos, z = _butane_model(trainable=("dihedral_v", "charge"))
    out = model(_graph(pos, z, 20.0))
    out["energy"].pow(2).sum().backward()
    P = model.ff.params
    assert P["dihedral_v"].requires_grad and P["dihedral_v"].grad is not None
    assert P["charge"].requires_grad and P["charge"].grad is not None
    assert not P["bond_k"].requires_grad
    assert not P["sigma"].requires_grad
    frozen, _, _ = _butane_model()
    assert not any(p.requires_grad for p in frozen.parameters())
    every, _, _ = _butane_model(trainable="all")
    assert all(p.requires_grad for p in every.ff.params.values())
    with pytest.raises(ValueError):
        _butane_model(trainable=("not_a_group",))


def test_training_step_reduces_loss():
    # refit the alkane torsions of one library against energies produced by
    # another (the L-OPLS-style task of Siu et al. 2012)
    torch.manual_seed(0)
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    target_model = OPLS("oplsaa-1996", top, cutoff=20.0)
    model = OPLS("oplsaa", top, cutoff=20.0, trainable=("dihedral_v",))
    graphs = [_graph(torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3),
                     z, 20.0) for _ in range(8)]
    batch = collate(graphs)
    target = target_model(batch)["energy"].detach()
    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (model(batch)["energy"] - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_shared_forcefield_accumulates_gradients():
    ff = OPLSForceField(builtin_library("oplsaa"), trainable=("dihedral_v",))
    pos, z, types, bonds = _butane()
    butane = OPLS(ff, MolecularTopology.from_bonds(types, bonds),
                  cutoff=20.0)
    eth_types = ["opls_80"] * 2 + ["opls_85"] * 6
    eth_bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    eth_pos = [[0, 0, 0], [1.53, 0, 0],
               [-0.4, 1.0, 0], [-0.4, -0.5, 0.9], [-0.4, -0.5, -0.9],
               [1.93, -1.0, 0], [1.93, 0.5, 0.9], [1.93, 0.5, -0.9]]
    ethane = OPLS(ff, MolecularTopology.from_bonds(eth_types, eth_bonds),
                  cutoff=20.0)
    assert butane.ff is ethane.ff
    loss = butane(_graph(pos, z, 20.0))["energy"].pow(2).sum()
    loss.backward()
    g1 = ff.params["dihedral_v"].grad.clone()
    loss2 = ethane(_graph(eth_pos, [6, 6] + [1] * 6, 20.0))["energy"] \
        .pow(2).sum()
    loss2.backward()
    assert float((ff.params["dihedral_v"].grad - g1).abs().max()) > 0.0


def test_export_library_roundtrip(tmp_path):
    model, pos, z = _butane_model()
    lib = model.export_library()
    assert abs(lib.bond_types[f"{CT}-{CT}"]["k"] - 268.0) < 1e-9
    assert abs(lib.angle_types[f"{CT}-{CT}-{CT}"]["theta0"] - 112.7) < 1e-9
    assert abs(lib.dihedral_types[f"{CT}-{CT}-{CT}-{CT}"]["v"][1] - 1.3) < 1e-9
    assert lib.templates == builtin_library("oplsaa").templates
    pos_t, z_t, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    e1 = float(model(_graph(pos, z, 20.0))["energy"])
    e2 = float(OPLS(lib, top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e2) < 1e-12
    # ... and through a .frc file
    path = lib.save_frc(tmp_path / "exported.frc")
    e3 = float(OPLS(str(path), top, cutoff=20.0)(_graph(pos, z, 20.0))["energy"])
    assert abs(e1 - e3) < 1e-12


def test_masses_property():
    model, pos, z = _butane_model()
    m = model.masses
    assert abs(float(m.sum()) - (4 * 12.011 + 10 * 1.008)) < 1e-9


def test_from_config_and_key_translation(tmp_path):
    from xnns.common.config import from_dict
    from xnns.common.models import build_model
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    top.save(tmp_path / "top.json")
    lib = builtin_library("oplsaa")
    lib.save(tmp_path / "lib.json")
    cfg = from_dict({"model": {
        "name": "opls", "cutoff": 12.0,
        "parameter_file": str(tmp_path / "lib.json"),   # translated key
        "topology": str(tmp_path / "top.json"),
        "fudgeQQ": 0.5,                                 # translated key
        "trainable": ["dihedral_v"],
    }})
    model = build_model(cfg.model)
    assert isinstance(model, OPLS)
    assert model.cutoff == 12.0
    assert model.ff.params["dihedral_v"].requires_grad
    e = float(model(_graph(pos, z, 12.0))["energy"])
    direct = float(OPLS("oplsaa", top, cutoff=12.0)(
        _graph(pos, z, 12.0))["energy"])
    assert abs(e - direct) < 1e-12
    # inline types + bonds also work, and "frc" is a spelling of "library"
    cfg2 = from_dict({"model": {
        "name": "opls", "cutoff": 12.0, "frc": "oplsaa",
        "types": types, "bonds": [list(b) for b in bonds]}})
    assert abs(float(build_model(cfg2.model)(
        _graph(pos, z, 12.0))["energy"]) - direct) < 1e-12
    with pytest.raises(ValueError):
        build_model(from_dict({"model": {"name": "opls"}}).model)


def test_dataset_training_smoke():
    torch.manual_seed(0)
    pos, z, types, bonds = _butane()
    top = MolecularTopology.from_bonds(types, bonds)
    model = OPLS("oplsaa", top, cutoff=12.0, trainable=("dihedral_v",))
    fmodel = ForceStressOutput(model)
    fmodel.train()
    structures = []
    for _ in range(4):
        p = torch.as_tensor(pos) + 0.05 * torch.randn(len(pos), 3)
        structures.append({
            "pos": p, "atomic_numbers": torch.tensor(z),
            "energy": torch.randn(()), "forces": torch.randn(len(pos), 3)})
    ds = AtomicDataset(structures, cutoff=12.0)
    batch = collate([ds[i] for i in range(len(ds))])
    out = fmodel(batch)
    loss, logs = weighted_loss(out, batch, energy_weight=1.0,
                               force_weight=0.1, stress_weight=0.0)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert model.ff.params["dihedral_v"].grad is not None
